"""
@ Author: DopeorNope Lee(Seungyoo Lee)
@ Description: training LLM using the DPO Method
@ Date: 07.Aug.2023
@ Last Update: 21.Jun.2024

@ Notice:
Unauthorized copying and sharing of code without the author's permission are prohibited.

"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from trl import DPOTrainer
from huggingface_hub import login
import argparse





def get_args():

    """
    이 함수는 각 DPO로 훈련시키기 위해 중요한 인수인자를 sh 파일로 부터 받아와서 

    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--datapath", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps",type=int,default=100)
    parser.add_argument("--token_id", type =str, default='',help='please enter yout huggingface token ID')
    parser.add_argument("--weight_decay",type=float,default=0.05)
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--sanity_check", type=bool, default=False)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--ignore_bias_buffers", type=bool, default=False)
    parser.add_argument("--lora_target_modules",type=list, default = ['embed_tokens', 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head'] )
    
    return parser.parse_args()


# pair 한 데이터셋을 준비하는 함수입니다.
def paired_data_preparation(
    data_dir: str = "", #default
    sanity_check: bool = False,
    cache_dir: str = None,
    split_criteria: str = "train",
    num_proc: int=24,
) -> Dataset:
    """
    이 데이터셋은 이후 딕셔너리 형태로 변환되며  다음과 같은 형태로 prompt, chosen, reject로 담기게 됩니다.
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompt의 구조는 다음과 같이 담기게 됩니다(알파카 프롬프트):  
      "###질문: " + <prompt> + "\n\n###답변: "
    """

    dataset = load_dataset(data_dir, split=split_criteria ,cache_dir=cache_dir)
    
    
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    # 함수내에 정의되는 함수로, 각 데이터 포맷과 구조에 맞게 매핑해주는 역할을 진행합니다.
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["###질문: " + question + "\n\n###답변: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )





def main():
    args = get_args()


    print(
            f"################################ Training Arguments ################################"
            f"model_name_or_path: {args.model_name_or_path}\n"
            f"datapath: {args.datapath}\n"
            f"output_dir: {args.output_dir}\n"
            f"per_device_train_batch_size: {args.per_device_train_batch_size}\n"
            f"per_device_eval_batch_size: {args.per_device_eval_batch_size}\n"
            f"num_epochs: {args.num_epochs}\n"
            f"max_step: {args.max_step}\n"
            f"learning_rate: {args.learning_rate}\n"
            f"cutoff_len(max_length): {args.max_length}\n"
            f"lora_r: {args.lora_r}\n"
            f"lora_alpha: {args.lora_alpha}\n"
            f"lora_dropout: {args.lora_dropout}\n"
            f"lora_target_modules: {args.lora_target_modules}\n"
            f"max_prompt_length: {args.max_prompt_length}\n"
            f"##################################################################################\n"
            
    )

    login(token=args.token_id)


    # SFT MODEL
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,      
        device_map="auto")


    model.config.use_cache = False
    model.is_parallelizable = True
    model.model_parallel = True

    model.config.max_position_embeddings= args.max_prompt_length

    print("model's max_position_embeddings :",model.config.max_position_embeddings)


    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    #model.resize_token_embeddings(len(tokenizer))
    
    train_dataset = paired_data_preparation(data_dir= args.datapath, split_criteria= "train", sanity_check=args.sanity_check)
    
    train_dataset = train_dataset.filter(
    
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    
        and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)
    

    eval_dataset = paired_data_preparation(data_dir= args.datapath, split_criteria= "validation", sanity_check=True)
    
    eval_dataset = eval_dataset.filter(
    
       lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    
       and len(x["prompt"]) + len(x["rejected"]) <= args.max_length)
    


    training_args = TrainingArguments(
        num_train_epochs= args.num_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
       # max_steps = args.max_step,
        logging_steps = args.logging_steps,
        save_steps = args.save_steps,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        gradient_checkpointing = args.gradient_checkpointing,
        learning_rate = args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        output_dir = args.output_dir,
        #report_to = args.report_to,
        lr_scheduler_type = args.lr_scheduler_type,
        warmup_steps = args.warmup_steps,
        optim = args.optimizer_type,
        bf16 = True,
        remove_unused_columns = False,
        run_name = "dpo_llama2",
    )


    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        target_modules = args.lora_target_modules,
        bias = "none",
        task_type = "CAUSAL_LM")
    
        
    print("###################################################################################")    
    print("############################  MODEL was Lodaded in GPU ############################")
    print("###################################################################################")    

    dpo_trainer = DPOTrainer(
        model,
        ref_model = None,   # ref 모델을 None으로 놓게 되면 SFT + adapter가 붙은 모델에서 adapter를 떼고, policy에 따른 최적화를 진행하게 됩니다. 두개의 모델을 로드할 필요가 없어 메모리 이득을 꾀할 수 있습니다.
        args = training_args,
        beta = args.beta,
        train_dataset= train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        peft_config = peft_config,
        max_prompt_length = args.max_prompt_length,
        max_length = args.max_length,
    )

    print("###################################################################################")    
    print("########################  Trainin Process is preparing now  #######################")
    print("###################################################################################")    

    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)


if __name__ == "__main__" :
    main()