#######################################################################
#Modified according to the experiment requirements
#Added Weights and Biases logging
#Import for different local datasets
#Append EOS token to all samples in datasets
#Original License below:
#######################################################################
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import pickle
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
import glob
from sft_trainer.sft_trainer_mod import SFTTrainer
import ast
from os import environ            # to interact with weights and biases
import einops

# This example fine-tunes Llama v2 model on Guanace dataset
# using QLoRA. At the end of the script we perform merging the weights
# Use it by correctly passing --model_name argument when running the
# script. 
#
# Versions used:
# accelerate == 0.21.0
# peft == 0.4.0
# bitsandbytes == 0.40.2
# transformers == 4.31.0

# For models that have `config.pretraining_tp > 1` install:
# pip install git+https://github.com/huggingface/transformers.git

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=64)                           # lora alpha
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64*4)                               # lora r-> alpha*4
    max_seq_length: Optional[int] = field(default=4096)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )

    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 8bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(                                      ####epoch
        default=6,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.01, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=True,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(                                                 ##################################### model_dir
        default="./results/noGuanaco7b64",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    train_eval_dir: str = field(
        default="./data/train_test_datasets/run_5_noGuanaco",                 ######################################### location of the train and eval dataset
        metadata={"help": "The directory of the train and eval datasets."},
    )

    wnb_project: str = field(                                                   #wnb
        default="LLama",
        metadata={"help": "Project name for weights and biases"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if  environ.get('LORA_ALPHA') is not None:    
    script_args.lora_alpha = int(environ.get('LORA_ALPHA'))
    print('updated lora alpha from ENV: '+str(script_args.lora_alpha))

if  environ.get('BITQ') is not None:    
    script_args.use_4bit = ast.literal_eval(environ.get('BITQ'))
    print('updated use 4 bit from ENV: '+str(script_args.use_4bit))

if  environ.get('TRAIN_EVAL_DIR') is not None:    
    script_args.train_eval_dir = environ.get('TRAIN_EVAL_DIR')
    print('updated train eval Dir from ENV: '+script_args.train_eval_dir)

if  environ.get('OUPUT_DIR') is not None:    
    script_args.output_dir = environ.get('OUPUT_DIR')
    print('updated output dir from ENV: '+script_args.output_dir)

if  environ.get('BASE_MODEL') is not None:    
    script_args.model_name = environ.get('BASE_MODEL')
    print('updated base model from ENV: '+script_args.model_name)

output_dir_exist = os.path.exists(script_args.output_dir)
if not output_dir_exist:

   os.makedirs(script_args.output_dir)

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        load_in_8bit=args.use_8bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"":0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        use_auth_token=True,
        trust_remote_code=True
    )
    
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1 

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


if  environ.get('WANDB_API_KEY') is not None:    

    import wandb

    os.environ['WANDB_PROJECT'] = script_args.wnb_project
    report_to = 'wandb'

else: 
    report_to = None
    print('WARNING: no weights and biases logging')
   
#with wandb.init(config=config):
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    eval_accumulation_steps = script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    #max_steps=script_args.max_steps,
    num_train_epochs = script_args.num_train_epochs,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    evaluation_strategy = 'steps',
    eval_steps = 500,
    report_to = report_to
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False

# write all parameters into a protocol to double check
protocol = [{k: v} for k, v in asdict(script_args).items()]


def load_ds_from_folder(folder,tokenizer,shorten = {},concat = True):

    def append_eos(ds):

        ds['text'] = ds['text']+tokenizer.eos_token

        return ds

    files = glob.glob(folder+"/*")
    ds = {}
    for file in files: 
        name = file.split('/')[-1].split('.')[0]
        test_eval = file.split('/')[-2]

        with open(file, 'rb') as handle:
            loaded_ds = pickle.load(handle).map(append_eos)

            if name in shorten.keys():
                loaded_ds = Dataset.from_pandas(pd.DataFrame(data = list(loaded_ds)[:shorten[name][test_eval]]))           # shorten a specific dataset 
            print('Dataset: '+test_eval+' '+name)
            print(loaded_ds)
            protocol.append({'Dataset: '+test_eval+' '+name : str(loaded_ds)})

            ds[name] = loaded_ds
        
    if concat:
        return concatenate_datasets(ds.values()).shuffle()
    else: 
        return ds

train_folder = script_args.train_eval_dir + '/train'
eval_folder = script_args.train_eval_dir + '/eval'

shorten_dict = {'CheungGuanaco':{'train':8000,'eval':600}}
train_ds = load_ds_from_folder(train_folder,tokenizer,shorten=shorten_dict)
test_ds = load_ds_from_folder(eval_folder,tokenizer,shorten=shorten_dict,concat=False)

print('Model and checkpoints will be saved at: ' +script_args.output_dir)

with open(script_args.output_dir+"/protocol.txt", "w") as file:
    for dictionary in protocol:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")
   
# Fix weird overflow issue with fp16 training
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)



 

