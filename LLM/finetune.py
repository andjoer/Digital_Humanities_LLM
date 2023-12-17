# coding=utf-8
#######################################################################
#Modified according to the experiment requirements
#Added Weights and Biases logging
#Import for different local datasets
#Append EOS token to all samples in datasets
#Borrows from The HuggingFace Inc. team. All rights reserved.
#######################################################################


import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import pickle
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import LoraConfig
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import glob
from sft_trainer.sft_trainer_mod import SFTTrainer, DataCollatorForCompletionOnlyLM
import ast
from os import environ            # to interact with weights and biases
import einops
from functools import partial

from utils import formatting_func_standard, formatting_func_chat

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
    lora_r: Optional[int] = field(default=512)                               # lora r-> alpha*4
    max_seq_length: Optional[int] = field(default=4096)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )

    use_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 8bit precision base model loading"},
    )
    use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Use Lora"},
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
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})    ### save_steps
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=True,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(                                                 ##################################### model_dir
        default="./results/LlamaChatOnlyBspSimp7b64",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    checkpoint: str = field(                                                 ##################################### checkpoint
        default="",
        metadata={"help": "The directory of a checkpoint to load."},
    )


    train_eval_dir: str = field(
        default="./data/train_test_datasets/run_6_onlybsp_simple",                 ######################################### location of the train and eval dataset
        metadata={"help": "The directory of the train and eval datasets."},
    )

    wnb_project: str = field(                                                   #wnb
        default="LLama",
        metadata={"help": "Project name for weights and biases"},
    )

    eval_collator: str = field(                                                   # eval data collator
        default="completion",
        metadata={"help": "Data collator for the evaluation dataset"},
    )

    train_collator: str = field(                                                   # train data collator
        default="all",
        metadata={"help": "Data collator for the training dataset"},
    )
    eval_steps: int = field(                                                   # train data collator
        default=100,
        metadata={"help": "Steps between the evaluations"},
    )

class ConfigManager:

    def __init__(self, arg_class: Any) -> None:
        self.parser = HfArgumentParser(arg_class)
        self.config = self.parser.parse_args_into_dataclasses()[0]
        self.env_mappings = {
            'LORA_ALPHA': 'lora_alpha',
            'LORA_R': 'lora_r',
            'BITQ': 'use_4bit',
            'TRAIN_EVAL_DIR': 'train_eval_dir',
            'OUPUT_DIR': 'output_dir',
            'BASE_MODEL': 'model_name',
            'BITH': 'use_8bit',
            'USE_LORA': 'use_lora',
            'EVAL_COLLATOR': 'eval_collator',
            'TRAIN_COLLATOR': 'train_collator',
            'EVAL_STEPS': 'eval_steps',
            'SAVE_STEPS': 'save_steps',
            'CHECKPOINT': 'checkpoint'
        }
    
    def update_from_env(self) -> None:
        for env_key, attr_name in self.env_mappings.items():
            env_value = environ.get(env_key)
            if env_value is not None:
                # Convert booleans and integers from environment variables
                if isinstance(getattr(self.config, attr_name), bool):
                    env_value = ast.literal_eval(env_value)
                elif isinstance(getattr(self.config, attr_name), int):
                    env_value = int(env_value)
                setattr(self.config, attr_name, env_value)
                print(f"updated {attr_name} from ENV: {getattr(self.config, attr_name)}")

class DatasetManager:

    def __init__(self, script_args, tokenizer):
        self.script_args = script_args
        self.tokenizer = tokenizer
        self.protocol = []
        
        if "chat" in script_args.model_name:
            self.formatting_func = partial(formatting_func_chat)
        else:
            self.formatting_func = partial(formatting_func_standard, eos_token=self.tokenizer.eos_token)

    @staticmethod
    def append_eos(ds):
        # If you want to modify the 'text' column, do it here.
        return ds

    def load_ds_from_folder(self, folder, shorten=None, concat=True):
        if shorten is None:
            shorten = {}

        files = glob.glob(folder + "/*")
        ds = {}
        
        for file in files:
            name = file.split('/')[-1].split('.')[0]
            test_eval = file.split('/')[-2]
            
            with open(file, 'rb') as handle:
                loaded_ds = pickle.load(handle).map(self.append_eos)

            if name in shorten.keys():
                loaded_ds = Dataset.from_pandas(pd.DataFrame(data=list(loaded_ds)[:shorten[name][test_eval]]))

            loaded_ds = loaded_ds.map(self.formatting_func, batched=True)

            print('Dataset:', test_eval, name)
            print(loaded_ds)
            self.protocol.append({'Dataset: ' + test_eval + ' ' + name: str(loaded_ds)})

            ds[name] = loaded_ds

        if concat:
            return concatenate_datasets(ds.values()).shuffle()
        else:
            return ds

    def load_datasets(self, shorten_dict=None):
        if shorten_dict is None:
            shorten_dict = {}

        train_folder = self.script_args.train_eval_dir + '/train'
        eval_folder = self.script_args.train_eval_dir + '/eval'

        train_ds = self.load_ds_from_folder(train_folder, shorten=shorten_dict)
        test_ds = self.load_ds_from_folder(eval_folder, shorten=shorten_dict, concat=False)
        
        return train_ds, test_ds

class ModelManager:
    
    def __init__(self, script_args,train_ds,test_ds):
        self.script_args = script_args
        self.protocol = [{k: v} for k, v in asdict(script_args).items()]
        self.ensure_output_dir()
        self.model, self.peft_config, self.tokenizer = self.create_and_prepare_model()
        
        # Write protocol
        with open(self.script_args.output_dir + "/protocol.txt", "w") as file:
            for dictionary in self.protocol:
                for key, value in dictionary.items():
                    file.write(f"{key}: {value}\n")
        
        # Set properties
        self.tokenizer.padding_side = "right"
        self.configure_collators()
        self.training_arguments = self.configure_training_arguments()
        self.trainer = self.initialize_trainer(train_ds, test_ds)  # Assuming train_ds and test_ds are globally available
    
    def ensure_output_dir(self):
        output_dir_exist = os.path.exists(self.script_args.output_dir)
        if not output_dir_exist:
            os.makedirs(self.script_args.output_dir)
    
    def create_and_prepare_model(self):
        compute_dtype = getattr(torch, self.script_args.bnb_4bit_compute_dtype)
        print('quantize:')
        print(script_args.use_4bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.script_args.use_4bit,
            load_in_8bit=self.script_args.use_8bit,
            bnb_4bit_quant_type=self.script_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.script_args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and self.script_args.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

        # Load the entire model on the GPU 0
        # switch to `device_map = "auto"` for multi-GPU
        device_map = 'auto'
        if 'mistral' in self.script_args.model_name:
            model = AutoModelForCausalLM.from_pretrained(
                self.script_args.model_name, 
                #quantization_config=bnb_config, 
                device_map=device_map, 
                use_auth_token=True,
                trust_remote_code=True
            )
        else:
                model = AutoModelForCausalLM.from_pretrained(
                self.script_args.model_name, 
                #quantization_config=bnb_config, 
                device_map=device_map, 
                use_auth_token=True,
                trust_remote_code=True
            )
        
        # check: https://github.com/huggingface/transformers/pull/24906
        model.config.pretraining_tp = 1 

        if self.script_args.use_lora:
            if 'falcon' in self.script_args.model_name:
                peft_config = LoraConfig(
                lora_alpha=self.script_args.lora_alpha,
                lora_dropout=self.script_args.lora_dropout,
                r=self.script_args.lora_r,
                target_modules=["query_key_value"],
                bias="none",
                task_type="CAUSAL_LM", 
            )     
            elif 'mistral' in self.script_args.model_name:
                peft_config = LoraConfig(
                    lora_alpha=self.script_args.lora_alpha,
                    lora_dropout=self.script_args.lora_dropout,
                    r=self.script_args.lora_r,
                    bias="none",
                    task_type="CAUSAL_LM", 
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )            
            
            else:  
                peft_config = LoraConfig(
                    lora_alpha=self.script_args.lora_alpha,
                    lora_dropout=self.script_args.lora_dropout,
                    r=self.script_args.lora_r,
                    bias="none",
                    task_type="CAUSAL_LM", 
                )
        else: 
            peft_config = None

        tokenizer = AutoTokenizer.from_pretrained(self.script_args.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        return model, peft_config, tokenizer
       
    
    def configure_training_arguments(self):
        report_to = None
        if environ.get('WANDB_API_KEY') is not None:
            import wandb
            os.environ['WANDB_PROJECT'] = self.script_args.wnb_project
            report_to = 'wandb'
        else:
            print('WARNING: no weights and biases logging')

        training_arguments = TrainingArguments(
            output_dir=self.script_args.output_dir,
            per_device_train_batch_size=self.script_args.per_device_train_batch_size,
            per_device_eval_batch_size=self.script_args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.script_args.gradient_accumulation_steps,
            eval_accumulation_steps = self.script_args.gradient_accumulation_steps,
            optim=self.script_args.optim,
            save_steps=self.script_args.save_steps,
            logging_steps=self.script_args.logging_steps,
            learning_rate=self.script_args.learning_rate,
            fp16=self.script_args.fp16,
            bf16=self.script_args.bf16,
            max_grad_norm=self.script_args.max_grad_norm,
            #max_steps=script_args.max_steps,
            num_train_epochs = self.script_args.num_train_epochs,
            warmup_ratio=self.script_args.warmup_ratio,
            group_by_length=self.script_args.group_by_length,
            lr_scheduler_type=self.script_args.lr_scheduler_type,
            evaluation_strategy = 'steps',
            eval_steps = self.script_args.eval_steps,
            report_to = report_to
        )
        return training_arguments
    
    def configure_collators(self):
        if "chat" in self.script_args.model_name.lower() and "llama" in self.script_args.model_name.lower():
            response_template = '[/INST]'
        else:
            response_template = '### assistant:'

        if self.script_args.eval_collator == 'all':
            self.eval_data_collator = None

        elif self.script_args.eval_collator == 'completion':
            self.eval_data_collator = DataCollatorForCompletionOnlyLM(response_template,self.tokenizer)
        else:
            self.eval_data_collator = None

        if self.script_args.train_collator == 'all':
            self.train_data_collator = None

        elif self.script_args.train_collator == 'completion':
            self.train_data_collator = DataCollatorForCompletionOnlyLM(response_template,tokenizer)
        else:
            self.train_data_collator = None
        
    
    def initialize_trainer(self, train_ds, test_ds):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            train_data_collator = self.train_data_collator,
            eval_data_collator=self.eval_data_collator,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=self.script_args.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            packing=self.script_args.packing,
        )
        return trainer

    def train_model(self):
        if self.script_args.checkpoint:
            self.trainer.train(self.script_args.checkpoint)
        else:
            self.trainer.train()
        
        if self.script_args.merge_and_push:
                output_dir = os.path.join(self.script_args.output_dir, "final_checkpoints")
                self.trainer.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                # Free memory for merging weights
                del self.model
                torch.cuda.empty_cache()

                from peft import AutoPeftModelForCausalLM

                model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
                model = model.merge_and_unload()

                output_merged_dir = os.path.join(self.script_args.output_dir, "final_merged_checkpoint")
                model.save_pretrained(output_merged_dir, safe_serialization=True)
                self.tokenizer.save_pretrained(output_merged_dir)


def load_and_train(script_args):
    # Load Dataset
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    dataset_manager = DatasetManager(script_args, tokenizer)
    train_ds, test_ds = dataset_manager.load_datasets()

    # Initialize and train Model
    model_manager = ModelManager(script_args, train_ds, test_ds)
    model_manager.train_model()

    print("Training Completed!")

if __name__ == "__main__":
     # Initialize Config Manager
    config_manager = ConfigManager(ScriptArguments)
    config_manager.update_from_env()
    script_args = config_manager.config
    load_and_train(script_args)

