from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    AutoTokenizer
)
from os import environ

from peft import AutoPeftModelForCausalLM
import torch
from tqdm import tqdm

input_dirs = ['./results/DettmersAll7b64/checkpoint-7000']

if  environ.get('INPUT_DIR') is not None:    
    input_dirs = [environ.get('INPUT_DIRS')]
    print('updated models from ENV: '+str(input_dirs))

for input_dir in tqdm(input_dirs):
    output_merged_dir = input_dir + '-merged'

    model = AutoPeftModelForCausalLM.from_pretrained(input_dir, device_map= {"":0}, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    model.save_pretrained(output_merged_dir, safe_serialization=True)
    
    del model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_merged_dir)