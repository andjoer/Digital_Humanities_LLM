from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    AutoTokenizer
)
from peft import AutoPeftModelForCausalLM
import torch
from tqdm import tqdm

input_dirs = ['./results/run_1/checkpoint-1000','./results/OnlyBsp7b16/checkpoint-1000']

for input_dir in tqdm(input_dirs):
    output_merged_dir = input_dir + '-merged'

    model = AutoPeftModelForCausalLM.from_pretrained(input_dir, device_map= {"":0}, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    model.save_pretrained(output_merged_dir, safe_serialization=True)
    
    del model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_merged_dir)