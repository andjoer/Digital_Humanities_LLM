import os
import json
import dataclasses
from evaluate import evaluate_models

from peft import AutoPeftModelForCausalLM
import torch

from transformers import (
    AutoTokenizer,
    AutoTokenizer
)

from os import environ

def find_latest_checkpoint(folder_path):
    """
    Find the checkpoint directory with the highest number in a specified folder.
    
    :param folder_path: Path to the folder containing checkpoint directories.
    :return: Path to the latest checkpoint directory.
    """
    checkpoints = [dir for dir in os.listdir(folder_path) if dir.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(folder_path, latest_checkpoint)

def find_best_performance_in_checkpoint(checkpoint_dir, metric_name):
    """
    Find the best performance for a specified metric in a checkpoint.

    :param checkpoint_dir: Path to the checkpoint directory.
    :param metric_name: Name of the metric to evaluate.
    :return: Step number with the best performance and the performance value.
    """
    try:
        with open(os.path.join(checkpoint_dir, 'trainer_state.json'), 'r') as file:
            data = json.load(file)
            log_history = data['log_history']
            return find_min_loss_metric(log_history, metric_name)
    except FileNotFoundError:
        print(f"trainer_state.json not found in {checkpoint_dir}")
        return None, None

# Function from previous example
def find_min_loss_metric(log_history, metric_name):
    """
    Find the maximum value of a specified loss metric and the step(s) at which it occurs.

    :param log_history: List of log entries.
    :param metric_name: Name of the loss metric to search for.
    :return: A tuple containing the maximum loss value and a list of steps at which it occurs.
    """
    min_loss = float('inf')
    min_loss_step = 0

    # Iterate through the log history
    for entry in log_history:
        # Check if the entry contains the specified metric
        metric_key = f'eval_{metric_name}_loss'

        if metric_key in entry:
            # Update max_loss and max_loss_steps if a new maximum is found
            if entry[metric_key] < min_loss:
                min_loss = entry[metric_key]
                min_loss_step = entry['step']
            elif entry[metric_key] == min_loss:
                min_loss_step = entry['step']

    return min_loss, min_loss_step

def merge_ckpt(input_dir):
    output_merged_dir = input_dir + '-merged'

    model = AutoPeftModelForCausalLM.from_pretrained(input_dir, device_map= {"":0}, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    model.save_pretrained(output_merged_dir, safe_serialization=True)
    
    del model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_merged_dir)


@dataclasses.dataclass
class Args:
    eval_dir: str
    models: list
    samples: int
    loadDict: str


def main(model_name='DettmersAll7b64',metric_name='bsp_ds',do_evaluation = True, base_path = './results'):


    samples = 200
    eval_dir = 'data/train_test_datasets/run_9_rede_arg_bsp_bspsynth/eval'

    if  environ.get('EVAL_DIR') is not None:    
        eval_folder= environ.get('EVAL_DIR')
        print('updated eval_dir from ENV: '+str(eval_folder))
    if  environ.get('OUT_MODEL') is not None:    
        model_name = [environ.get('OUT_MODEL')]
        print('updated models from ENV: '+str(model_name))
    if  environ.get('EVAL_SAMPLES') is not None:    
        samples = int(environ.get('EVAL_SAMPLES'))
        print('updated models from ENV: '+str(samples))

    model_path = os.path.join(base_path,model_name)
    latest_checkpoint_dir = find_latest_checkpoint(model_path)
    if latest_checkpoint_dir:
        max_loss_value, max_loss_steps = find_best_performance_in_checkpoint(latest_checkpoint_dir, metric_name)
        if max_loss_value is not None:
            print(f"The best performance for '{metric_name}' is {max_loss_value} at steps {max_loss_steps}")

            if do_evaluation:
                print('start evaluation')

                ckpt = 'checkpoint-' + str(max_loss_steps)
                ckpt_path = os.path.join(model_path,ckpt)
                merge_ckpt(ckpt_path)
                ckpt_path += '-merged'
                eval_args = Args(eval_dir=eval_dir, models = [ckpt_path],samples = samples, loadDict='')
                evaluate_models(eval_args)

        else:
            print("Could not find the performance data in the latest checkpoint.")
    else:
        print("No checkpoints found in the specified folder.")

if __name__ == "__main__":

    main()



