from pathlib import Path
import os

def set_protobuf_implementation_to_python():
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Call the function to set the environment variable before using Protobuf or the Transformers library
#set_protobuf_implementation_to_python()
import shutil

import gradio as gr
import json
import random
import time
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig
import argparse
import sys
#from .auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer

from modules import utils
from modules.models import unload_model
import modules.shared as shared
import numpy as np
import re
from modules.ui import create_refresh_button
from datasets import load_dataset

from peft.tuners.lora import LoraLayer
from peft import (
    LoraConfig,
    get_peft_model
)


import pkg_resources

def check_protobuf_installed():
    try:
        pkg_resources.get_distribution('protobuf')
        return True
    except pkg_resources.DistributionNotFound:
        return False


import logging
#from datasets import load_dataset
# Create a logger and set the log level
# Configure the root logger
# Configure the logger for your script
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

# Create a stream handler and set the log level


# Configure the logger for the imported library
#imported_logger = logging.getLogger('auto_gptq.modeling._base')

#imported_logger.setLevel(logging.INFO)


#if not imported_logger.handlers:
#    stream_handler = logging.StreamHandler()
    # Create a formatter to specify the log message format
#    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#    stream_handler.setFormatter(formatter)
#   imported_logger.addHandler(stream_handler)


from importlib.metadata import version

#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

#logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
#rootLogger = logging.getLogger()
#rootLogger.propagate = True
#consoleHandler = logging.StreamHandler(sys.stdout)
#consoleHandler.setFormatter(logFormatter)
#rootLogger.addHandler(consoleHandler)

params = {
    "display_name": "Merge-Quantize-CPU",
    "is_tab": True,
    "list_by_time":True,
}

selected_lora_main_sub =''
selected_lora_main =''
selected_lora_sub = ''

refresh_symbol = '\U0001f504'  # 🔄

import json

def compare_and_log_state_dicts(dict1, dict2):

    differences = []
  

    for key1, value1 in dict1.items():
        if key1 in dict2:
            value2 = dict2[key1]
            if not torch.equal(value1, value2):
                # Convert tensors to lists
                list1 = value1.flatten().tolist()
                list2 = value2.flatten().tolist()


                # Identify differing elements
                differing_indices = [i for i, (v1, v2) in enumerate(zip(list1, list2)) if v1 != v2]
                print(f"{key1} differ in {len(differing_indices)} values")    
                #if differing_indices:
                #    # Include only differing elements in the output
                #    difference = {
                #        'key': key1,
                #        'differing_elements': [(i, list1[i], list2[i]) for i in differing_indices],
                #    }
                #    #differences.append(difference)
            else:
                print(f"{key1} is same")      


    # Save differences to a JSON file
    with open('state_dict_diff.json', 'w') as json_file:
        json.dump(differences, json_file, indent=2)
 
# Example usage:
# Assuming base_model1 and base_model2 are two instances of the same model
#state_dict1 = base_model1.state_dict()
#state_dict2 = base_model2.state_dict()




def comare_dict(model1, model2):

    max_memory = None
    base_model_name_or_path1 = Path(f'{shared.args.model_dir}/{model1}')
    base_model_name_or_path2 = Path(f'{shared.args.model_dir}/{model2}')


    print(f"Unloading model from memory")
    unload_model()

    device_arg = { 'device_map': 'auto' }
    device_map_arg = {"": "cpu"}

    print(f"Loading base model A: {base_model_name_or_path1}")
    base_model1 = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path1,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map_arg,
        return_dict=True,
        max_memory=max_memory, 
        )

    print(f"Loading base model B: {base_model_name_or_path2}")
    base_model2 = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path2,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map_arg,
        return_dict=True,
        max_memory=max_memory, 
        )

    state_dict1 = base_model1.state_dict()
    state_dict2 = base_model2.state_dict()

    compare_and_log_state_dicts(state_dict1, state_dict2)

    print("finished")

def calc_trainable_parameters(model):
    trainable_params = 0
    all_param = 0 
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return trainable_params,all_param



def load_data(data_path, tokenizer, n_samples):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print("Loading Alpaca dataset")

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"]
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset



def process_mergeCPU(model_name, peft_model_name, output_dir, gpu_cpu, gpu_memory,cpu_memory,safetensor):
    
    global selected_lora_sub
    max_memory = dict()
    int_gpu = int(gpu_memory)
    if int_gpu > 0:    
        if torch.cuda.is_available():
            print(f"GPU: {int_gpu}GIB")
            max_memory.update({i: f"{int_gpu}GIB" for i in range(torch.cuda.device_count())})
    int_cpu = int(cpu_memory)
    if int_cpu > 0 and max_memory:
        max_memory["cpu"] = f"{int_cpu}GIB"
        print(f"CPU: {max_memory['cpu']}")
    if not max_memory:
        max_memory = None

#offload_folder=offload_folder,
        
    lora_subfolder = selected_lora_sub
    if lora_subfolder == 'Final':
        lora_subfolder = ''

    base_model_name_or_path = Path(f'{shared.args.model_dir}/{model_name}')

    if lora_subfolder!='':
        peft_model_path = Path(f'{shared.args.lora_dir}/{peft_model_name}/{lora_subfolder}')
    else:    
        peft_model_path = Path(f'{shared.args.lora_dir}/{peft_model_name}')

    print(f"Unloading model from memory")
    unload_model()

    device_arg = { 'device_map': 'auto' }
    if gpu_cpu=='CPU':
        device_map_arg = {"": "cpu"}
    else:
        device_map_arg = 'auto'

    print(f"Loading base model: {base_model_name_or_path}")
    yield f"Loading base model: {base_model_name_or_path}"


    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map_arg,
        return_dict=True,
        max_memory=max_memory, 
        )

    model_trainable_params, model_all_paramsbase = calc_trainable_parameters(base_model)
    print(f"Model Trainable params: {model_trainable_params:,d} ({100 * model_trainable_params / model_all_paramsbase:.4f} %), All params: {model_all_paramsbase:,d}")

    #first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    #first_weight_old = first_weight.clone()
    #attributes_and_methods = vars(base_model.model)
    #print(attributes_and_methods)

    #print(f"saving state dict")
    #torch.save(base_model.model.state_dict(), 'model_checkpoint.pth')

    if peft_model_name!="None":
        print(f"Loading PEFT: {peft_model_path}")
        yield f"Loading PEFT: {peft_model_path}"
        lora_model = PeftModel.from_pretrained(
            base_model,
            peft_model_path,
            device_map=device_map_arg,
            torch_dtype=torch.float16,
            max_memory=max_memory,
        )

        model_trainable_params, model_all_params = calc_trainable_parameters(lora_model)
        print(f"LoRA  Trainable params: {model_trainable_params:,d} ({100 * model_trainable_params / model_all_params:.4f} %), Params with Lora: {model_all_params:,d} from {model_all_paramsbase:,d}")


        #lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight
        #print(f"Layer[0] LoRA weight: {first_weight_old} -> {lora_weight}")

        #assert torch.allclose(first_weight_old, first_weight)

        # merge weights - new merging method from peft
        print(f"Running merge_and_unload")
        yield f"Running merge_and_unload"
        lora_model = lora_model.merge_and_unload()
        lora_model.train(False)

        # did we do anything?
        #assert not torch.allclose(first_weight_old, first_weight)


    #print(f"Changing state dict")
    #lora_model_sd = lora_model.state_dict()
    #deloreanized_sd = {
    #    k.replace("base_model.model.", ""): v
    #    for k, v in lora_model_sd.items()
    #    if "lora" not in k
    #}

    #max_shard_size="400MB"
    print(f"Saving model in 10GB shard size ... wait - don't touch anyhing yet!")
    yield f"Saving model in 10GB shard size ... wait - don't touch anyhing yet!"
    LlamaForCausalLM.save_pretrained(base_model, f"{output_dir}", safe_serialization=safetensor) #, state_dict=deloreanized_sd)


    tokenizer_path = os.path.join(base_model_name_or_path, "tokenizer.model")
    # Check if the tokenizer.model file exists
    if not os.path.isfile(tokenizer_path):
        print(f"The tokenizer.model file does not exist at the path: {tokenizer_path}")
        tokenizer_config_path = os.path.join(base_model_name_or_path, "tokenizer_config.json")
        output_tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
        if os.path.isfile(tokenizer_config_path):
            shutil.copyfile(tokenizer_config_path, output_tokenizer_config_path)
            print(output_tokenizer_config_path)
        
        tokenizer_config_path2 = os.path.join(base_model_name_or_path, "tokenizer.json")
        output_tokenizer_config_path2 = os.path.join(output_dir, "tokenizer.json")
        if os.path.isfile(tokenizer_config_path2):
            shutil.copyfile(tokenizer_config_path2, output_tokenizer_config_path2)
            print(output_tokenizer_config_path2)
        
        tokenizer_config_path3 = os.path.join(base_model_name_or_path, "special_tokens_map.json")
        output_tokenizer_config_path3 = os.path.join(output_dir, "special_tokens_map.json")
        if os.path.isfile(tokenizer_config_path3):
            shutil.copyfile(tokenizer_config_path3, output_tokenizer_config_path3)
            print(output_tokenizer_config_path3)
 
    else:
        print(f"Loading tokenizer")
   
        tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.save_pretrained(f"{output_dir}")

    print(f"Model saved to {output_dir}")
    yield f"Model saved to {output_dir}"


    # Write content to the merge file

    merge_file_path = os.path.join(output_dir, "_merge.txt")
    with open(merge_file_path, 'w') as merge_file:
        merge_file.write("This is a merge file content.\n")
        merge_file.write(f"Base Model: {model_name}\n")
        merge_file.write(f"LORA: {peft_model_name}\n")
        merge_file.write(f"Checkpoint: {selected_lora_sub}\n")



    print(f"**** DONE ****")
"""
    traindataset = get_wikitext2(128, 0, 2048, pretrained_model_dir)
    if args.dataset == 'wikitext':
        traindataset = get_wikitext2(128, 0, args.seqlen, tokenizer)
    elif args.dataset == 'c4':
        traindataset = get_c4(128, 0, args.seqlen, tokenizer)

     model.quantize(traindataset, use_triton=use_triton, batch_size=batch_size)

"""
# so slow!
def get_wikitext2_v2(nsamples, seed, seqlen, tokenizer):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    print (f"Processing Wikitext_v2: nsamples: {nsamples}, seqlen: {seqlen}")
    # load dataset and preprocess 
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    #testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    #testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    traindataset = []
    print("[", end="")
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
        print("+", end="")
    print("]")    
    return traindataset


def get_wikitext2(nsamples, seed, seqlen, tokenizer):

    logger = logging.getLogger(__name__)

    print (f"Processing Wikitext: nsamples: {nsamples}, seqlen: {seqlen}")
    wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikilist = [' \n' if s == '' else s for s in wikidata['text'] ]

    text = ''.join(wikilist)
    logger.info("Tokenising wikitext2")
    trainenc = tokenizer(text, return_tensors='pt')

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset

def get_wikitext2_v3(nsamples, seed, seqlen, tokenizer):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    print (f"Processing Wikitext_v3: nsamples: {nsamples}, seqlen: {seqlen}")
    # load dataset and preprocess 
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    print (f"Dataset loaded {len(traindata)} items")
    #testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # this is literally insane to to kenize the whole thing
    #trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    #print (f"Dataset tokenized to {trainenc.input_ids.shape[1]} tokens")
    #testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    traindataset = []
    print("[", end="")

    numdiv = int(seqlen / 50)+1
    

    for _ in range(nsamples):
        while True:

            i = random.randint(0, len(traindata) - (numdiv+1))
            text = ''
            for k in range(numdiv):
                text = text+"\n"+traindata[i+k]['text']
            
            trainenc = tokenizer(text, return_tensors='pt')
            #print (f"{trainenc.input_ids.shape[1]}")
            if trainenc.input_ids.shape[1] >= seqlen:
                break


        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
        print("+", end="")
    print("]")    
    return traindataset


def get_c4(nsamples, seed, seqlen, tokenizer):

    print (f"Processing C4: nsamples: {nsamples}, seqlen: {seqlen}")

    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False
    )

   
    random.seed(seed)
    trainloader = []
    print("[", end="")
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        trainloader.append({'input_ids':inp,'attention_mask': attention_mask})
        print("+", end="")
    print("]")     
    return trainloader



'''
Using transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_id = "WizardLM/WizardLM-7B-V1.0"

quantization_config = GPTQConfig(
bits=4,
group_size=128,
dataset=”c4",
desc_act=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map=’auto’)

from huggingface_hub import notebook_login

notebook_login()

quant_model.push_to_hub(“WizardLM-7B-V1.0-gptq-4bit”)
tokenizer.push_to_hub(“WizardLM-7B-V1.0-gptq-4bit”)
'''

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Define the original info method
original_info = logging.Logger.info

def format_time(seconds: float):
    if seconds < 120:
        return f"{seconds:.0f}s"

    minutes = seconds / 60
    if minutes < 120:
        return f"{minutes:.0f}min"

    hours = minutes / 60
    return f"{hours:.0f}h"


start_time = 0.00
disp_layernum = 0
# Define your custom info method
def custom_info(self, msg, *args, **kwargs):
    # Your custom implementation goes here
    global start_time
    global disp_layernum

    infotext = ""
    if msg.startswith("Start quantizing layer"):
        splitmsg = msg.split()
        current_step = int(splitmsg[3].split("/")[0])
        total_steps = int(splitmsg[3].split("/")[1])
        if current_step<2:
            start_time = time.perf_counter()
            disp_layernum = 0
        else:
            time_elapsed = time.perf_counter() - start_time
            if time_elapsed <= 0:
                timer_info = ""
                total_time_estimate = 999
            else:
                its = (current_step-1) / time_elapsed
                if its > 1:
                    timer_info = f"{its:.0f}layer/s"
                else:
                    timer_info = f"{1.0/its:.0f}s/layer"

                total_time_estimate = (1.0 / its) * (total_steps)
            #[{timer_info}],
            infotext = f" ({format_time(time_elapsed)}/{format_time(total_time_estimate)})   ETA: {RED}{format_time(total_time_estimate - time_elapsed)}{RESET}"
       
        if current_step==2:
            print(f"[Benchmark: {RED}{timer_info}{RESET}]")

        if current_step==total_steps:
            print (f"{RED}Finishing the last Layer{RESET}")
            print(" + Packing model layers (~5 min) - (Almost there!)")
        else:
            print(f"{RED}{msg}{RESET} {infotext}")    



    if msg.startswith("Quantizing"):
        COLUMN_WIDTH = 60
        quantizing_msg = f" - {msg}"
        # Calculate the length of the message and add dots to fill the COLUMN_WIDTH
        padding = max(0, COLUMN_WIDTH - len(quantizing_msg))
        quantizing_msg += '.' * padding
        print(f"{GREEN}{quantizing_msg}{RESET}", end='')

    if msg.startswith("avg loss"):
        print(f" {YELLOW}{msg}{RESET}")

    if msg.startswith("model.layers."):
        parts = msg.split('.')
        num_layer = int(parts[2])+1
        if num_layer> disp_layernum:
            disp_layernum = num_layer
            print(f" - {YELLOW}Packing Layer: {num_layer}{RESET}")
        

           
    #if display_all:
    #    print(f" - {YELLOW}{msg}{RESET}")

    # Call the original info method to maintain its behavior
    original_info(self, msg, *args, **kwargs)


def process_Quant(model_name, output_dir,groupsize,wbits,desact,fast_tokenizer,gpu_memory,cpu_memory,low_cpu, dataset_type,max_seq_len,num_samples):

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    base_model_name_or_path = Path(f'{shared.args.model_dir}/{model_name}')
    max_memory = dict()
    int_gpu = int(gpu_memory)
    if int_gpu > 0:    
        if torch.cuda.is_available():
            print(f"GPU: {int_gpu}GIB")
            max_memory.update({i: f"{int_gpu}GIB" for i in range(torch.cuda.device_count())})
    int_cpu = int(cpu_memory)
    if int_cpu > 0 and max_memory:
        max_memory["cpu"] = f"{int_cpu}GIB"
        print(f"CPU: {max_memory['cpu']}")
    if not max_memory:
        max_memory = None

    #imported_logger.setLevel(logging.INFO)



    print(f"Unloading model from memory")
    unload_model()        

    start = time.time()
    start_total = time.time()

    print(f"Transformers: AutoTokenizer (fast_tokenizer: {fast_tokenizer}, trust_remote_code: False): Loading {base_model_name_or_path} ")
    yield f"Transformers: AutoTokenizer: loading {base_model_name_or_path}"
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        use_fast=fast_tokenizer,
        trust_remote_code=False
    )

    bits_i = int(wbits)
    group_size = int(groupsize)
    if group_size==0:
        group_size = -1

    print(f"AutoGPTQForCausalLM: Loading Model {base_model_name_or_path} bits {bits_i}, groups {group_size}, desc_act {desact}")
    yield f"AutoGPTQForCausalLM: Loading Model {base_model_name_or_path}"

    #damp_percent = 0.01
    quantize_config = BaseQuantizeConfig(
        bits=bits_i,
        group_size=group_size,
        desc_act=desact
    )


    torch_dtype  = torch.float16

    #if dtype == 'float16':
    #    torch_dtype  = torch.float16
    #elif dtype == 'float32':
    #    torch_dtype  = torch.float32
    #elif dtype == 'bfloat16':
    #    torch_dtype  = torch.bfloat16
    #else:
    #    raise ValueError(f"Unsupported dtype: {dtype}")


    model = AutoGPTQForCausalLM.from_pretrained(
        base_model_name_or_path,
        quantize_config=quantize_config,
        max_memory=max_memory,
        low_cpu_mem_usage=low_cpu, 
        torch_dtype=torch_dtype,
        trust_remote_code=False
    )

    end = time.time()

    # get model maximum sequence length
    model_config = model.config.to_dict()
    seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
    if any([k in model_config for k in seq_len_keys]):
        for key in seq_len_keys:
            if key in model_config:
                #model.seqlen = model_config[key]
                print(f"Model Nax seqlen: {model.seqlen}")
                break
    else:
        print("can't get model's sequence length from model config, will set to 2048.")

    #by default?    
    #model.seqlen = 2048

 
    model_trainable_params, model_all_params = calc_trainable_parameters(model)
    print(f"Before Trainable params: {model_trainable_params:,d} ({100 * model_trainable_params / model_all_params:.4f} %), All params: {model_all_params:,d}")


    print(f"Loaded in : {end - start:.2f} sec")

    start = time.time()



    print(f"Loading Dataset")
    yield f"Loading Dataset"

    num_samples_int = int(num_samples)
    print(f"num_samples {num_samples}")

    if dataset_type=="Wikitext2":
        examples_for_quant = get_wikitext2_v3(num_samples_int, 0, int(max_seq_len), tokenizer)
    elif  dataset_type=="c4":
        examples_for_quant = get_c4(num_samples_int, 0, int(max_seq_len), tokenizer)
    else:
        print(f"Loading Dataset extensions/merge_quant_cpu/dataset/alpaca_data_cleaned.json")
        examples = load_data("extensions/merge_quant_cpu/dataset/alpaca_data_cleaned.json", tokenizer, num_samples_int)

        examples_for_quant = [
            {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
            for example in examples
        ]

    
    print(f"Quantize started... (it will take time! 30-35 min for 13b, 20-25 min for 7b)")
    yield f"Quantizing ..sit tight... (see terminal for progress)"

    # load from dataset
    # Load data and tokenize examples
    '''
    n_samples = 1024
    data = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split=f"train[:{n_samples*5}]")
    tokenized_data = tokenizer("\n\n".join(data['text']), return_tensors='pt')

    # Format tokenized examples
    examples_for_quant = []
    for _ in range(n_samples):
        i = random.randint(0, tokenized_data.input_ids.shape[1] - tokenizer.model_max_length - 1)
        j = i + tokenizer.model_max_length
        input_ids = tokenized_data.input_ids[:, i:j]
        attention_mask = torch.ones_like(input_ids)
        examples_for_quant.append({'input_ids': input_ids, 'attention_mask': attention_mask})

'''


    # Monkey-patch the Logger class
    logging.Logger.info = custom_info

    quant_batch_size = 1

    model.quantize(
        examples_for_quant,
        batch_size=quant_batch_size,
        use_triton=False,
        autotune_warmup_after_quantized=False
    )


    logging.Logger.info = original_info


    end = time.time()
    
    minutes = int(end - start)/60
    print(f"Quantization finished in: {minutes} min")

    print(f"Saving Quantizied model...")
    yield f"Saving Quantizied model..."

    quantized_model_dir = f"{output_dir}"
    model.save_quantized(quantized_model_dir,use_safetensors=True)

    model_trainable_params, model_all_params = calc_trainable_parameters(model)
    print(f"After  Trainable params: {model_trainable_params:,d} ({100 * model_trainable_params / model_all_params:.4f} %), All params: {model_all_params:,d}")


    print(f"Saving tokenizer ...")
    tokenizer.save_pretrained(f"{output_dir}")

    end = time.time()
    minutes = int(end - start_total)/60
    print(f"{RED}**** Done ****{RESET} It only took: {minutes} min")
    yield f"Done in {minutes} minutes"


#pip install protobuf==3.20.*    


def clean_path(base_path: str, path: str):
    """Strips unusual symbols and forcibly builds a path as relative to the intended directory."""
    # TODO: Probably could do with a security audit to guarantee there's no ways this can be bypassed to target an unwanted path.
    # Or swap it to a strict whitelist of [a-zA-Z_0-9]
    path = path.replace('\\', '/').replace('..', '_')
    if base_path is None:
        return path

    return f'{Path(base_path).absolute()}/{path}'


def extract_lora_layers(model_name, output_folder):

    device_map_arg = {"": "cpu"}


    base_model_name_or_path = Path(f'{shared.args.model_dir}/{model_name}')
    print(f"Unloading model from memory")
    unload_model()
    print(f"Loading model: {base_model_name_or_path}")
    yield f"Loading model: {base_model_name_or_path}"

    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map_arg,
        )

    #config = LlamaConfig.from_pretrained(base_model_name_or_path)
    #extracted_model = type(base_model)(base_model.config)  # Create a new instance of the same type as the base model

    config = LoraConfig(
        bias="none",
        task_type="CAUSAL_LM"
    )

    lora_model = get_peft_model(base_model, config)

    print(f"Extracting")
    for key, module in base_model.named_modules():
        if isinstance(module, LoraLayer):
            lora_model.add_module(key, module)  # Add the LoRa layer to the extracted model

  
    if not Path(output_folder).exists():
        os.mkdir(output_folder)

    lora_model.save_pretrained(output_folder)    
    # Save the model's state dict to a binary file
    #model_path = output_folder + "/adapter_model.bin"
    #torch.save(extracted_model.state_dict(), model_path)

    # Save the model's configuration to a JSON file
    #config = extracted_model.config
    #config_path = output_folder + "/adapter_config.json"

    #with open(config_path, "w") as config_file:
    #    json.dump(config, config_file)

    print("Done")
    return "Done"

# correct combine
'''
# Add first lora
lora_path = ...
lora_name = "add_detail"
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    f"{lora_path}/unet",
    lora_name
)
pipe.text_encoder = PeftModel.from_pretrained(
    pipe.text_encoder,
    f"{lora_path}/text_encoder",
    lora_name
)

# Merge first LoRA to model weights
pipe.unet = pipe.unet.merge_and_unload()
pipe.text_encoder = pipe.text_encoder.merge_and_unload()

# Load second LoRA
lora_path = ...
lora_name = "3DMM_V11"
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    f"{lora_path}/unet",
    lora_name
)
pipe.text_encoder = PeftModel.from_pretrained(
    pipe.text_encoder,
    f"{lora_path}/text_encoder",
    lora_name
)

torch.manual_seed(1428928479)
image = pipe(
    prompt = "candid RAW portrait photo of a woman (Crystal Simmerman:1.0) with (dark hair:1.0) and a (purple colored suit:1.0) on a dark street with shopping windows (at night:1.2), bokeh, Ilford Delta 3200 film, dof, high definition, detailed, intricate, flashlight",
    negative_prompt = "bad-hands-5, asian, cropped, lowres, poorly drawn face, out of frame, blurry, blurred, text, watermark, disfigured, closed eyes, ugly, cartoon, render, 3d, plastic, 3d (artwork), rendered, comic",
    num_inference_steps=20,
    guidance_scale=7,
).images[0]


    #using torch
    l1 = torch.load(_path_1)
    l2 = torch.load(_path_2)

    l1pairs = zip(l1[::2], l1[1::2])
    l2pairs = zip(l2[::2], l2[1::2])

    for (x1, y1), (x2, y2) in zip(l1pairs, l2pairs):
        # print("Merging", x1.shape, y1.shape, x2.shape, y2.shape)
        x1.data = alpha_1 * x1.data + alpha_2 * x2.data
        y1.data = alpha_1 * y1.data + alpha_2 * y2.data

        out_list.append(x1)
        out_list.append(y1)

    if opt == "unet":

        print("Saving merged UNET to", output_path)
        torch.save(out_list, output_path)

'''


# combine loras simple, not good

#model = AutoModel... # base model # set `load_in_8bit` to `False`
#for peft_model_id in peft_model_ids:
#    model = PeftModel.from_pretrained(model, peft_model_id)
#    model = model.merge_and_unload()

'''
# Add first LoRA
lora_path = ...
lora_name = "add_detail"
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    f"{lora_path}/unet",
    lora_name
)
pipe.text_encoder = PeftModel.from_pretrained(
    pipe.text_encoder,
    f"{lora_path}/text_encoder",
    lora_name
)

# Add second LoRA
lora_path = ...
lora_name = "3DMM_V11"
pipe.unet.load_adapter(
    f"{lora_path}/unet",
    adapter_name=lora_name
)
pipe.text_encoder.load_adapter(
    f"{lora_path}/text_encoder",
    adapter_name=lora_name
)

def create_weighted_lora_adapter(pipe, adapters, weights, adapter_name="default"):
    pipe.unet.add_weighted_adapter(adapters, weights, adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.add_weighted_adapter(adapters, weights, adapter_name)

    return pipe

# Mix two LoRAs together
pipe = create_weighted_lora_adapter(pipe, ["add_detail", "3DMM_V11"], [1.0, 1.0], "combined")
pipe.unet.set_adapter("combined")
pipe.text_encoder.set_adapter("combined")

'''
def atoi(text):
    return int(text) if text.isdigit() else text.lower()

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def list_Folders_byAlpha(directory):

    if not directory.endswith('/'):
        directory += '/'

    subfolders = []
    path = directory
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]

    time_sorted_list = sorted(full_list, key=natural_keys, reverse=False)

    for entry in time_sorted_list:
        if os.path.isdir(entry):
            entry_str = f"{entry}"  # Convert entry to a string
            full_path = entry_str
            entry_str = entry_str.replace('\\','/')
            entry_str = entry_str.replace(f"{directory}", "")  # Remove directory part
            subfolders.append(entry_str)

    return subfolders        


def list_subfoldersByTime(directory):

    if not directory.endswith('/'):
        directory += '/'
    subfolders = []
    path = directory
    name_list = os.listdir(path)
    full_list = [os.path.join(path,i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime,reverse=True)

    for entry in time_sorted_list:
        if os.path.isdir(entry):
            entry_str = f"{entry}"  # Convert entry to a string
            full_path = entry_str
            entry_str = entry_str.replace('\\','/')
            entry_str = entry_str.replace(f"{directory}", "")  # Remove directory part
            subfolders.append(entry_str)

    return subfolders

def list_subfolders(directory):
    subfolders = []
    
    if os.path.isdir(directory):
        
        subfolders.append('Final')

        for entry in os.scandir(directory):
            if entry.is_dir() and entry.name != 'runs':
                subfolders.append(entry.name)

    return sorted(subfolders, key=natural_keys, reverse=True)

def get_available_loras():
    model_dir = shared.args.lora_dir 
       
    subfolders = []
    
    if params.get("list_by_time",False):
        subfolders = list_subfoldersByTime(model_dir)
    else:
        subfolders = list_Folders_byAlpha(model_dir)      

    subfolders.insert(0, 'None')
    return subfolders      

def ui():
    global selected_lora_main_sub
    global selected_lora_main
    global selected_lora_sub
    
    #imported_logger.setLevel(logging.INFO)
    #imported_logger.info(f"*** Extension Merge Loaded ***")
    print('Torch: ', version('torch'))
    print('Transformers: ', version('transformers'))
    print('Accelerate: ', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    protobuf_installed = check_protobuf_installed()

    if protobuf_installed:
        protobuf_version = pkg_resources.get_distribution('protobuf').version
        print(f"Protobuf installed: Version: {protobuf_version}")

    else:
        print("Protobuf is not installed. You will probably need: pip install protobuf==3.20.* ")



    model_name = "None"
    lora_names = "None"

    with gr.Accordion("Memory", open=True):
        with gr.Row():
            with gr.Column():
                gpu_memory = gr.Slider(label=f"gpu-memory in MiB for device ", maximum=48, value=0,step=1)
                cpu_memory = gr.Slider(label="cpu-memory in MiB", maximum=64, value=0,step=1)
            with gr.Column():
                gr.Markdown("Merge / Quantization")


    with gr.Accordion("Process", open=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("1. Merge HF Model with Lora:")
                with gr.Row():
                    gr_modelmenu = gr.Dropdown(choices=utils.get_available_models(), value=model_name, label='LlaMA Model (float 16) HF only, No GPTQ, No GGML',elem_classes='slim-dropdown')
                    create_refresh_button(gr_modelmenu, lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')
                with gr.Row():
                    with gr.Column(scale=5):    
                        with gr.Row():
                            loramenu = gr.Dropdown(multiselect=False, choices=get_available_loras(), value='None', label='LoRA and checkpoints', elem_classes='slim-dropdown', allow_custom_value=True)
                            create_refresh_button(loramenu, lambda: None, lambda: {'choices': get_available_loras()}, 'refresh-button')
                    with gr.Column(scale=1):
                        lora_list_by_time = gr.Checkbox(value = params["list_by_time"],label="Sort by Time added",info="Sorting")
                with gr.Row():                            
                    lorasub2 = gr.Radio(choices=[], value='', label='Checkpoints')
                    gr.Markdown('If no lora is selected, it will just resave the model in float16')
                gr_gpu_cpu = gr.Radio(choices=['GPU (Auto)','CPU'], value = 'CPU')
                safetensor = gr.Checkbox(label="Safe Tensor", value=True)
                output_dir = gr.Textbox(label='Output Dir', info='The folder name of your merge (relative to text-generation-webui)', value='models/my_merged_model_HF')
                
    
        
            with gr.Column():
                gr.Markdown("2. Quantize to GPTQ:")
                with gr.Row():
                    gr_modelmenu2 = gr.Dropdown(choices=utils.get_available_models(), value=model_name, label='LlaMA Model (float 16) HF only, No GPTQ, No GGML',elem_classes='slim-dropdown')
                    create_refresh_button(gr_modelmenu2, lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')
                
                with gr.Row():
                    with gr.Column():
                        groupsize = gr.Dropdown(label="Groupsize", choices=["None", '32', '64', '128', '1024'], value='128', interactive=False)
                        wbits = gr.Dropdown(label="wbits", choices=["None", '1', '2', '3', '4', '8'], value='4', interactive=False)
                                
                        desact = gr.Checkbox(label="Quantize with desc_act (slow down inference, better perplexity)", value=False)
                        fast_tokenizer= gr.Checkbox(label="Use fast tokenizer", value=True)
                        low_cpu = gr.Checkbox(label="Low CPU memory usage", value=False)
                    with gr.Column():
                        dataset_type = gr.Radio(choices=['Alpaca', 'Wikitext2', 'c4'], value='Wikitext2', interactive=True, label="Dataset. Alpaca is fastest, but offers less quality. Wikitext and c4 add 1/3 of processing time")
                        max_seq_len = gr.Number(value = 2048,label="Max Sequence Length")
                        num_samples = gr.Number(value = 128,label="Number of samples")
                with gr.Row():        
                    autoformat = gr.Textbox(label='Name', info='Base name that will be expanded in the Formatted Output Dir', value='quantizied_model', interactive=True)
                    output_dirQ = gr.Textbox(label='Formatted Output Dir', info='The folder name of your merge (relative to text-generation-webui)', value='models/quantizied_model_GPTQ', interactive=True)
        with gr.Row():
            with gr.Column():    
                gr_apply = gr.Button(value='Do Merge')
                #gr_compare = gr.Button('Compare 1 and 2')
            with gr.Column():                    
                gr_applyQuant = gr.Button(value='Do Quantization')        
    #with gr.Accordion("Extract Lora", open=False):
    #    with gr.Column():
    #        with gr.Row():
    #            
    #            with gr.Column():
    #                with gr.Row():
    #                    gr_modelmenu3 = gr.Dropdown(choices=utils.get_available_models(), value=model_name, label='LlaMA Model (float 16) HF only, No GPTQ, No GGML',elem_classes='slim-dropdown')
    #                    create_refresh_button(gr_modelmenu3, lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')
    #        with gr.Row():        
    #            
    #            output_dirQ3 = gr.Textbox(label='Output Dir', info='The folder name of your output (relative to text-generation-webui)', value='loras/extracted_lora', interactive=True)
    #
    #        gr_applySplit = gr.Button(value='Do De-Merge')        
                       
    gr_out = gr.Markdown('')   
    gr_apply.click(process_mergeCPU, inputs=[gr_modelmenu, loramenu,output_dir,gr_gpu_cpu,gpu_memory,cpu_memory,safetensor], outputs=gr_out)
    gr_applyQuant.click(process_Quant,[gr_modelmenu2, output_dirQ,groupsize,wbits,desact,fast_tokenizer,gpu_memory,cpu_memory,low_cpu,dataset_type, max_seq_len,num_samples], gr_out)
    #gr_applySplit.click(extract_lora_layers,[gr_modelmenu3, output_dirQ3], gr_out)

    #gr_compare.click(comare_dict,[gr_modelmenu,gr_modelmenu2], None )

    def auto_format(inputs, wbits,groupsize):
        bits = int(wbits)
        group = int(groupsize)

        base_model_name_or_path = Path(f'{shared.args.model_dir}/{inputs}-{bits}b-{group}g-GPTQ')
        output = base_model_name_or_path
        return output
    
    def auto_form2(inputs,wbits,groupsize):
        #out = auto_format(inputs,wbits,groupsize)
        return inputs

    gr_modelmenu2.change(auto_form2,[gr_modelmenu2,wbits,groupsize],autoformat)

    autoformat.change(auto_format,[autoformat,wbits,groupsize],output_dirQ)

    def update_merge_name(module,lora):
        global selected_lora_sub

        if not module:
            module = "module"
        if not lora:
            lora = "lora"
        sublora = ''

        if selected_lora_sub!='' and selected_lora_sub!='Final':
            sublora = "+"+selected_lora_sub

        modulest = f"{module}"
        modulest = modulest.replace("_HF", "")
        modulest = modulest.replace("_Hf", "")
        
        out = f"{shared.args.model_dir}/{modulest}+{lora}{sublora}_HF"
        return out
      

    gr_modelmenu.change(update_merge_name,[gr_modelmenu,loramenu],output_dir)

    def update_lotra_subs_main(selectlora):
        global selected_lora_main
        global selected_lora_sub 
        selected_lora_main = ''
        selected_lora_sub = ''   
        if selectlora:
            model_dir = f"{shared.args.lora_dir}/{selectlora}"  # Update with the appropriate directory path
            selected_lora_main = selectlora
            subfolders = list_subfolders(model_dir)
            selected_lora_sub = 'Final'
            return gr.Radio.update(choices=subfolders, value ='Final') 

        return gr.Radio.update(choices=[], value ='')    


    loramenu.change(update_lotra_subs_main,loramenu, lorasub2).then(update_merge_name,[gr_modelmenu,loramenu],output_dir)

    def change_sort(sort):
        global params
        params.update({"list_by_time": sort})
   
 
    def update_reloadLora():
        return gr.Radio.update(choices=get_available_loras())
    
    lora_list_by_time.change(change_sort,lora_list_by_time,None).then(update_reloadLora,None, loramenu)    

    def update_lotra_sub(sub):
        global selected_lora_sub
        selected_lora_sub = sub
        
    lorasub2.change(update_lotra_sub, lorasub2, None ).then(update_merge_name,[gr_modelmenu,loramenu],output_dir)  
