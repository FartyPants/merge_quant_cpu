from pathlib import Path

import gradio as gr
import json
import random
import time
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer 
import argparse
import sys
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer

from modules import utils
from modules.models import unload_model
import modules.shared as shared
import modules.monkey_patch_gptq_lora as monkeypatch

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
imported_logger = logging.getLogger('auto_gptq.modeling._base')

imported_logger.setLevel(logging.INFO)
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
}

refresh_symbol = '\U0001f504'  # 🔄

def load_data(data_path, tokenizer, n_samples):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

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


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_classes=elem_class)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button

def process_mergeCPU(model_name, peft_model_name, output_dir, gpu_cpu):
    
    base_model_name_or_path = Path(f'{shared.args.model_dir}/{model_name}')
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

    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map_arg,
        )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    print(f"Loading PEFT: {peft_model_path}")
    yield f"Loading PEFT: {peft_model_path}"
    lora_model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        device_map=device_map_arg,
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight
    #print(f"Layer[0] LoRA weight: {first_weight_old} -> {lora_weight}")

    #assert torch.allclose(first_weight_old, first_weight)

    # merge weights - new merging method from peft
    print(f"Running merge_and_unload")
    yield f"Running merge_and_unload"
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    print(f"Changing state dict")
    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }
    print(f"Loading tokenizer")
   
    tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path)
    #max_shard_size="400MB"
    print(f"Saving model in 10GB shard size ... wait - don't touch anyhing yet!")
    yield f"Saving model in 10GB shard size ... wait - don't touch anyhing yet!"
    LlamaForCausalLM.save_pretrained(base_model, f"{output_dir}", state_dict=deloreanized_sd)
   
    tokenizer.save_pretrained(f"{output_dir}")

    print(f"Model saved to {output_dir}")
    yield f"Model saved to {output_dir}"
    print(f"**** DONE ****")
"""
    if args.dataset == 'wikitext':
        traindataset = get_wikitext2(128, 0, args.seqlen, tokenizer)
    elif args.dataset == 'c4':
        traindataset = get_c4(128, 0, args.seqlen, tokenizer)

     model.quantize(traindataset, use_triton=use_triton, batch_size=batch_size)


def get_wikitext2(nsamples, seed, seqlen, tokenizer):

    logger = logging.getLogger(__name__)

    wikidata = Dataset.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    wikilist = [' \n' if s == '' else s for s in wikidata['text'] ]

    text = ''.join(wikilist)
    logger.info("Tokenising wikitext2")
    trainenc = tokenizer(text, return_tensors='pt')

    import random
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

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = Dataset.load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False
    )

    import random
    random.seed(seed)
    trainloader = []
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

    return trainloader

"""

def process_Quant(model_name, output_dir,groupsize,wbits,desact,fast_tokenizer,gpu_memory,cpu_memory,low_cpu):
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
 

    print(f"Loaded in : {end - start:.2f} sec")

    start = time.time()

    print(f"Loading Dataset extensions/merge_quant_cpu/dataset/alpaca_data_cleaned.json")
    yield f"Loading Dataset"

    num_samples = 128
    examples = load_data("extensions/merge_quant_cpu/dataset/alpaca_data_cleaned.json", tokenizer, num_samples)

    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]
    print(f"Quantize started... (it will take time! ~30 min for 13b)")
    yield f"Quantize started...sit tight..."


    quant_batch_size = 1

    model.quantize(
        examples_for_quant,
        batch_size=quant_batch_size,
        use_triton=False,
        autotune_warmup_after_quantized=False
    )



    end = time.time()
    
    minutes = int(end - start)/60
    print(f"Quantization finished in: {minutes} min")

    print(f"Saving Quantizied model...")
    yield f"Saving Quantizied model..."

    quantized_model_dir = f"{output_dir}"
    model.save_quantized(quantized_model_dir,use_safetensors=True)

    print(f"Saving tokenizer model...")
    tokenizer.save_pretrained(f"{output_dir}")

    end = time.time()
    minutes = int(end - start_total)/60
    print(f"***Done**** It only took: {minutes} min")
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




def ui():

    imported_logger.info(f"*** Extension Merge Loaded ***")
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
    with gr.Accordion("Merge HF Model with Lora", open=True):
        
        with gr.Column():
            with gr.Row():
                
                with gr.Column():
                    with gr.Row():
                        gr_modelmenu = gr.Dropdown(choices=utils.get_available_models(), value=model_name, label='LlaMA Model (float 16) HF only, No GPTQ, No GGML',elem_classes='slim-dropdown')
                        create_refresh_button(gr_modelmenu, lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')
                with gr.Column():
                    with gr.Row():
                        gr_loramenu = gr.Dropdown(multiselect=False, choices=utils.get_available_loras(), value=lora_names, label='LoRA', elem_classes='slim-dropdown')
                        create_refresh_button(gr_loramenu, lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': lora_names}, 'refresh-button')
            gr_gpu_cpu = gr.Radio(choices=['GPU (Auto)','CPU'], value = 'CPU')
            output_dir = gr.Textbox(label='Output Dir', info='The folder name of your merge (relative to text-generation-webui)', value='models/my_merged_model_HF')
            gr_apply = gr.Button(value='Do Merge')
 
    with gr.Accordion("Quantize HF Model", open=True):
        with gr.Column():
            with gr.Row():
                
                with gr.Column():
                    with gr.Row():
                        gr_modelmenu2 = gr.Dropdown(choices=utils.get_available_models(), value=model_name, label='LlaMA Model (float 16) HF only, No GPTQ, No GGML',elem_classes='slim-dropdown')
                        create_refresh_button(gr_modelmenu2, lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')
                with gr.Column():
                    groupsize = gr.Dropdown(label="Groupsize", choices=["None", 32, 64, 128, 1024], value='128')
                    wbits = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value='4', interactive=False)
                    
                    desact = gr.Checkbox(label="Quantize with desc_act (can slow down inference but the perplexity may be better)", value=False)
                    fast_tokenizer= gr.Checkbox(label="Use fast tokenizer", value=True)
                    low_cpu = gr.Checkbox(label="Low CPU memory usage", value=True)
                    gpu_memory = gr.Slider(label=f"gpu-memory in MiB for device ", maximum=48, value=0,step=1)
                    cpu_memory = gr.Slider(label="cpu-memory in MiB", maximum=64, value=0,step=1)
            with gr.Row():        
                autoformat = gr.Textbox(label='Name', info='Base name that will be expanded in the Formatted Output Dir', value='quantizied_model', interactive=True)
                output_dirQ = gr.Textbox(label='Formatted Output Dir', info='The folder name of your merge (relative to text-generation-webui)', value='models/quantizied_model_GPTQ', interactive=True)

            gr_applyQuant = gr.Button(value='Do Quantization')        
                     
    gr_out = gr.Markdown('')   
    gr_apply.click(process_mergeCPU, inputs=[gr_modelmenu, gr_loramenu,output_dir,gr_gpu_cpu], outputs=gr_out)
    gr_applyQuant.click(process_Quant,[gr_modelmenu2, output_dirQ,groupsize,wbits,desact,fast_tokenizer,gpu_memory,cpu_memory,low_cpu], gr_out)
    
    def auto_format(inputs, wbits,groupsize):
        bits = int(wbits)
        group = int(groupsize)
        output = f"models/{inputs}-{bits}b-{group}g-GPTQ"
        return output
    
    def auto_form2(inputs,wbits,groupsize):
        #out = auto_format(inputs,wbits,groupsize)
        return inputs

    gr_modelmenu2.change(auto_form2,[gr_modelmenu2,wbits,groupsize],autoformat)

    autoformat.change(auto_format,[autoformat,wbits,groupsize],output_dirQ)

    def update_merge_name(module,lora):
        if not module:
            module = "module"
        if not lora:
            lora = "lora"

        modulest = f"{module}"
        modulest = modulest.replace("_HF", "")
        modulest = modulest.replace("_Hf", "")
        out = "models/"+f"{modulest}+{lora}_HF"
        return out
      

    gr_modelmenu.change(update_merge_name,[gr_modelmenu,gr_loramenu],output_dir)
    gr_loramenu.change(update_merge_name,[gr_modelmenu,gr_loramenu],output_dir)


