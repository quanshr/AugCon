from rouge_chinese import Rouge
import jieba
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel

import config


def gen_count(context):
    return len(context) // config.character_per_piece


def calc_rouge(hyp, ref):
    rouge = Rouge()
    hyp = ' '.join(jieba.lcut(hyp))
    ref = ' '.join(jieba.lcut(ref))
    return rouge.get_scores(hyp, ref)[0]['rouge-l']


def load_model():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
    model = AutoModelForCausalLM.from_pretrained(  # To reduce GPU memory usage, 4-bit QLoRA FT is used here
        config.model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config.lora_config)
    return model


def merge_lora():
    print('Start merging LoRA...')
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    if os.path.exists('results/LoRA'):
        model = PeftModel.from_pretrained(model, 'results/LoRA')
        model = model.merge_and_unload()
        model.save_pretrained('results/sft_model')
    else:
        model = PeftModel.from_pretrained(model, 'Qwen-DailyM-32B-LoRA')
        model = model.merge_and_unload()
        model.save_pretrained('Qwen-DailyM-32B')
