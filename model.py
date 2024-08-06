import os
import torch
from typing import *
import copy
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model(model_type, model_ckpt, quantization=None, local_rank=None):
    if "baichuan" in model_type.lower():
        AutoLoad = AutoModelForCausalLM # 1、首先根据是否含有baichuan来初始化自动加载预训练模型的工具。
    elif "chatglm" in model_type.lower():
        AutoLoad = AutoModel
    else:
        raise NotImplementedError
    device_map = {"": torch.cuda.current_device()}# 2、根据现在是否是分布式系统来决定是否需要进行GPU的索引配置
    # if we are in a distributed setting, we need to set the device map and max memory per device
    local_rank = os.environ.get('LOCAL_RANK', local_rank)
    if local_rank is not None:
        device_map = {'': int(local_rank)}
        print(f"local rank {local_rank} map {device_map}")
    if quantization == '4bit':
        print("load model with 4bit quantization")
        model = AutoLoad.from_pretrained(# 3、模型准备
            model_ckpt,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(# 量化的设置
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
    elif quantization == '8bit':
        print("load model with 8bit quantization")
        model = AutoLoad.from_pretrained(
            model_ckpt,
            device_map=device_map,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        )
    else:
        print("load model with bfloat16")
        model = AutoLoad.from_pretrained(
            model_ckpt,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        # 4、 分词器的实现，这里实现了数据预处理，尤其是在进行批次处理时，所有样本需要有相同的序列长度
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        print(f"pass unk_token_id {tokenizer.unk_token_id} to pad_token_id")
        tokenizer.pad_token_id = tokenizer.unk_token_id # 这段代码确保了分词器（tokenizer）具有一个 pad_token_id，
    print(f'memory usage of model: {model.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model("baichuan", "./pretrained/baichuan-7b", "4bit", 0)
    print(tokenizer.unk_token_id)
