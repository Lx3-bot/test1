import os
import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
from dataclasses import field, fields, dataclass
import bitsandbytes as bnb

from model import load_model
from dataset import belle_open_source_500k


### 定义一些配置信息
@dataclass
class FinetuneArguments:
    model_name: str = field()
    model_path: str = field()
    data_name: str = field()
    data_path: str = field()
    train_size: int = field(default=-1)
    test_size: int = field(default=200)
    max_len: int = field(default=1024)
    lora_rank: int = field(default=8)#低秩矩阵的秩
    lora_modules: str = field(default=None) #指定对什么层进行Lora配置，默认是全部he
    quantization: str = field(default="4bit")

# 因为一般对线性层进行Lora配置，因此首先需要找到线性层
def find_all_linear_names(model):
    # cls = bnb.nn.Linear8bitLt
    cls = bnb.nn.Linear4bit # 这行代码定义了要查找的特定线性层类型，这里是4位精度的线性层。
    lora_module_names = set() # 创建一个集合，用于存储线性层名称
    for name, module in model.named_modules():# 遍历模型的所有模块及其名称。named_modules() 方法返回模型中所有子模块的名称和模块对象。
        if isinstance(module, cls):# 检查当前的model是否是cls的实例，实际上就是看当前模块是否使用了线性层
            names = name.split('.')# 因为模块名称可能包含层级的名称，所以需要拆分为列表。
            lora_module_names.add(names[0] if len(names) == 1 else names[-1]) 
    #上面这段添加模块名的逻辑中，首先需要确定当前查找到的模块是否不包含其他子模块，即==1，那么就将该模块添加进去
    #如果包含子模块，就需要将最后一个子模块名称添加进去，因为一般来说我们只关注最后一个模块名，比如encoder.layer.0.mlp.fc1


    #因为在量化的时候会将全部线性层进行量化，而输出层需要较高精度，不能4bit输出，并且这一嗯不适合进行LoRA微调，因此需要将lm_head从lora_module_names中移除
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names) # 返回找到的线性层名称列表
def main():
    # 解析这两个配置了和配置文件中的配置信息，将和微调相关的配置参数放入args，将和训练相关的配置放到training_args中
    args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    set_seed(training_args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"world size {world_size} local rank {local_rank}")

    ####### prepare model ############
    #模型加载
    model, tokenizer = load_model(args.model_name, args.model_path, args.quantization, local_rank)
    #模型准备
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)# 找到全部被量化的线性层，以供配置lora
    target_modules = args.lora_modules.split(",") if args.lora_modules is not None else modules #全部需要配置lora的线性层

#lora的一些配置型相关的信息
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    print(config)
    model = get_peft_model(model, config)#得到lora挂载模型

    #计算可训练的参数个数：
    print_trainable_parameters(model)

    ############# prepare data ###########
    data = eval(args.data_name)(args.data_path, tokenizer, args.max_len)
    if args.train_size > 0:
        data = data.shuffle(seed=training_args.seed).select(range(args.train_size))

    if args.test_size > 0:
        train_val = data.train_test_split(
            test_size=args.test_size, shuffle=True, seed=training_args.seed
        )
        train_data = train_val["train"].shuffle(seed=training_args.seed)
        val_data = train_val["test"].shuffle(seed=training_args.seed)
    else:
        train_data = data['train'].shuffle(seed=training_args.seed)
        val_data = None

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer,
                                                          pad_to_multiple_of=8,
                                                          return_tensors="pt",
                                                          padding=True),
    )
    trainer.train(resume_from_checkpoint=False)
     #在 trainer.train() 调用之后添加以下代码
     #model = model.merge_and_unload()
    model.save_pretrained(training_args.output_dir)#这里只是保存了进行lora微调之后的部分权重，但是和没有和原始模型的权重进行合并

if __name__ == "__main__":
    main()