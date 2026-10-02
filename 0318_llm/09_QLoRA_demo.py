from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
bits_and_bytes_config = BitsAndBytesConfig(
    # 1、是否开启4bit量化
    load_in_4bit=True,
    # 2、4bit量化的方式，使用nf4
    bnb_4bit_quant_type="nf4",
    # 3、是否开启双重量化
    bnb_4bit_use_double_quant=False,

    # 4、反量化时的数据类型
    bnb_4bit_compute_dtype= torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B",quantization_config = bits_and_bytes_config)


from peft import prepare_model_for_kbit_training

# 1、将模型当中敏感层的数据类型，置为高精度，fp32，包括 RMSNorm 输出层
# 2、会做和get_peft_model方法类似事情：将base model当中线性层，所有的权重冻结，requires_grad置为False
model = prepare_model_for_kbit_training(model=model)


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)


tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-8B")


#1、第一步：对数据集进行处理，处理成trl所需要的language modeling类型的对话格式
from datasets import load_dataset,Dataset,DatasetDict
# 1.1、将两个数据集文件，分别加载到dataset dict当中的train 和test分片中
dataset_dict:DatasetDict = load_dataset("json",data_files = {"train":"data/psychology_data.jsonl"})
dataset_dict["train"] = dataset_dict["train"].select(range(5000))
dataset_dict:DatasetDict = dataset_dict["train"].train_test_split(test_size=0.2)

from typing import List,Dict
# 1.2 、定义数据处理的 mapping function
def convert_function(examples:Dict[str, List]):
    """
    数据集分批次调用该方法，修改数据格式
    """
    conversation_list:List[List[Dict]] = examples["conversation"]
    message_list = []
    for data in conversation_list:
        # data是单条样本
        coversation_dict = data[0]
        current_data_message_list = [
            {"role":"user","content":coversation_dict["human"]},
            {"role":"assistant","content":coversation_dict["assistant"]}
        ]
        message_list.append(current_data_message_list)
    
    return {"messages":message_list}

# 1.3、调用dataset_dict.map方法，传入mapping function，对数据进行格式化
mapped_dataset_dict = dataset_dict.map(convert_function,batched=True,remove_columns = ['conversation_id', 'category', 'conversation', 'dataset'])


# 2、构造SFTTrainer实例
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
# 2.1、构造训练时的参数SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/09_QLoRA_demo"
sft_config = SFTConfig(
    # 数据规模相关
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=160,
    num_train_epochs=1,
    # 训练日志相关
    logging_strategy="steps",
    logging_steps=40,
    report_to="tensorboard",
    # 学习率和优化器相关
    # 注意： LoRA微调的学习率，会比全参微调，高一个数量级左右
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    optim="paged_adamw_32bit",
    # 评估和保存相关
    eval_strategy="steps",
    eval_steps=40,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=80,
    save_total_limit=2,
    output_dir="finetuned/09_QLoRA_demo",
    # 优化相关
    bf16=True,
    gradient_checkpointing=True,
    activation_offloading=True,
    max_length=700,

    # 其他参数：是否只对assistant answer计算损失、chat_template_path
    assistant_only_loss=True,
    chat_template_path= "./test_chat_template.jinja"
    
)

trainer = SFTTrainer(
    model = peft_model,
    processing_class = tokenizer,
    train_dataset = mapped_dataset_dict["train"],
    eval_dataset = mapped_dataset_dict["test"],
    args = sft_config
)

trainer.train()
trainer.save_model("finetuned/09_QLoRA_demo")