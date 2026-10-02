from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer
lora_config = LoraConfig(
    r=16,
    # target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]
    target_modules = "all-linear",
    lora_alpha= 16,
    lora_dropout=0.05,
)
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B")

peft_model = get_peft_model(model,lora_config)

tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")


#1、第一步：对数据集进行处理，处理成trl所需要的language modeling类型的对话格式
from datasets import load_dataset
# 1.1、将两个数据集文件，分别加载到dataset dict当中的train 和test分片中
dataset_dict = load_dataset("json",data_files = {"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})

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
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/06_lora_demo"
sft_config = SFTConfig(
    # 数据规模相关
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=500,
    num_train_epochs=1,
    # 训练日志相关
    logging_strategy="steps",
    logging_steps=50,
    report_to="tensorboard",
    # 学习率和优化器相关
    # 注意： LoRA微调的学习率，会比全参微调，高一个数量级左右
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    # optim=
    # 评估和保存相关
    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    output_dir="finetuned/06_lora_demo",
    # 优化相关
    bf16=True,
    gradient_checkpointing=True,
    activation_offloading=True,
    max_length=700,

    # 其他参数：是否只对assistant answer计算损失、chat_template_path
    assistant_only_loss=True,
    chat_template_path= "./chat_template.jinja"
    
)

trainer = SFTTrainer(
    model = peft_model,
    processing_class = tokenizer,
    train_dataset = mapped_dataset_dict["train"],
    eval_dataset = mapped_dataset_dict["test"],
    args = sft_config
)

trainer.train()
trainer.save_model("finetuned/06_lora_demo")