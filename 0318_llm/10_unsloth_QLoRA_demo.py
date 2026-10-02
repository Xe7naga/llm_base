# 1、需要将Unsloth的import放在最前，确保unsloth能够将所有优化措施全部都使用上
from unsloth import FastLanguageModel
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from datasets import Dataset,DatasetDict, load_dataset
# 2、先通过FastLanguageModel.from_pretrained，对模型进行量化
model, tokenizer = FastLanguageModel.from_pretrained(
    # 想让Unsloth加载本地模型，需要使用：./xxxx 这种方式来写路径，不能直接写 model/Qwen3-8B
    model_name="./model/Qwen3-8B",
    load_in_4bit=True,
    # 额外添加下面两个参数，让Unsloth只加载本地模型的参数其他配置文件
    use_exact_model_name=True,
    local_files_only=True
    )



# 3、再通过FastLanguageModel.get_peft_model，对量化后的模型，插入可训练的A,B矩阵
model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    # target_modules="all-linear"
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)


dataset_dict:DatasetDict = load_dataset("json",data_files = {"train":"data/psychology_data.jsonl"})
dataset_dict["train"] = dataset_dict["train"].select(range(5000))
dataset_dict:DatasetDict = dataset_dict["train"].train_test_split(test_size=0.2)

from typing import List,Dict
def convert_function(examples:Dict[str, List]):
    """
    数据集分批次调用该方法，修改数据格式
    """
    conversation_list:List[List[Dict]] = examples["conversation"]
    text_list = []
    for data in conversation_list:
        # data是单条样本
        coversation_dict = data[0]
        current_data_message_list = [
            {"role":"user","content":coversation_dict["human"]},
            {"role":"assistant","content":coversation_dict["assistant"]}
        ]
        # Unsloth需要接收，tokenzier.apply_chat_template之后的文本结果
        text_list.append(tokenizer.apply_chat_template(current_data_message_list,tokenize=False, add_generation_prompt=False))

    
    return {"text":text_list}

mapped_dataset_dict = dataset_dict.map(convert_function, batched=True, remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])


from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/10_unsloth_demo"
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
    output_dir="finetuned/10_unsloth_demo",
    # 优化相关
    bf16=True,
    gradient_checkpointing=True,
    # 注意，此处不能传递为True，否则会报错
    activation_offloading=False,
    max_length=700,
    
)

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = mapped_dataset_dict["train"],
    eval_dataset = mapped_dataset_dict["test"],
    args = sft_config
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"

)

trainer.train()

trainer.save_model("finetuned/10_unsloth_demo")

model.save_pretrained_merged("./finetuned/Qwen3-8B-SFT-unsloth-merged", tokenizer, save_method="merged_16bit")