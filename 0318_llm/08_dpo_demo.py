"""
DPO Lora微调
"""
from pickle import TRUE
from pandas import ExcelFile
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig,get_peft_model

train_dataset = load_dataset("./data/ultrafeedback_binarized")["train_prefs"]

train_dataset = train_dataset.remove_columns(['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])
train_dataset = train_dataset.select(range(30000))
test_dataset = load_dataset("./data/ultrafeedback_binarized")["test_prefs"]
test_dataset = test_dataset.remove_columns(['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])

from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/08_dpo_demo"
dpo_config = DPOConfig(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    max_steps=500,
    logging_strategy="steps",
    logging_steps=50,
    report_to="tensorboard",
    # DPO的学习率，会比SFT的会更小，因为学习的数据相对偏好，优化过程会更难，需要通过小学习率去学习
    # DPO全参微调：1e-7 - 1e-6 区间之内， DPO LoRA微调：1e-6 - 5e-6
    learning_rate=3e-6,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    output_dir="finetuned/08_dpo_demo",
    bf16=True,
    gradient_checkpointing=True,
    max_length=1400,
    # DPOConfig没有这个参数，但是在底层，DPOTrainer能够识别到prompt部分，不会对这部分计算损失
    # assistant_only_loss=True

    # DPOConfig里面额外的参数有哪些？
    beta= 0.1,
    max_prompt_length=1000
)

model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B")
model.warnings_issued = {}
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type= "CAUSAL_LM"
)

peft_model = get_peft_model(model,lora_config)

trainer = DPOTrainer(
    model=peft_model,
    # 传递为None，表示参考模型使用和被训练的模型，一样的模型
    ref_model=None,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer
)


trainer.train()
trainer.save_model("finetuned/08_dpo_demo")



