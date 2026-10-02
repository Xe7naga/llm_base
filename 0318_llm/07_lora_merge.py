"""
LoRA参数合并
"""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--base_model_path",type=str, help="基座模型的路径")
parser.add_argument("--adapter_path",type=str,help="适配器(A,B)权重路径")
parser.add_argument("--merged_model_path",type=str,help="合并之后的模型保存路径")

args = parser.parse_args()

base_model_path = args.base_model_path
adapter_path = args.adapter_path
merged_model_path = args.merged_model_path

# 1、加载base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 2、通过PeftModel.from_pretrained，加载适配器
from peft import PeftModel

peft_model = PeftModel.from_pretrained(model=base_model,model_id=adapter_path)


# 3、调用peft_model merge_and_unload方法，进行合并

merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)