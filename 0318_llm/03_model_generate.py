from transformers import AutoModelForCausalLM, AutoTokenizer
# 通过argparser，从命令行接收参数，避免将参数，写死在脚本当中
from argparse import ArgumentParser
from typing import Dict
from torch import Tensor
# 1、定义argparser
# 1.1、构造parser实例
parser = ArgumentParser()

# 1.2、定义命令行可以接收哪些参数
parser.add_argument("--model_path",type=str,help="推理模型的参数路径")
parser.add_argument("--prompt",type=str,help="需要进行推理的prompt")


# 1.3、接收命令行传过来的参数
args = parser.parse_args()

# 2、加载模型和tokenizer
model_path  = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_path,device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


# 3、构造输入的token ids 序列
prompt = args.prompt
message_list = [{"role":"user","content":prompt}]
tokenized_result:Dict[str,Tensor] = tokenizer.apply_chat_template(message_list,tokenize=True,add_generation_prompt = True,enable_thinking=True,return_tensors = "pt")
input_ids = tokenized_result["input_ids"].to("cuda")
attention_mask = tokenized_result["attention_mask"].to("cuda")

# 4、让模型做自回归生成
# 调用model.generate方法，内部封装了，自回归生成的过程
# result: 数据，自回归生成的结果，张量类型，带批次的维度，张量里面所有的token ids：既包含输入的token序列，也包含生成的token序列
result = model.generate(input_ids,max_new_tokens = 500,eos_token_id = [151645,151643])
# 5、处理输出
# 5.1、取出第0个元素，也就是我们输入的这个样本
token_ids = result[0]
# 5.2、从result里面去掉输入的部分
output_tokens = token_ids[len(input_ids[0]):]

decoded_result = tokenizer.decode(output_tokens,skip_special_tokens=True)
print("-----------最终解码之后的结果为：\n\n")
print(decoded_result)



