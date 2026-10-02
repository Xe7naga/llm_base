from transformers import AutoTokenizer
from dataclasses import dataclass
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")

@dataclass
class SFTConfig:
    lr:float = 3e-5
    log_iter:int = 100
    log_dir:str = "logs/02_sft_demo"
    train_data_size: int =200
    batch_size:int = 4
    warmup_ratio:float = 0.1
    save_dir:str = "finetuned/02_sft_demo"

from typing import List
def get_train_data(sft_config:SFTConfig):
    """
    加载训练数据，通过tokenizer，调用chat_template，把数据转换成可训练的格式
    """
    from datasets import load_dataset
    train_data = load_dataset("./data/ultrachat_200k")["train_sft"]
    result_list = []
    for i in range(sft_config.train_data_size):
        message_list = train_data[i]["messages"]

        result:List = tokenizer.apply_chat_template(message_list,tokenize=True,truncation = True,max_length = 2900)["input_ids"]
        result_list.append(result)

    return result_list


from transformers import PreTrainedTokenizerFast
from typing import List
def create_answer_mask(labels,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从labels当中找出assistant回答的部分，然后输出一个与labels相同shape的mask
    """

    
    # 构建answer mask，输入的labels为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和labels相同shape的全0矩阵
    # labels: batch_size假设为2
    answer_mask = torch.zeros_like(labels)

    # 2、找到<|im_end|> 所对应的token_id
    eos_token_id = tokenizer.encode("<|im_end|>")[0]

    # 3、遍历labels中的每一个样本
    # labels.shape: batch_size, seq_len
    for idx,ids in enumerate(labels):
        # 3.1、获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()
        # 3.2、解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 3.3、设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 4、结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取，每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[10,20]
        assistant_ends:[15,25]
    """

    use_ends = [pos for pos in eos_positions[::2]]
    assistant_ends = [pos for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么user_turns:[10,20]，assistant_ends:[15,25]

    
    要想获取到assistant的回答的起始位置，就需要跳过<|im_end|>, \n, <|im_start|>,assistant , \n 这5个token
    要想获取到assistant的回答的结束位置，需要将<|im_end|>也包括进去，又因为列表切片是左闭右开的，所以需要向后移动一位
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    # 多轮对话没有被截断或者最后一轮整个assistant回答被截断，user轮数和assistant轮数一致
    if num_user_turns == num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start = user_end + 5
            answer_end = assistant_end + 1
            mask[answer_start:answer_end] = 1

    # 最后一轮，assistant回答被部分截断，此时user轮数比assistant轮数多一轮
    elif num_user_turns == num_assistant_turns + 1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start = user_end + 5
            answer_end = assistant_end + 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1] 
        last_answer_start = last_user_end + 5
        mask[last_answer_start:] = 1


import torch
def compute_loss(output_logits, target_labels,assistant_answer_mask):
    """
    计算一个批次数据的损失，
    output_logits: 模型前向传播后输出的logits，shape: [batch_size, seq_len, vocab_size]
    target_labels: 当前这个批次的答案 token_ids ， shape:[batch_size, seq_len]
    assistant_answer_mask: assistant回答的掩码，shape:[batch_size, seq_len]
    """
    # 1、基于output_logits，算得对数概率分布
    # log_probs: [batch_size, seq_len, vocab_size]
    log_probs = torch.log_softmax(output_logits, dim = -1)
    
    # 2、基于对数概率分布，以及torch.gather算子，找到模型输出实际答案的token的对数概率
    # labels_log_probs: [batch_size, seq_len]
    labels_log_probs = torch.gather(
        log_probs,
        dim = -1,
        index = target_labels.unsqueeze(-1)

    ).squeeze(-1)
    
    # 3、对assistant answer 做一个掩码，将需要计算损失的地方，log_probs保持为原值，不需要计算的地方，置为0
    masked_labels_log_probs = labels_log_probs * assistant_answer_mask
    # negative_masked_labels_log_probs: batch_size,seq_len
    negative_masked_labels_log_probs = (-1) * masked_labels_log_probs

    #  average_loss : 标量，表示的这个批次的平均损失
    average_loss = negative_masked_labels_log_probs.sum() / assistant_answer_mask.sum()



    return average_loss




    

import numpy as np
def cosine_decay(batch, total_batch, lr, warmup_ratio):

    warmup_batch = total_batch * warmup_ratio

    if batch< warmup_batch:
        # 基于y = kx 线性预热
        k  = lr / warmup_batch
        x = batch
        return k * x

    else:
        # 衰减进度，从0到1
        progress = (batch - warmup_batch) / (total_batch - warmup_batch)
        # 余弦衰减： 0.5*(1+cos(π * progress)) progress从0到1的过程，cos值从最大值到最小值
        decay = 0.5 * (1+np.cos( np.pi * progress))
        return decay * lr 
    


from torch.optim.adamw import AdamW
import tqdm
from torch.utils.tensorboard import SummaryWriter




def train(sft_config:SFTConfig):
    """
    主训练逻辑
    """
    # 1、加载模型
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B-Base")

    model.train()
    model.to("cuda")

    # 2、获取数据
    train_data = get_train_data(sft_config)

    # 3、算出总共有多少批次
    total_batch = (len(train_data) + sft_config.batch_size - 1) // sft_config.batch_size

    # 4、构造优化器
    optimizer = AdamW(model.parameters(),lr = sft_config.lr)

    # 5、训练的可观测性
    progress_bar = tqdm.tqdm(total = total_batch)
    total_loss = []
    writer = SummaryWriter(log_dir=sft_config.log_dir)
    for i in range(total_batch):

        # 1、构造数据
        batch_train_data = train_data[i * sft_config.batch_size :(i+1) * sft_config.batch_size]

        max_batch_length = max([len(data) for data in  batch_train_data])

        for data in batch_train_data:
            padding_length = max_batch_length - len(data)
            data.extend([tokenizer.pad_token_id] * padding_length)

        data_tensor = torch.tensor(batch_train_data,dtype=torch.long).to("cuda")
        input_ids = data_tensor[:,:-1]
        labels = data_tensor[:,1:]

        assistant_answer_mask= create_answer_mask(labels,tokenizer)

        # 2、前向传播

        output_logits = model(input_ids).logits

        loss = compute_loss(output_logits, labels, assistant_answer_mask)

        # 3、反向传播，计算梯度

        loss.backward()
        total_loss.append(loss.item())

        # 4、优化器更新参数

        current_learning_rate = cosine_decay(i, total_batch, sft_config.lr, sft_config.warmup_ratio)
        writer.add_scalar(tag="train_learning_rate",scalar_value=current_learning_rate,global_step=i)
        optimizer.param_groups[0]["lr"] = current_learning_rate

        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{total_loss[-1]:.4f}", lr=f"{current_learning_rate:.2e}")


        should_log = i % sft_config.log_iter == 0 or i == total_batch -1

        if should_log :
            current_log_loss = total_loss[-sft_config.log_iter:]
            average_loss = sum(current_log_loss) / len(current_log_loss) # 或者除以 sft_confg.log_iter           
            writer.add_scalar(tag="train_loss",scalar_value=average_loss,global_step=i)

    return model,tokenizer


def save_model_tokenizer(model,tokenizer,save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main():

    sft_config = SFTConfig()
    model, tokenizer = train(sft_config)
    save_model_tokenizer(model,tokenizer,sft_config.save_dir)


if __name__=="__main__":
    main()