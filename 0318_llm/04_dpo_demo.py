from transformers import AutoTokenizer
from dataclasses import dataclass
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")

@dataclass
class DPOConfig:
    lr:float = 3e-5
    log_iter:int = 100
    log_dir:str = "logs/02_sft_demo"
    train_data_size: int =200
    batch_size:int = 4
    warmup_ratio:float = 0.1
    save_dir:str = "finetuned/02_sft_demo"
    beta:float = 0.1

from typing import List
def get_train_data(dpo_config:DPOConfig):
    """
    加载训练数据，通过tokenizer，调用chat_template，把数据转换成可训练的格式
    """
    from datasets import load_dataset
    train_data = load_dataset("./data/ultrafeedback_binarized")["train_prefs"]
    chosen_result_list = []
    rejected_result_list = []

    for i in range(dpo_config.train_data_size):

        chosen_message_list = train_data[i]["chosen"]

        chosen_result:List = tokenizer.apply_chat_template(chosen_message_list,tokenize=True,truncation = True,max_length = 2900)["input_ids"]
        chosen_result_list.append(chosen_result)



        rejected_message_list = train_data[i]["rejected"]

        rejected_result:List = tokenizer.apply_chat_template(rejected_message_list,tokenize=True,truncation = True,max_length = 2900)["input_ids"]
        rejected_result_list.append(rejected_result)



    return chosen_result_list,rejected_result_list


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
def compute_loss(chosen_log_probs, rejected_log_probs, ref_chosen_log_probs,ref_rejected_log_probs,beta):
    """
    计算一个批次数据的损失，
    chosen_log_probs: 被训练的模型，输出喜欢回答的对数概率，shape:[batch_size], chosen_log_probs[0]，表示模型基于第0个样本的prompt，输出喜欢回答的对数概率
    rejected_log_probs： 被训练的模型，输出拒绝回答的对数概率，shape:[batch_size], chosen_log_probs[0]，表示模型基于第0个样本的prompt，输出拒绝回答的对数概率
    ref_chosen_log_probs：参考模型，输出喜欢回答的对数概率，shape:[batch_size], chosen_log_probs[0]，表示参考模型基于第0个样本的prompt，输出喜欢回答的对数概率
    ref_rejected_log_probs: 参考模型,输出拒绝回答的对数概率，shape:[batch_size], chosen_log_probs[0]，表示参考模型基于第0个样本的prompt，输出拒绝回答的对数概率
    """

    margin = chosen_log_probs - rejected_log_probs - (ref_chosen_log_probs - ref_rejected_log_probs)
    
    # result.shape:[batch_size,]
    result = (-1) * torch.nn.functional.logsigmoid( beta * margin)

    # 计算批次的平均损失
    average_loss = result.sum() / len(result)


    return average_loss


def compute_log_probs(output_logits,labels,assistant_answer_mask):
    """
    计算模型输出 某个回答的对数概率
    output_logits: 模型前向传播所得的结果，shape: batch_size, seq_len, vocab_size
    labels: 实际答案，shape: batch_size, seq_len
    assistant_answer_mask, shape: batch_size, seq_len
    """

    log_probs = torch.log_softmax(output_logits, dim = -1)
    
    # 2、基于对数概率分布，以及torch.gather算子，找到模型输出实际答案的token的对数概率
    # labels_log_probs: [batch_size, seq_len]
    labels_log_probs = torch.gather(
        log_probs,
        dim = -1,
        index = labels.unsqueeze(-1)

    ).squeeze(-1)
    
    # 3、对assistant answer 做一个掩码，将需要计算损失的地方，log_probs保持为原值，不需要计算的地方，置为0
    # masked_labels_log_probs: shape, batch_size, seq_len
    masked_labels_log_probs = labels_log_probs * assistant_answer_mask


    # 通过.sum，传入dim=-1，计算得到，每个样本，模型输出回答的对数概率，
    # log_probs: shape: batch_size
    # 除以样本中的有效token数量，可做可不做，看具体库的实现
    log_probs = masked_labels_log_probs.sum(dim = -1) / assistant_answer_mask.sum(dim = -1)


    return log_probs

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

def train(dpo_config: DPOConfig):
    """
    主训练逻辑
    """
    # 1、加载模型
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("finetuned/02_sft_demo")
    ref_model = AutoModelForCausalLM.from_pretrained("finetuned/02_sft_demo")


    model.train()
    model.to("cuda")

    ref_model.eval()
    ref_model.to("cuda")

    # 2、获取数据
    chosen_train_data, rejected_train_data = get_train_data(dpo_config)

    # 3、算出总共有多少批次
    total_batch = (len(chosen_train_data) + dpo_config.batch_size - 1) // dpo_config.batch_size

    # 4、构造优化器
    optimizer = AdamW(model.parameters(),lr = dpo_config.lr)

    # 5、训练的可观测性
    progress_bar = tqdm.tqdm(total = total_batch)
    total_loss = []
    writer = SummaryWriter(log_dir=dpo_config.log_dir)
    for i in range(total_batch):

        # 1、构造数据
        # 1.1、构造chosen： chosen_input_ids, chosen_labels,chosen_assistant_answer_mask
        batch_chosen_train_data = chosen_train_data[i * dpo_config.batch_size :(i+1) * dpo_config.batch_size]

        chosen_max_batch_length = max([len(data) for data in  batch_chosen_train_data])

        for data in batch_chosen_train_data:
            padding_length = chosen_max_batch_length - len(data)
            data.extend([tokenizer.pad_token_id] * padding_length)

        chosen_data_tensor = torch.tensor(batch_chosen_train_data,dtype=torch.long).to("cuda")
        chosen_input_ids = chosen_data_tensor[:,:-1]
        chosen_labels = chosen_data_tensor[:,1:]

        chosen_assistant_answer_mask= create_answer_mask(chosen_labels,tokenizer)

        # 1.1、构造rejected： rejected_input_ids, rejected_labels,rejected_assistant_answer_mask
        batch_rejected_train_data = rejected_train_data[i * dpo_config.batch_size :(i+1) * dpo_config.batch_size]

        rejected_max_batch_length = max([len(data) for data in  batch_rejected_train_data])

        for data in batch_rejected_train_data:
            padding_length = rejected_max_batch_length - len(data)
            data.extend([tokenizer.pad_token_id] * padding_length)

        rejected_data_tensor = torch.tensor(batch_rejected_train_data,dtype=torch.long).to("cuda")
        rejected_input_ids = rejected_data_tensor[:,:-1]
        rejected_labels = rejected_data_tensor[:,1:]

        rejected_assistant_answer_mask= create_answer_mask(rejected_labels,tokenizer)


        # 2、前向传播

        # 2.1、训练模型，基于chosen和rejected数据的两次前向传播
        # shape: batch_size, seq_len, vocab_size
        chosen_output_logits = model(chosen_input_ids).logits
        rejected_output_logits = model(rejected_input_ids).logits

        # 2.2、参考模型，基于chosen和rejected数据的两次前向传播

        # 参考模型，不需要做参数更新，前向传播过程，放到torch.no_grad上下文管理器中，避免构建计算图
        with torch.no_grad():
            ref_chosen_output_logits = ref_model(chosen_input_ids).logits
            ref_rejected_output_logits = ref_model(rejected_input_ids).logits

        
        chosen_log_probs = compute_log_probs(
            output_logits=chosen_output_logits,
            labels=chosen_labels,
            assistant_answer_mask=chosen_assistant_answer_mask
            )
        
        rejected_log_probs = compute_log_probs(
            output_logits=rejected_output_logits,
            labels=rejected_labels,
            assistant_answer_mask=rejected_assistant_answer_mask
        )

        ref_chosen_log_probs = compute_log_probs(
            output_logits=ref_chosen_output_logits,
            labels=chosen_labels,
            assistant_answer_mask=chosen_assistant_answer_mask
        )

        ref_rejected_log_probs = compute_log_probs(
            output_logits=ref_rejected_output_logits,
            labels=rejected_labels,
            assistant_answer_mask=rejected_assistant_answer_mask
        )


        loss = compute_loss(
            chosen_log_probs=chosen_log_probs,
            rejected_log_probs=rejected_log_probs,
            ref_chosen_log_probs=ref_chosen_log_probs,
            ref_rejected_log_probs=ref_rejected_log_probs,
            beta=dpo_config.beta
        )

        # 3、反向传播，计算梯度

        loss.backward()
        total_loss.append(loss.item())

        # 4、优化器更新参数

        current_learning_rate = cosine_decay(i, total_batch, dpo_config.lr, dpo_config.warmup_ratio)
        writer.add_scalar(tag="train_learning_rate",scalar_value=current_learning_rate,global_step=i)
        optimizer.param_groups[0]["lr"] = current_learning_rate

        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{total_loss[-1]:.4f}", lr=f"{current_learning_rate:.2e}")


        should_log = i % dpo_config.log_iter == 0 or i == total_batch -1

        if should_log :
            current_log_loss = total_loss[-dpo_config.log_iter:]
            average_loss = sum(current_log_loss) / len(current_log_loss) # 或者除以 sft_confg.log_iter           
            writer.add_scalar(tag="train_loss",scalar_value=average_loss,global_step=i)

    return model,tokenizer


def save_model_tokenizer(model,tokenizer,save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def main():

    dpo_config = DPOConfig()
    model, tokenizer = train(dpo_config)
    save_model_tokenizer(model,tokenizer,dpo_config.save_dir)


if __name__=="__main__":
    main()