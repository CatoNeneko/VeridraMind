import json

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
from typer import prompt


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。

    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


class PretrainDataset(Dataset):


    def __init__(self, data_path, tokenizer, max_seq_len=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len


        self.samples = load_dataset('json', data_files=data_path, split='train')


    def load_data(self, path):

        samples = []
        with open(path, 'r', encoding='utf-8') as f:

            for line_num, line in enumerate(f,1):

                data = json.loads(line.strip())
                samples.append(data)

        return samples


    def __len__(self):

        return len(self.samples)









    def __getitem__(self, index):

        sample = self.samples[index]


        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len - 2,
        ).input_ids




        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))

        input_ids = torch.tensor(input_ids, dtype=torch.long)


        labels = input_ids.clone()
        labels[len(tokens):] = -100


        attention_mask = torch.arange(self.max_seq_len).lt(len(tokens)).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class SFTDataset(Dataset):

    def __init__(self, jsonl_path, tokenizer, max_seq_len=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids

        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)



    def creat_chat_prompt(self, conversations):

        messages = [message.copy() for message in conversations]
        tools = None
        if messages and messages[0].get("role") == "system":

            tools = messages[0].get("functions") or messages[0].get("tools")
            if isinstance(tools, str):

                tools = json.loads(tools)


        for message in messages:
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, str):

                message["tool_calls"] = json.loads(tool_calls)


        template_args = {
            "tokenize": False,
            "add_generation_prompt": False,
        }

        if tools is not None:
            template_args["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **template_args)


    def generate_labels(self, input_ids):


        labels = [-100]*len(input_ids)
        i = 0

        while i < len(input_ids):

            if input_ids[i:i+len(self.bos_id)] == self.bos_id:

                start = i + len(self.bos_id)

                end = start
                while end < len(input_ids):

                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1


                response_end = (
                    min(end + len(self.eos_id), len(input_ids))
                    if end < len(input_ids)
                    else len(input_ids)
                )

                for j in range(start, response_end):
                    labels[j] = input_ids[j]


                i = max(i + 1, response_end)
            else:
                i += 1


        return labels



    def __getitem__(self, index):
        sample = self.samples[index]


        conversation = pre_processing_chat(sample['conversations'])


        prompt= self.creat_chat_prompt(conversation)


        prompt = post_processing_chat(prompt)


        input_ids = self.tokenizer(prompt).input_ids[:self.max_seq_len]


        labels = self.generate_labels(input_ids)

        padding_len = self.max_seq_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len

        labels += [-100] * padding_len

        attention_mask = [1] * (self.max_seq_len - padding_len) + [0] * padding_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


class RLHFDataset(Dataset):

    def __init__(self, jsonl_path, tokenizer, max_length = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)





    def create_chat_prompt(self, conversations):

        messages =[]

        answer = ""


        for i, turn in enumerate(conversations):

            role = 'user' if i%2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn["content"]})
            answer = turn['content']


        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        prompt = post_processing_chat(prompt)


        return prompt, answer


    def __getitem__(self, index):
        sample = self.samples[index]


        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {'prompt': prompt, 'answer': answer}



class DPODataset(Dataset):

    def __init__(self, jsonl_path, tokenizer, max_length = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        sample = self.samples[index]

        chosen = sample['chosen']
        rejected = sample['rejected']


        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        chosen_prompt = post_processing_chat(chosen_prompt)


        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = post_processing_chat(rejected_prompt)


        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )


        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )


        chosen_input_ids = chosen_encoding['input_ids']

        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)


        rejected_input_ids = rejected_encoding['input_ids']

        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)


        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)

        chosen_loss_mask = [m * a for m, a in zip(chosen_loss_mask, chosen_encoding['attention_mask'])]
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        rejected_loss_mask = [m * a for m, a in zip(rejected_loss_mask, rejected_encoding['attention_mask'])]
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)


        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,

            'attention_mask_chosen': torch.tensor(chosen_encoding['attention_mask'][:-1], dtype=torch.long),
            'attention_mask_rejected': torch.tensor(rejected_encoding['attention_mask'][:-1], dtype=torch.long),
            'mask_rejected': mask_rejected
        }


    def generate_loss_mask(self, input_ids):
        """
        生成loss mask，确保只有assistant的回复部分参与loss计算。
        """

        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1


                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1


        return loss_mask

