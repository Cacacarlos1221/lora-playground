# lora_distilgpt2_demo.py
# 目标：在 CPU 上用 LoRA 微调 distilgpt2，跑一个最小 demo

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def main():
    # 1. 设备选择：你现在是 CPU，这里只是通用写法
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 2. 准备一个小小的玩具数据集（几句中文/英文都行）
    train_texts = [
        "LoRA is a parameter-efficient fine-tuning method.",
        "We are fine-tuning a small GPT-2 model on CPU.",
        "This is just a tiny demo to understand the pipeline.",
        "Large language models can be adapted with LoRA.",
        "Transformers and PEFT make LoRA easy to use.",
        "微调大模型不一定要改所有参数，LoRA 只改一小部分。",
        "现在我们在 CPU 上做一个教学用的 LoRA demo。",
        "如果以后有 GPU，可以把同样流程换到大模型上。",
        "数据集很小，主要是看懂训练链路。",
        "loss 能正常下降，就说明链路是通的。",
    ]

    dataset = Dataset.from_dict({"text": train_texts})

    # 3. 加载 tokenizer 和基础模型（distilgpt2 很小，CPU 也扛得住）
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 家族没有 pad_token，设成 eos_token 以方便 Trainer 处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # 4. 文本 -> token
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    # 因为是自回归语言模型，标签和输入一样
    def add_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch

    tokenized_ds = tokenized_ds.map(add_labels, batched=True)

    # 5. 配置 LoRA，只对 attention 相关的投影层加 adapter
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],  # GPT-2 系列的注意力层名字
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 看看 LoRA 只开放了少量参数

    # 6. Trainer 参数（CPU 上就别搞太重）
    training_args = TrainingArguments(
        output_dir="./lora-distilgpt2-output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=20,
        save_total_limit=1,
        fp16=False,  # CPU 上关掉混合精度
        report_to="none",
    )

    # 7. 构建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
    )

    # 8. 开始训练（小数据集 + 1 epoch，CPU 也能顶得住）
    trainer.train()

    # 9. 保存 LoRA adapter（注意：只保存 LoRA 而不是整个基础模型）
    adapter_path = "./lora-distilgpt2-output/adapter"
    model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
