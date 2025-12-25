# demos/infer_lora_distilgpt2.py
# 对比：原始 distilgpt2 vs LoRA 微调后的模型

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def generate_with(model, tokenizer, prompt, device, max_new_tokens=50):
    """给定模型和 prompt，生成一段文本。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    base_name = "distilgpt2"
    adapter_dir = "./lora-distilgpt2-output/adapter"

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 原始 base 模型
    base_model = AutoModelForCausalLM.from_pretrained(base_name)
    base_model.to(device)
    base_model.eval()

    # 3. Base + LoRA adapter 组合成“LoRA 模型”
    lora_model = AutoModelForCausalLM.from_pretrained(base_name)
    lora_model = PeftModel.from_pretrained(lora_model, adapter_dir)
    lora_model.to(device)
    lora_model.eval()

    # 4. 选一个和训练数据风格接近的 prompt
    prompt = "LoRA is a parameter-efficient fine-tuning method because"
    print("Prompt:\n", prompt)

    print("\n=== Base model output ===")
    print(generate_with(base_model, tokenizer, prompt, device))

    print("\n=== LoRA model output ===")
    print(generate_with(lora_model, tokenizer, prompt, device))


if __name__ == "__main__":
    main()
