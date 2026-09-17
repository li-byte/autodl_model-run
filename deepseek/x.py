from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = r"E:\models\Qwen2.5-3B-Instruct-GPTQ-Int4"

# 加载
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 单次提问
prompt = "你好，请介绍一下自己"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)