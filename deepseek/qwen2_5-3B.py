from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':

    model_name = "E:\models\Qwen2.5-3B-Instruct-GPTQ-Int4"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "请简要介绍一下大型语言模型。"
    messages = [
        {"role": "system", "content": "你是Qwen，由阿里云创建。你是一个有用的助手."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
