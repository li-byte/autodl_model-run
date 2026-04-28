# train_grpo.py
import re
from trl import GRPOTrainer, GRPOConfig
from vllm import SamplingParams
from config import SEED
from config import REASONING_END, SOLUTION_START, SOLUTION_END
# -----------------------------
# GRPO奖励函数
# -----------------------------
def match_format_approximately(completions, **kwargs):
    """
    奖励函数：根据是否包含推理与答案标记，给出分数
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    """
    奖励函数：根据模型答案是否正确评分
    """
    match_format = re.compile(
        rf"{REASONING_END}.*?{SOLUTION_START}(.+?){SOLUTION_END}.*$", flags=re.MULTILINE | re.DOTALL
    )
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    for r, true_answer in zip(responses, answer):
        guess = match_format.search(r)
        if guess is None:
            scores.append(-2.0)
        else:
            scores.append(5.0 if guess.group(1).strip() == true_answer.strip() else -1.5)
    return scores

# -----------------------------
# GRPO训练
# -----------------------------
def train_grpo(model, tokenizer, dataset, max_prompt_length, max_completion_length, swanlab_callback=None):
    """
    使用GRPO进行强化学习训练
    """
    sampling_params = SamplingParams(
        min_p=0.1, top_p=1.0, top_k=-1, seed=SEED,
        stop=[tokenizer.eos_token], include_stop_str_in_output=True
    )
    training_args = GRPOConfig(
        vllm_sampling_params=sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=10,
        save_steps=100,
        report_to="swanlab" if swanlab_callback else None,
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[match_format_approximately, check_answer],
        args=training_args,
        train_dataset=dataset,
        callbacks=[swanlab_callback] if swanlab_callback else None
    )
    trainer.train()