import json
import random
if __name__ == '__main__':

    # 原始单条样本
    sample = {
        "instruction": "根据图片和描述回答问题",
        "input": {
            "text": "描述这个图片",
            "image": "./data/images/dog_running.jpg"
        },
        "output": "这是一幅充满奇幻色彩的卡通风格插画，画面背景为柔和的灰绿色，整体氛围既复古又带点科幻感。画面中央是一台造型奇特的机器，机器顶部有一个透明罩，里面悬浮着一个大脑，大脑上方还有类似电波的金色线条。机器的正面有一块小屏幕，屏幕上显示着“065 五 -”的字样，屏幕下方是一个出纸口，正吐出一张带有文字或符号的纸张，机器前方的地面上还散落着两张空白的纸。机器的两侧各连接着一个大型的圆形磁盘（类似唱片或卷轴），左侧的磁盘由一只紫色的卡通小动物操作，它手里还拿着一个带有图案的小圆盘，尾巴细长地垂在地上，眼睛睁得大大的，显得很专注。右侧的磁盘旁边站着一只粉色的卡通小动物，它站在一个小木凳上，眼睛同样睁得很大，似乎在观察机器的运行。在机器的正前方，还有一只蓝色的卡通小动物，它的眼睛也很大，嘴巴微张，似乎对机器吐出的纸张很感兴趣，身体朝向机器，姿态显得有些好奇和期待。整体画面中的小动物们都有着圆圆的身体、大大的眼睛，形象可爱，机器的设计则充满了想象力，结合大脑、电波、磁盘等元素，营造出一种充满童趣的科幻实验场景，仿佛这些小动物正在进行某种神秘的研究或发明。"
    }

    # 数据增强模板
    instruction_variants = [
        "请根据图片和描述回答问题",
        "请详细描述图片内容",
        "对图片进行生动描写",
        "描述图片中的场景和元素",
        "用丰富细节描写图片内容"
    ]

    output_variants = [
        sample["output"],
        sample["output"].replace("充满奇幻色彩的卡通风格", "色彩鲜艳的卡通风格"),
        sample["output"].replace("柔和的灰绿色", "明亮的绿色"),
        sample["output"].replace("复古又带点科幻感", "现代感十足且富有科幻感"),
        sample["output"].replace("小动物们都有着圆圆的身体、大大的眼睛", "小动物们形态各异，表情丰富")
    ]

    # 生成增强后的样本列表
    augmented_samples = []

    for i in range(20):  # 生成 20 个变体
        inst = random.choice(instruction_variants)
        out = random.choice(output_variants)
        augmented_samples.append({
            "instruction": inst,
            "input": sample["input"],
            "output": out
        })

    # 保存为 train_augmented.jsonl
    with open("train_augmented.jsonl", "w", encoding="utf-8") as f:
        for item in augmented_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("增强数据生成完成，总样本数:", len(augmented_samples))