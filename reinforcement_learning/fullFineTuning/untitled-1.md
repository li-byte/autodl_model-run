# 全量微调（Full Fine-Tuning）项目文档

## 📋 项目简介

本项目展示如何对大语言模型进行**全量微调**，通过自定义数据集让模型学习特定知识。

**主要特点**：
- 使用 Qwen3-4B 预训练模型
- 完整参数微调（Full Fine-Tuning）
- 多种显存优化技术
- 包含训练和推理两个完整脚本

---

## 📁 目录结构

```
fullFineTuning/
├── full_fine_tuning.py      # 训练脚本
├── full_fine_tuning_model.py # 推理脚本
└── untitled-1.md            # 本文档
```

---

## 📄 full_fine_tuning.py - 训练脚本

### 1. 环境配置

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

配置 HuggingFace 镜像、SwanLab 密钥和 PyTorch 显存优化：

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["SWANLAB_API_KEY"] = "JIfWrqblrMOK4g5iXPfJj"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 减少显存碎片
```

### 2. 配置参数

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

```python
MAX_SEQ_LENGTH = 2048
SEED = 3407
```

### 3. 系统提示和标记

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

定义推理过程和答案的标记符：

```python
REASONING_START = "<开始推理>"
REASONING_END = "<结束推理>"
SOLUTION_START = "<答案>"
SOLUTION_END = "<答案结束>"
```

### 4. create_local_dataset - 创建数据集

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

**作用**：创建本地训练数据集，包含 4 个问答样本。

**流程**：
1. 定义 4 个问答样本（问题、推理过程、答案）
2. 使用聊天模板格式化数据
3. 对数据进行 tokenization

### 5. init_model - 初始化模型

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

**作用**：加载预训练模型（Qwen3-4B），优化显存使用。

**关键优化**：

| 参数 | 作用 |
|------|------|
| `torch.bfloat16` | 使用 bfloat16 精度减少显存 |
| `device_map="auto"` | 自动设备映射 |
| `use_cache=False` | 禁用 KV 缓存 |
| `gradient_checkpointing_enable` | 启用梯度检查点 |

### 6. train_full_finetune - 训练函数

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

**作用**：使用 Transformers 的 Trainer 进行全量微调。

**训练配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `per_device_train_batch_size` | 1 | 单批次大小为 1 |
| `gradient_accumulation_steps` | 8 | 梯度累积 8 步（等效批次大小 8）|
| `num_train_epochs` | 100 | 训练 100 轮 |
| `optim` | `"adamw_8bit"` | 使用 8-bit AdamW 优化器 |
| `bf16` | `True` | 启用 bfloat16 混合精度训练 |
| `gradient_checkpointing` | `True` | 梯度检查点 |

### 7. test_inference - 推理测试

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

**作用**：测试训练后模型的推理能力。

### 8. main - 主函数

**文件**：[`full_fine_tuning.py`](full_fine_tuning.py)

**流程**：
1. 初始化模型
2. 创建数据集
3. 开始训练
4. 保存模型到 `./full_finetuned_model`
5. 测试推理

---

## 📄 full_fine_tuning_model.py - 推理脚本

### 1. load_model - 加载模型

**文件**：[`full_fine_tuning_model.py`](full_fine_tuning_model.py)

**作用**：加载训练好的全量微调模型。

### 2. test_inference - 推理测试

**文件**：[`full_fine_tuning_model.py`](full_fine_tuning_model.py)

**作用**：使用加载的模型进行推理测试。

### 3. main - 主函数

**文件**：[`full_fine_tuning_model.py`](full_fine_tuning_model.py)

**作用**：执行模型加载和推理测试的主流程。

---

## 🎯 训练方式详解

### 全量微调（Full Fine-Tuning）

**定义**：更新模型的**所有参数**，而不仅仅是部分参数。

**使用场景**：
1. 数据集小但任务特定 - 模型需要完全学习新信息
2. 希望模型深度融入新知识 - 让新知识完全融入模型权重
3. 推理简单 - 不需要额外的 LoRA 适配器

### 显存优化策略

| 优化技术 | 说明 |
|---------|------|
| `bfloat16` | 半精度浮点数，比 float32 节省一半显存 |
| `gradient_checkpointing` | 前向传播时不保存中间激活，反向传播时重新计算 |
| `gradient_accumulation_steps=8` | 小批次累积梯度，等效大批次 |
| `adamw_8bit` | 8 位优化器，减少优化器状态显存 |
| `use_cache=False` | 禁用 KV 缓存 |

### 损失函数

使用 `DataCollatorForLanguageModeling(mlm=False)`，即**因果语言建模（CLM）**：
- 模型学习预测下一个 token
- 标准的自回归语言模型训练方式

---

## 🔍 加载方式说明

### 为什么使用 `AutoModelForCausalLM` 直接加载？

| 微调方式 | 训练时保存 | 测试时加载 |
|---------|-----------|-----------|
| **全量微调** | 完整模型权重 | `AutoModelForCausalLM.from_pretrained()` |
| **LoRA 微调** | 基础模型 + LoRA 适配器 | 基础模型 + `PeftModel.from_pretrained()` |

**原因**：
1. 参数已完全更新 - 保存的是完整的模型文件
2. 无需额外适配器 - 不依赖 PEFT/LoRA 库
3. 推理效率更高 - 无需适配器加载步骤

### 加载代码示例

**文件**：[`full_fine_tuning_model.py`](full_fine_tuning_model.py)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

---

## 📊 数据流图

```
预训练模型 (Qwen3-4B)
       ↓
  [初始化模型]
  加载模型，设置梯度检查点
       ↓
   [创建数据集]
  4个问答样本 → 格式化 → tokenization
       ↓
     [训练模型]
  Trainer + 全量微调 + 显存优化
       ↓
     [保存模型]
  ./full_finetuned_model
       ↓
     [推理测试]
  AutoModelForCausalLM 直接加载推理
```

---

## 🚀 使用指南

### 1. 安装依赖

```bash
pip install torch transformers datasets pandas
```

### 2. 训练模型

```bash
python full_fine_tuning.py
```

训练输出：
- 模型保存位置：`./full_finetuned_model`
- 自动执行推理测试

### 3. 测试模型

```bash
python full_fine_tuning_model.py
```

---

## 💡 全量微调 vs LoRA 微调对比

| 特性 | 全量微调 | LoRA 微调 |
|------|---------|-----------|
| **更新参数** | 所有参数 | 仅适配器参数 |
| **显存需求** | 高 | 低 |
| **保存大小** | 完整模型（GB 级） | 适配器（MB 级） |
| **推理加载** | 直接加载 | 基础模型 + 适配器 |
| **适合场景** | 深度融合、数据量大 | 资源受限、快速适配 |

---

## ⚙️ 参数调优建议

| 参数 | 默认值 | 调优建议 |
|------|--------|---------|
| `per_device_train_batch_size` | 1 | 显存充足时可增大 |
| `gradient_accumulation_steps` | 8 | 配合 batch_size 调整 |
| `num_train_epochs` | 100 | 小数据集可设大值 |
| `learning_rate` | 默认 | 全量微调建议使用较小学习率 |

---

## 📝 总结

通过本项目可以学习到：

- ✅ 如何准备和格式化训练数据
- ✅ 如何配置 Trainer 进行全量微调
- ✅ 如何优化显存使用（bfloat16、梯度检查点等）
- ✅ 全量微调后如何加载和推理
- ✅ 全量微调与 LoRA 的区别和适用场景