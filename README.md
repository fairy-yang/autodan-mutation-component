# AutoDAN-Turbo Lightweight Mutation Module

## 功能

基于 AutoDAN-Turbo 原始 `Attacker.use_strategy()` 文本变异接口，使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为轻量化生成模型，用于安全测评平台的文本测试样例自动生成与变异。

本仓库是轻量化组件，不是论文完整复现。当前实现加载已有 Strategy Library，按调度选择策略生成候选文本；不包含 Target、Scorer、Summarizer、Retrieval 组成的持续学习闭环。

## 流程

平台输入
→ AutoDAN Strategy Library
→ AutoDAN Attacker
→ Qwen Chat Template
→ Qwen2.5-1.5B-Instruct
→ 结构化变异结果

## 平台入口

核心接口：

```python
from autodan_mutation_service import run_mutation

results = run_mutation(items, model_config=model_config)
```

支持单条 `dict` 和批量 `list[dict]` 输入。

## 输入字段

- `id`
- `text`，必填
- `risk_category`
- `source`
- `parent_id`
- `version`

## 输出字段

- `id`
- `original_text`
- `mutated_text`
- `strategy`
- `risk_category`
- `source`
- `parent_id`
- `version`
- `method`
- `model`
- `generation`

`generation` 会返回本次生成的长度预算信息，例如输入 token 数、实际使用 token 数、上下文窗口、是否发生截断。

## 长文本配置

旧版本在服务层写死 `max_length=10000`，长文本库输入时不容易判断真实上下文预算。现在改为按模型上下文窗口和 `max_new_tokens` 计算输入预算，并在超限时可配置处理方式。

示例：

```python
model_config = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "temperature": 0.8,
    "top_p": 0.9,
    "max_new_tokens": 512,
    "max_context_tokens": 8192,
    "allow_input_truncation": True,
    "truncation_strategy": "middle",
}
```

`truncation_strategy` 支持：

- `middle`：保留头尾，适合长文档类输入。
- `head`：保留开头。
- `tail`：保留结尾。

如果希望超限时直接报错，设置：

```python
"allow_input_truncation": False
```

## AutoDL 启动

如果上一次 AutoDL 实例和环境还保留，先不要重装。进入终端后执行：

```bash
conda activate autodanturbo
cd /root/autodl-tmp/AutoDAN-Turbo
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
git status
```

拉取本分支测试：

```bash
git fetch origin codex/long-input-budget
git checkout codex/long-input-budget
python example_call.py
```

如需用你自己的批量数据，保持 `items` 为 `list[dict]`，每条至少包含 `text` 字段。

## 安装

```bash
pip install -r requirements.txt
```

## 模型

默认模型：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

模型权重不随代码仓库提交。

## 当前完成情况

- AutoDAN-Turbo 原始 Attacker 方法复用：完成
- Qwen2.5-1.5B-Instruct 轻量化适配：完成
- 单条文本生成：完成
- 批量文本生成：完成
- 结构化输入输出：完成
- 长文本上下文预算与截断信息返回：完成
- Python 平台调用接口：完成

本模块不重新执行 AutoDAN-Turbo 的 warm-up、lifelong strategy learning 和完整论文实验流程，而是加载已有 Strategy Library 完成在线文本变异。
