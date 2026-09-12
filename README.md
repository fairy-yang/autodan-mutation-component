# AutoDAN-Turbo Lightweight Mutation Module

## 功能

基于 AutoDAN-Turbo 原始 Attacker.use_strategy() 文本变异方法，
使用 Qwen/Qwen2.5-1.5B-Instruct 作为轻量化生成模型，
用于安全测评平台的文本测试样例自动生成与变异。

## 流程

平台输入
→ AutoDAN Strategy Library
→ AutoDAN Attacker
→ Qwen Chat Template
→ Qwen2.5-1.5B-Instruct
→ 结构化变异结果

## 平台入口

核心接口：

run_mutation(items)

支持单条 dict 和批量 list 输入。

## 输入字段

- id
- text
- risk_category
- source
- parent_id
- version

其中 text 为必填字段。

## 输出字段

- id
- original_text
- mutated_text
- strategy
- risk_category
- source
- parent_id
- version
- method
- model

## 安装

pip install -r requirements.txt

## 模型

默认模型：

Qwen/Qwen2.5-1.5B-Instruct

模型权重不随代码仓库提交。

## 当前完成情况

- AutoDAN-Turbo 原始 Attacker 方法复用：完成
- Qwen2.5-1.5B-Instruct 轻量化适配：完成
- 单条文本生成：完成
- 批量文本生成：完成
- 结构化输入输出：完成
- Python 平台调用接口：完成

本模块不重新执行 AutoDAN-Turbo 的 warm-up、
lifelong strategy learning 和完整论文实验流程，
而是加载已有 Strategy Library 完成在线文本变异。
