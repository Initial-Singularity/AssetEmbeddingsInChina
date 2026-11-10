# 快速上手

在合成数据上、约一分钟内端到端跑完整条预训练-微调流水线（无需 CSMAR 数据、无需 GPU），并亲眼看到微调后的 BERT 嵌入拉开与预训练基线的差距。这是本文核心机制的微缩版。

## 准备环境

```bash
git clone https://github.com/Initial-Singularity/AssetEmbeddingsInChina
cd AssetEmbeddingsInChina
uv sync --extra notebook
```

## 运行

```bash
uv run jupyter lab examples/quickstart.ipynb
# 或无界面运行：
uv run python examples/quickstart.py
```

该 notebook 使用 `examples/data/` 中随附的合成数据集——每个数值都是生成的，因此无需任何外部数据。

## 它做了什么

notebook 跑的是 W2V、BERT-PT、BERT-FT、RV 这条链路——与论文相同的四个阶段，只是用的是玩具数据：

1. **W2V**——在合成的组合“句子”上训练 Skip-gram Word2Vec。
2. **BERT-PT**（预训练）——在汇总的历史数据上训练 BERT，其嵌入层由 W2V 矩阵初始化（即**热启动**）。
3. **BERT-FT**（微调）——在微调期数据（单个季度 2024Q2）上精修 BERT，从预训练模型初始化（即**预训练-微调**这一步）。
4. **RV**——在相对估值基准上为三个嵌入（W2V、BERT-PT、BERT-FT）打分。

## 你应当看到什么

在同一基准上，BERT-FT 相对 PT 训练的基线明显领先约 **+0.07 $R^2$**：逐季度微调捕捉到了静态基线遗漏的时变结构。（具体数值随合成种子而变，但排序是稳定的。）

## 下一步

- [方法与实验](../explanation/method.md)——每个阶段如何对应到论文，以及完整的实验目录。
- [训练](../how-to/train.md) / [评估](../how-to/evaluate.md)——各阶段的配置与 CLI。
