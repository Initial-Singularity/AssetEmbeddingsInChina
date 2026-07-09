# 快速上手

在合成数据上、约一分钟内端到端跑完整条预训练-微调训练流水线——无需 CSMAR 数据、无需 GPU——最终得到三份资产嵌入 CSV（W2V、BERT-PT、BERT-FT）。这是论文估计流程的微缩版。

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

notebook 跑的是 **W2V → BERT-PT → BERT-FT** 链路——与论文相同的训练阶段，只是用的是玩具数据：

1. **W2V**——在合成的组合“句子”上训练 Skip-gram Word2Vec。
2. **BERT-PT**（预训练）——在汇总的历史数据上训练 BERT，其嵌入层由 W2V 矩阵热启动（**W2V→BERT**）。
3. **BERT-FT**（微调）——按“季度”精修 BERT，每次拟合都从预训练模型初始化（**PT→FT**）。

每个阶段都把其嵌入写成一份 `(Token, Embed_1..Embed_d)` CSV——这正是流水线的交付物。

## 你应当看到什么

运行结束时会打印三份嵌入 CSV 的汇总表（模型、训练数据、形状、路径），并快速展示它们编码了什么：某只探针股票在三个嵌入空间中的 top-5 余弦近邻。合成 universe 带有潜在的簇结构，因此一只股票的近邻应当留在其所属簇内；两处初始化（W2V→BERT、PT→FT）则让三份嵌入彼此可比。（具体近邻随合成种子而变，但簇结构是稳定的。）

## 下一步

- [方法与实验](../explanation/method.md)——每个阶段如何对应到论文。
- [训练](../how-to/train.md)——各阶段的配置与 CLI。
