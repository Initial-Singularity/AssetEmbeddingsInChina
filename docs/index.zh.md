# AssetEmbeddingsInChina

> 从中国 A 股市场的部分机构持仓中学习资产嵌入（asset embeddings）。
> 配套工作论文 **"Do Asset Embeddings Need Complete Holdings? Lessons from China"**（2026）的训练代码。

沿用 [Gabaix et al. (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4507511) 的思路，我们把每个投资者的投资组合视为一个*句子*，把每只股票视为一个*词元（token）*：持仓的排序携带经济含义（最大的仓位即首要敞口），正如语言中词序携带语义一样。由此即可用 NLP 技术学习资产表示，用于资产定价与投资者建模。

<p align="center"><img src="figures/portfolio_sentence.svg" alt="Portfolio-sentence isomorphism" width="640"></p>

## 论文的两处不同

在估计上，我们与 Gabaix et al. 有两处不同，而两者回答的是同一个问题：当持仓只能被*部分观测*时，如何估计潜在的需求因子？中国的混合披露规则相当于对真实持仓加了一层删失（censoring）过滤。

其一，**用预训练-微调替代逐季度独立拟合**：逐季度独立拟合出的嵌入只能识别到一个旋转变换，因而在时间上不可直接比较。先在前 80% 的持仓观测上做一次预训练（按时间顺序切分，截点为 2023Q1：前 73 个季度用于预训练，其后 6 个季度用于逐季度微调），再令每个季度的估计都从这个基础出发，就使各季度的旋转更可能一致——预训练学到跨季度共同、稳定的持仓结构，微调捕捉各季度特有的变化。其二，**用训练好的 Word2Vec 嵌入热启动（warm-start）BERT 的嵌入层**而非随机初始化，这恰好为删失披露下观测稀疏的资产提供了一个合理的先验。参见 [方法：论文 ↔ 代码](explanation/method.md)。

<p align="center"><img src="figures/pipeline.svg" alt="Pretrain-finetune pipeline" width="720"></p>

## 本仓库交付什么

一套覆盖完整嵌入流水线的训练框架：三个模型（推荐系统、Word2Vec、BERT）、预训练-微调估计流程，以及配套的数据工具。产出是一个逐季度的 embedding CSV 序列——每只股票每季度一个向量，且在时间上可比。配套论文研究这些嵌入对公司估值与收益相关性的解释力。

## 快速上手

```bash
git clone https://github.com/Initial-Singularity/AssetEmbeddingsInChina
cd AssetEmbeddingsInChina
uv sync --extra notebook                      # notebook 与 ipykernel 仅为演示用的可选依赖
uv run jupyter lab examples/quickstart.ipynb  # 或：uv run python examples/quickstart.py
```

该 notebook 在 `examples/data/` 中随附的合成数据集上跑完整的 **W2V-PT → W2V-FT → BERT-PT → BERT-FT** 链路（CPU，约 1 分钟），最终产出四份 embedding CSV。真实的生产运行需要 CSMAR 数据——参见 [准备数据](how-to/prepare-data.md)。完整走查：[快速上手教程](tutorials/quickstart.md)。

## 该往哪儿看

- **初次到访？** [快速上手](tutorials/quickstart.md)
- **找操作配方？** [安装](how-to/install.md) · [准备数据](how-to/prepare-data.md) · [训练](how-to/train.md)
- **想知道“为什么”？** [方法与实验](explanation/method.md)
- **只想要 embedding？** 从 Zenodo 下载：[10.5281/zenodo.20640781](https://doi.org/10.5281/zenodo.20640781)（CC-BY-4.0）
- **查点东西？** [CLI 参考](reference/cli.md) · [API 参考](reference/api/index.md)

## 引用

```bibtex
@unpublished{cheng2026asset,
  title  = {Do Asset Embeddings Need Complete Holdings? Lessons from China},
  author = {Cheng, Zitong and Huang, Dashan and Liu, Xiaobin and Zeng, Tao},
  year   = {2026},
  note   = {Working paper},
}
```

训练好的 embedding 已在 Zenodo 存档（CC-BY-4.0）：
[10.5281/zenodo.20640781](https://doi.org/10.5281/zenodo.20640781)。
