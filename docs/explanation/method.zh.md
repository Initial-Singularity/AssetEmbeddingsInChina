# 方法与实验

本页把手稿映射到代码——哪个文件估计哪个方程、哪个配置字段接通哪个阶段——然后罗列每一个实验族（其目的、其配置模板、其运行脚本）。符号沿用手稿的记号。

## 方法（论文与代码）

### 需求系统与资产嵌入

设 $h_{i,a,t}$ 为投资者 $i$ 在季度 $t$ 对资产 $a$ 的持仓金额。循 Gabaix et al. (2025)，一个需求系统把持仓与一个潜在资产向量联系起来：

$$h_{i,a,t} = \boldsymbol{\lambda}_i^{\top}\boldsymbol{x}_{a,t} + \delta_i + \delta_a + \epsilon_{i,a,t},$$

其中 $\boldsymbol{x}_{a,t}$ 是资产 $a$ 在季度 $t$ 的 $d$ 维潜向量——即*资产嵌入（asset embedding）*，整条流水线所估计的唯一对象——$\boldsymbol{\lambda}_i$ 刻画投资者 $i$ 的偏好，$\delta_i$、$\delta_a$ 分别是投资者与资产固定效应。其经济直觉是：投资者会给嵌入相近的资产分配相近的组合权重。

Word2Vec 与 BERT 并不直接使用持仓矩阵；它们利用的是持仓中的序数信息。对每个投资者-季度，按组合权重从大到小把持仓排成序列

$$\pi_{i,t} = \big(a_{i,t}(1),\, a_{i,t}(2),\, \ldots,\, a_{i,t}(L_{i,t})\big),$$

其中 $a_{i,t}(k)$ 是第 $k$ 大持仓对应的资产。有了这一排序，一个组合便可读作一个句子——一列资产词元（token）——而更大的披露持仓透露了投资者在其组合内部如何为各资产排定优先次序。

| 论文 | 代码 |
|---|---|
| 持仓到按排名排序的组合序列 $\pi_{i,t}$ | `scripts/data/sharehold.py`——按投资者/期分组持仓、按持仓比例排序、输出 `Portfolio` 序列（[CLI](../reference/cli.md#data-sharehold)） |
| 为训练序列化组合 | `asset_embeddings.datasets.PortfolioDataEncoder`（[API](../reference/api/datasets.md)） |
| 股票/词元 id 词表 | `Tokenizer_Preparer` 构建 `PreTrainedTokenizerFast`（[Preparers](preparers.md)） |

### 三种模型

流水线用三种模型估计 $\boldsymbol{x}_{a,t}$。三者共享同一目标——把每项资产表示为一个从共同持有模式中恢复出来的 $d$ 维向量——但所利用的结构不同：对无序持仓矩阵做线性分解（推荐系统）、在按权重排序的组合中利用局部窗口共现（Word2Vec）、以及经由 transformer 使用全组合的非线性上下文（BERT）。

#### 推荐系统（RS）

每个季度，对观测到的持仓面板 $\boldsymbol{D}_t$ 做主成分分析（PCA），前 $d$ 个主成分给出嵌入。由于中国的披露规则是混合式的——上市公司只报告其前十大股东，公募基金在 Q1/Q3 也只披露前十大重仓——一项资产没有出现在某投资者的观测组合中并不必然意味着未被持有；该投资者可能只是落在前十之外。因此 $\boldsymbol{D}_t$ 的元素区分已披露与未披露的持仓：

$$d_{i,a,t} = \begin{cases}\phi(h_{i,a,t}) & \text{if } a \text{ is reported in } i\text{'s quarter-}t \text{ portfolio},\\ \psi & \text{otherwise},\end{cases}$$

四种 $(\phi,\psi)$ 变体——RS-Binary、RS-Ranks、RS-Level0、RS-LevelMin——对未披露元素的处理各不相同。对 level 类变体，在做 PCA 之前先从元素中剔除投资者与资产固定效应，使嵌入反映主动的偏离，而非投资者财富或资产规模。

代码：`scripts/train/rs.py`、`AssetRSTrainConfig`（`model_type ∈ {RS_Binary, RS_Ranks, RS_Level0, RS_LevelMin}`）。RS 按季度独立拟合：其嵌入来自特征分解而非迭代优化，因此不存在可供下文预训练-微调范式介入的初始化环节。

#### Word2Vec（W2V）

出现在相似上下文中的词获得相似的含义——“bread”与“butter”常一同出现，因而得到相近的向量。skip-gram 模型把同一思想用于组合，学习嵌入使得在同一批组合中以相近权重被持有的资产在嵌入空间中彼此接近。设上下文窗口半径为 $w$：

$$\max_{\boldsymbol{E}_\text{in},\,\boldsymbol{E}_\text{out}}\ \sum_{(i,t)\in C}\ \sum_{c=1}^{L_{i,t}}\ \sum_{0<|j-c|\le w} \log P\!\big(a_{i,t}(j)\mid a_{i,t}(c)\big).$$

训练得到的输入矩阵 $\boldsymbol{E}_{\text{in}}^{*}$ 给出资产嵌入。论文采用 skip-gram；Gabaix et al. 使用 CBOW（continuous-bag-of-words）变体，下文的 `w2v_model_ablation` 族对二者做了比较。

代码：`scripts/train/w2v.py`（gensim Skip-gram/CBOW + 负采样）、`AssetW2VTrainConfig`。

#### BERT

一个以遮蔽语言建模目标训练的 transformer 编码器：由组合的其余部分预测被遮蔽的资产。Word2Vec 只看每个位置附近的固定窗口，而 BERT 以整个剩余组合为条件。设 $M_{i,t}$ 为被遮蔽的位置、$\tilde{\pi}_{i,t}$ 为遮蔽后的序列：

$$\max_{\boldsymbol{E}_\text{emb},\,\theta_\text{enc},\,\boldsymbol{W}_\text{pred}}\ \sum_{(i,t)\in C}\ \mathbb{E}_{M_{i,t}}\sum_{j\in M_{i,t}} \log P_\theta\!\big(a_{i,t}(j)\mid \tilde{\pi}_{i,t}\big).$$

季度 $t$ 的资产嵌入是行 $[\boldsymbol{E}_{\text{emb}}^{*}]_a$；编码器与预测头在训练后丢弃。

代码：`scripts/train/bert.py`；`asset_embeddings.modules.BertEmbeddings`；随机遮蔽位置的切分是 `asset_embeddings.datasets.TokenMasker`；`AssetBERTTrainConfig`（遮蔽率 `dataset.mask_prob = 0.15`）。

### 预训练-微调范式

在估计 Word2Vec 与 BERT 嵌入时，论文在两处偏离了 Gabaix et al. (2025)。两处偏离回答的是同一个底层问题：当持仓只被部分观测时，如何估计潜在的需求因子？中国的混合披露规则相当于对真实持仓的一道删失（censoring）过滤，而这两处偏离恰恰是为删失最要紧的那些环节设计的。

**第一处偏离：以预训练-微调取代逐季度独立拟合。** Gabaix et al. 让 Word2Vec 与 BERT 每个季度各自训练。一个潜在问题是，由此得到的不同季度的嵌入不可直接比较：在需求系统中，$\boldsymbol{\lambda}_i^{\top}\boldsymbol{x}_{a,t}$ 对旋转 $\Lambda$ 不变——旋转 $\boldsymbol{x}_{a,t}$ 并反向旋转 $\boldsymbol{\lambda}_i$ 得到完全相同的拟合——因此每个季度的解只能识别到一个旋转。当各季度的旋转不同时，差值 $\boldsymbol{x}_{a,t+1} - \boldsymbol{x}_{a,t}$ 未必代表资产的真实变化。预训练-微调范式使各季度的旋转更可能一致：在汇总的历史语料上做一次预训练，产生一个基础嵌入；每个季度的估计都从这一基础初始化，使各季度的解大体上停留在同一坐标系内（不严格地说）。概括起来，预训练学习跨季度共同而稳定的持仓结构，微调则在其上捕捉季度特有的变化。（Gabaix et al. 发现，在披露统一的美国市场上预训练并不改善表现；论文表明在中国并非如此。）

**第二处偏离：用 Word2Vec 热启动 BERT 的嵌入层。** BERT 的嵌入层在预训练与微调两个阶段都被设为训练好的 Word2Vec 嵌入，而非随机初始化（冷启动）。Word2Vec 嵌入已把共同持有轮廓相似的资产放在相近位置，因此 BERT 从一个有意义的几何结构出发，而非从噪声出发。这加快了收敛，并且——在删失式披露下最要紧的一点——让观测稀疏的资产（在 Q1/Q3 只出现在寥寥数个披露组合中的资产）从一开始就获得合理的表示。这遵循迁移学习中用预训练表示初始化更复杂模型的标准做法。

两处偏离缺任何一处，BERT 嵌入的表现通常都会退化——下文的 `w2v_init_ablation` 与 `no_pt_init` 族分别隔离了二者的作用。

这里的 BERT 有两个可训练部分需要初始化——一个**嵌入层**（每只股票一个向量）和一个 transformer **编码器**——而这两处偏离恰恰是 BERT 配置（`templates/train_main.json`）中两个字段所设定的：

| 阶段 | `model.model_checkpoint`（为编码器播种） | `model.w2v_model` = `tokenizer.w2v_model`（为嵌入层播种） |
|---|---|---|
| **BERT 预训练** | `null`——编码器从零训练 | `checkpoints/pretrained/AssetW2V/d{dim}/AssetW2V_d{dim}_pretrained.model` |
| **BERT 微调，季度 $t$** | `checkpoints/pretrained/AssetBERT/d{dim}/AssetBERT_d{dim}_pretrained_best/model.safetensors` | `checkpoints/finetune/AssetW2V/d{dim}/{t}/AssetW2V_d{dim}_{t}.model` |

- **`model.model_checkpoint`** 为编码器播种。预训练时为 `null`，故编码器在汇总语料上从零训练。微调时为预训练好的 BERT，故编码器被沿用、仅被轻推——这就是第一处偏离的 **PT→FT** 初始化链路。（遮蔽语言模型的预测头每次都重新初始化；它在训练后本就被丢弃。）
- **`model.w2v_model`** 为嵌入层播种：`AssetBERT_Preparer` 把一个训练好的 Word2Vec 模型的向量拷入 BERT 的嵌入矩阵（见 [Preparers](preparers.md)）——这就是第二处偏离的 **W2V→BERT** 热启动。预训练时它是在汇总语料上训练的 W2V。微调时，播种发生在检查点加载*之后*，故季度 $t$ 自己的 W2V 覆盖嵌入层，而编码器保持从预训练 BERT 加载的状态。
- **`model.embedding_file`** 做同样的嵌入播种，只是从一份嵌入 CSV 而非 Word2Vec 模型读取；在主流水线中为 `null`。

于是每个微调后的季度都是*共享的预训练编码器加上该季度自己的嵌入层*，再经一次针对该季度持仓的短暂微调加以调整。完整的逐阶段配置见 [训练](../how-to/train.md)。配套论文研究由此得到的逐季度嵌入对公司估值与收益相关性的解释力。

### Text-embedding 对照

商用 LLM 基线——嵌入公司*名称*而非持仓——是 `asset_embeddings.scripts.data.text_embedding` 流水线（OpenAI / Cohere / Voyage / Gemini / Qwen / BGE）。见 [Text embeddings](../how-to/text-embeddings.md)。

## 实验

每个实验都是一个**族（family）**：一份 [`fast_generator`](../how-to/generate-configs.md) 模板（展开配置网格），加上一个 `scripts/run/*.sh`（通过 `try_run` + 标记机制运行它，`-k`/`-e` 选取子集）。每个族都产出自己的一套逐季度嵌入序列。

全程维度 $d \in \{4,8,10,16,32,64,128\}$；`main` 的季度为 2023Q2–2024Q3，time-split 各族为 2020Q4–2024Q3。

### 主流水线

| 族 | 它做什么 | 模板 | 脚本 |
|---|---|---|---|
| **main** | 跨架构 × 维度 × 季度，训练 W2V，再 BERT-PT，再 BERT-FT（及 RS）；头条嵌入序列。 | `train_main.json` | `train_main.sh` |

### 稳健性（替代的训练设计）

| 族 | 它做什么 | 模板 | 脚本 |
|---|---|---|---|
| **timesplit_robust** | 50/50 预训练/微调切分（相对头条流水线更均衡的替代），扩展到 16 个季度。 | `train_timesplit_robust.json` | `train_timesplit_robust.sh` |
| **semiannual_robust** | 仅在 Q2/Q4 上训练。 | `train_semiannual_robust.json` | `train_semiannual_robust.sh` |
| **semiannual_timesplit** | 50/50 time-split 语料的 Q2/Q4 子集。 | `train_semiannual_timesplit.json` | `train_semiannual_timesplit.sh` |
| **expansion** | 累积预训练：对每个季度 $t$，在 $t$ 之前的全部数据上从零重训，再在 $t$ 上微调。 | `train_expansion.json` | `train_expansion.sh` |
| **sliding_window** N{8,20,40} | 仅在最近 N 个季度（2 / 5 / 10 年）上预训练。 | `train_sliding_window_N{8,20,40}.json` | `train_sliding_window.sh` |

### 消融（是什么驱动了结果）

| 族 | 它做什么 | 模板 | 脚本 |
|---|---|---|---|
| **w2v_init_ablation** | 关于 W2V 初始化的 2×2 析因（PT w2v-init × FT w2v-reinit）；隔离 **W2V→BERT** 热启动。 | `train_w2v_init_ablation.json` | `train_w2v_init_ablation.sh` |
| **no_pt_init** | 每个季度从随机初始化微调 BERT（无预训练 BERT 检查点）；隔离 **PT→FT** 初始化。 | `train_no_pt_init.json` | `train_no_pt_init.sh` |
| **w2v_model_ablation**（+ `_timesplit`） | CBOW vs. Skip-gram W2V，其余冻结——W2V 算法是否通过 BERT 链路产生影响？ | `train_w2v_model_ablation.json`（+ `_timesplit`） | `train_w2v_model_ablation.sh`（+ `_timesplit`） |

### Text-embedding 基线

| 族 | 它做什么 | 模板 | 脚本 |
|---|---|---|---|
| **text_embedding** | 公司名称的 LLM 嵌入（OpenAI / Cohere / Voyage / Gemini / Qwen / BGE）。 | `data_text_embedding.json` | `data_text_embedding.sh` |
