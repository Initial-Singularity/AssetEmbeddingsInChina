# 方法与实验

本页把手稿映射到代码——哪个文件估计哪个方程、哪个配置字段接通哪个阶段——然后罗列每一个实验族（其目的、其配置模板、其运行脚本）。符号沿用手稿的记号。

## 方法（论文与代码）

### 组合即句子

每个投资者在季度 $t$ 的组合，是一段按持仓排序的序列——一个*句子*——而每只股票是一个*词元（token）*。持仓排序（最大持仓在前）携带经济含义，正如语言中的词序。正是这一点让资产定价问题得以用 NLP 架构来攻克。

| 论文 | 代码 |
|---|---|
| 持仓 → 按排序的组合序列 | `scripts/data/sharehold.py`——按投资者/期分组持仓、按持仓比例排序、输出 `Portfolio` 序列（[CLI](../reference/cli.md#data-sharehold)） |
| 为训练序列化组合 | `asset_embeddings.datasets.PortfolioDataEncoder`（[API](../reference/api/datasets.md)） |
| 股票/词元 id 词表 | `Tokenizer_Preparer` 构建 `PreTrainedTokenizerFast`（[Preparers](preparers.md)） |

### 三种架构

仓库用三种架构来估计嵌入序列 $\{\boldsymbol{x}_{a,t}\}$。

#### 推荐系统（RS）

对去均值持仓矩阵 $\tilde{\boldsymbol{H}}_t$ 做截断 SVD：

$$\tilde{\boldsymbol{H}}_t = \boldsymbol{U}_t\,\boldsymbol{\Sigma}_t\,\boldsymbol{V}_t^{\top},
\qquad \boldsymbol{X}_t^{\text{RS}} = \boldsymbol{V}_{t,d}\,\boldsymbol{\Sigma}_{t,d}.$$

四种 $(\phi,\psi)$ 编码（RS-Binary / RS-Ranks / RS-Level0 / RS-LevelMin）即 `model_type` 的可选项。

代码：`scripts/train/rs.py`、`AssetRSTrainConfig`（`model_type ∈ {RS_Binary, RS_Ranks, RS_Level0, RS_LevelMin}`）。RS 按季度独立拟合，处于下文耦合之外。

#### Word2Vec（W2V）

在按排序的组合序列上做 Skip-gram：

$$\mathcal{L}_{\text{W2V}} = -\sum_{(i,t)\in D}\ \sum_{c=1}^{L_{i,t}}\ \sum_{0<|j-c|\le w}
\log \mathbb{P}\!\big(a_{i,t}(j)\mid a_{i,t}(c)\big).$$

训练得到的输入矩阵 $\boldsymbol{E}_{\text{in}}^{*}$ 给出资产嵌入。

代码：`scripts/train/w2v.py`（gensim Skip-gram/CBOW + 负采样）、`AssetW2VTrainConfig`。

#### BERT

一个 Transformer 编码器，用遮蔽语言建模目标、以遮蔽率 $p_{\text{mask}}=0.15$ 训练：

$$\mathcal{L}_{\text{MLM}} = -\sum_{(i,t)\in D}\ \mathbb{E}_{M_{i,t}}\sum_{j\in M_{i,t}}
\log \mathbb{P}_\theta\!\big(a_{i,t}(j)\mid \tilde{\pi}_{i,t}\big).$$

季度 $t$ 的资产嵌入是行 $[\boldsymbol{E}_{\text{emb}}^{*}]_a$；编码器与头在训练后丢弃。

代码：`scripts/train/bert.py`；`asset_embeddings.modules.BertEmbeddings`；随机遮蔽位置的切分是 `asset_embeddings.datasets.TokenMasker`；`AssetBERTTrainConfig`。

### 预训练-微调耦合

目标是一个*时变*嵌入：每只股票每季度一个向量，使得从一个季度到下一个季度的变化本身就富含信息。这会撞上一个可识别性问题。神经模型学到的嵌入只在隐空间坐标轴的任意旋转、反射与重标号意义下被确定——在同一份数据上从不同随机起点训练两次，你会得到两个同样有效、却看上去截然不同的嵌入。于是，若每个季度独立训练，季度间的差 $\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$（股票 $a$ 在 $t{+}1$ 的向量减去其在 $t$ 的向量）多半只在度量那种任意的坐标漂移，而非股票的任何真实变化。

补救之道，是给每个季度**相同的起点**。模型在汇总的早期季度语料上*预训练*一次，产生单一的基解；随后每个后续季度都从这个共享基解*微调*，而非从零开始。因为所有季度从同一锚点出发、且只移动一点点，它们的嵌入便停留在同一坐标系中，$\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$ 也就转而反映股票的真实移动。

这里的 BERT 有两个可训练部分需要初始化——一个**嵌入层**（每只股票一个向量）和一个 Transformer **编码器**——而“从共享基解出发”恰恰是 BERT 配置（`templates/train_main.json`）中两个字段所设定的：

| 阶段 | `model.model_checkpoint`（为编码器播种） | `model.w2v_model` = `tokenizer.w2v_model`（为嵌入层播种） |
|---|---|---|
| **BERT 预训练** | `null`——编码器从零训练 | `checkpoints/pretrained/AssetW2V/d{dim}/AssetW2V_d{dim}_pretrained.model` |
| **BERT 微调，季度 $t$** | `checkpoints/pretrained/AssetBERT/d{dim}/AssetBERT_d{dim}_pretrained_best/model.safetensors` | `checkpoints/finetune/AssetW2V/d{dim}/{t}/AssetW2V_d{dim}_{t}.model` |

- **`model.model_checkpoint`** 为编码器播种。预训练时为 `null`，故编码器在汇总语料上从零训练。微调时为预训练好的 BERT，故编码器被沿用、仅被轻推——这就是 **PT-to-FT** 链路，是把每个季度的解系于同一共享坐标系的那部分。（遮蔽语言模型的预测头每次都重新初始化；它在训练后本就被丢弃。）
- **`model.w2v_model`** 为嵌入层播种：`AssetBERT_Preparer` 把一个训练好的 Word2Vec 模型的向量拷入 BERT 的嵌入矩阵（见 [Preparers](preparers.md)）——即 **W2V-to-BERT** 链路。预训练时它是在汇总语料上训练的 W2V。微调时，播种发生在检查点加载*之后*，故季度 $t$ 自己的 W2V 覆盖嵌入层，而编码器保持从预训练 BERT 加载的状态。
- **`model.embedding_file`** 做同样的嵌入播种，只是从一份嵌入 CSV 而非 Word2Vec 模型读取；在主流水线中为 `null`。

于是每个微调后的季度都是*共享的预训练编码器加上该季度自己的嵌入层*，再经一次针对该季度持仓的短暂微调加以调整。正是这个共享编码器，把逐季度的嵌入跨时间地固定在同一可比坐标系中。完整的逐阶段配置见 [训练](../how-to/train.md)。配套论文研究由此得到的逐季度嵌入对公司估值与收益相关性的解释力。

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
| **w2v_init_ablation** | 关于 W2V 初始化的 2×2 析因（PT w2v-init × FT w2v-reinit）；隔离 **W2V-to-BERT** 耦合。 | `train_w2v_init_ablation.json` | `train_w2v_init_ablation.sh` |
| **no_pt_init** | 每个季度从随机初始化微调 BERT（无预训练 BERT 检查点）；隔离 **PT-to-FT** 耦合。 | `train_no_pt_init.json` | `train_no_pt_init.sh` |
| **w2v_model_ablation**（+ `_timesplit`） | CBOW vs. Skip-gram W2V，其余冻结——W2V 算法是否通过 BERT 链路产生影响？ | `train_w2v_model_ablation.json`（+ `_timesplit`） | `train_w2v_model_ablation.sh`（+ `_timesplit`） |

### Text-embedding 基线

| 族 | 它做什么 | 模板 | 脚本 |
|---|---|---|---|
| **text_embedding** | 公司名称的 LLM 嵌入（OpenAI / Cohere / Voyage / Gemini / Qwen / BGE）。 | `data_text_embedding.json` | `data_text_embedding.sh` |
