# 方法与实验

本页把手稿映射到代码——哪个文件估计哪个方程、哪个配置字段接通哪个阶段——然后按论文顺序逐一介绍各项实验，以及产出它们的实验族（配置模板与运行脚本）。符号沿用手稿的记号。

## 方法（论文与代码）

### 组合即句子

每个投资者在季度 $t$ 的组合，是一段按持仓排序的序列——一个*句子*——而每只股票是一个*词元（token）*。持仓排序（最大持仓在前）携带经济含义，正如语言中的词序。正是这一点让资产定价问题得以用 NLP 架构来攻克。

| 论文 | 代码 |
|---|---|
| 由持仓生成按排序的组合序列 | 在本仓库之外准备：按投资者/期分组持仓、按持仓比例排序，写成组合 CSV 格式的 `Portfolio` 序列 |
| 为训练序列化组合 | `asset_embeddings.datasets.portfolio_csv`（[API](../reference/api/datasets.md)） |
| 股票/词元 id 词表 | `Tokenizer_Preparer` 构建 `PreTrainedTokenizerFast`（[Preparers](preparers.md)） |

### 三种架构

仓库用三种架构来估计嵌入序列 $\{\boldsymbol{x}_{a,t}\}$。

#### 推荐系统（RS）

对去均值持仓矩阵 $\tilde{\boldsymbol{H}}_t$ 做截断 SVD：

$$\tilde{\boldsymbol{H}}_t = \boldsymbol{U}_t\,\boldsymbol{\Sigma}_t\,\boldsymbol{V}_t^{\top},
\qquad \boldsymbol{X}_t^{\text{RS}} = \boldsymbol{V}_{t,d}\,\boldsymbol{\Sigma}_{t,d}.$$

四种 $(\phi,\psi)$ 编码（RS-Binary / RS-Ranks / RS-Level0 / RS-LevelMin）即 `model_type` 的可选项。

代码：`asset_embeddings/trainers/rs.py`（CLI `asset_embeddings/scripts/train/rs.py`）、`AssetRSTrainConfig`（`model_type ∈ {RS_Binary, RS_Ranks, RS_Level0, RS_LevelMin}`）。RS 按季度独立拟合：它没有可以预训练或热启动的迭代参数。

#### Word2Vec（W2V）

在按排序的组合序列上做 Skip-gram：

$$\mathcal{L}_{\text{W2V}} = -\sum_{(i,t)\in D}\ \sum_{c=1}^{L_{i,t}}\ \sum_{0<|j-c|\le w}
\log \mathbb{P}\!\big(a_{i,t}(j)\mid a_{i,t}(c)\big).$$

训练得到的输入矩阵 $\boldsymbol{E}_{\text{in}}^{*}$ 给出资产嵌入。

代码：`asset_embeddings/trainers/w2v.py`（CLI `asset_embeddings/scripts/train/w2v.py`；gensim Skip-gram/CBOW + 负采样）、`AssetW2VTrainConfig`。

#### BERT

一个 Transformer 编码器，用遮蔽语言建模目标、以遮蔽率 $p_{\text{mask}}=0.15$ 训练：

$$\mathcal{L}_{\text{MLM}} = -\sum_{(i,t)\in D}\ \mathbb{E}_{M_{i,t}}\sum_{j\in M_{i,t}}
\log \mathbb{P}_\theta\!\big(a_{i,t}(j)\mid \tilde{\pi}_{i,t}\big).$$

季度 $t$ 的资产嵌入是行 $[\boldsymbol{E}_{\text{emb}}^{*}]_a$；编码器与头在训练后丢弃。

代码：`asset_embeddings/trainers/bert.py`（CLI `asset_embeddings/scripts/train/bert.py`）；`asset_embeddings.modules.BertEmbeddings`；随机遮蔽位置的切分是 `asset_embeddings.datasets.mlm.MaskingCollator`；`AssetBERTTrainConfig`。

### 预训练-微调范式

目标是一个*时变*嵌入：每只股票每季度一个向量，使得从一个季度到下一个季度的变化本身就富含信息。这会撞上一个可识别性问题。神经模型学到的嵌入只在隐空间坐标轴的任意旋转、反射与重标号意义下被确定——在同一份数据上从不同随机起点训练两次，你会得到两个同样有效、却看上去截然不同的嵌入。于是，若每个季度独立训练，季度间的差 $\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$（股票 $a$ 在 $t{+}1$ 的向量减去其在 $t$ 的向量）多半只在度量那种任意的坐标漂移，而非股票的任何真实变化。

补救之道，是给每个季度**相同的起点**。模型在汇总的早期季度语料上*预训练*一次，产生单一的基解；随后每个评估季度都从这个共享基解*微调*，而非从零开始。因为所有季度从同一锚点出发、且只移动一点点，它们的嵌入便停留在同一坐标系中，$\boldsymbol{x}_{a,t+1}-\boldsymbol{x}_{a,t}$ 也就转而反映股票的真实移动。

这里的 BERT 有两个可训练部分需要初始化——一个**嵌入层**（每只股票一个向量）和一个 Transformer **编码器**——而“从共享基解出发”恰恰是 BERT 配置（`templates/train_main.json`）中两个字段所设定的：

| 阶段 | `model.model_checkpoint`（为编码器播种） | `model.w2v_model` = `tokenizer.w2v_model`（为嵌入层播种） |
|---|---|---|
| **BERT 预训练** | `null`——编码器从零训练 | `checkpoints/pretrained/AssetW2V/d{dim}/AssetW2V_d{dim}_pretrained.model` |
| **BERT 微调，季度 $t$** | `checkpoints/pretrained/AssetBERT/d{dim}/AssetBERT_d{dim}_pretrained_best/model.safetensors` | `checkpoints/finetune/AssetW2V/d{dim}/{t}/AssetW2V_d{dim}_{t}.model` |

- **`model.model_checkpoint`** 为编码器播种。预训练时为 `null`，故编码器在汇总语料上从零训练。微调时为预训练好的 BERT，故编码器被沿用、仅被轻推；正是这一点让每个季度的解处在同一个共享坐标系中。（遮蔽语言模型的预测头每次都重新初始化；它在训练后本就被丢弃。）
- **`model.w2v_model`** 为嵌入层播种：`AssetBERT_Preparer` 把一个训练好的 Word2Vec 模型的向量拷入 BERT 的嵌入矩阵（见 [Preparers](preparers.md)）；这就是**热启动**。预训练时它是在汇总语料上训练的 W2V。微调时，播种发生在检查点加载*之后*，故季度 $t$ 自己的 W2V 覆盖嵌入层，而编码器保持从预训练 BERT 加载的状态。
- **`model.embedding_file`** 做同样的嵌入播种，只是从一份嵌入 CSV 而非 Word2Vec 模型读取；在主流水线中为 `null`。

于是每个微调后的季度都是*共享的预训练编码器加上该季度自己的嵌入层*，再经一次针对该季度持仓的短暂微调加以调整。正是这个共享编码器，把逐季度的嵌入跨时间地固定在同一可比坐标系中。完整的逐阶段配置见 [训练](../how-to/train.md)。

### 训练设置

所有维度共用一套 BERT 配置；下表是 `templates/train_main.json` 与 `templates/train_timesplit_robust.json` 生成的设置，其余训练模板与之一致。

| | 预训练 | 微调 |
|---|---|---|
| 架构 | 4 层、2 头、FFN 512、上下文 256（对每个 $d$ 相同） | 相同，编码器从预训练检查点载入 |
| 数据 | 汇总的早期季度 | 单个季度的全部组合 |
| 轮数 | 至多 40，cosine 调度 | 30，cosine 调度 |
| 选择 | 在 10 % 随机验证集（seed 42）上按验证损失取最优轮；连续 5 轮相对改善不足 0.1 % 即早停 | 按训练损失取最优轮，无验证集，不早停 |
| 优化器 | AdamW，lr $10^{-3}$，warmup 500 步，梯度裁剪 1.0 | PagedAdam8bit，lr $10^{-3}$，无 warmup，梯度裁剪 1.0 |
| 批大小 / 掩码 | batch 128，掩码概率 0.15 | batch 32，掩码概率 0.15 |

预训练之所以在留出数据上选轮，是因为按训练损失的规则分不清"已收敛"和"几轮之后发散"。微调沿用原来的规则，让该季度的嵌入看到该季度的每一个组合，MP 分数因此读作可达到的最高预测水平。

### 三项基准

嵌入在三个互补维度上被打分。各估计量及其数学见 [评估](../how-to/evaluate.md)；映射关系：

| 基准 | 代码 |
|---|---|
| **RV**——相对估值（估值残差对嵌入的横截面解释性 $R^2$，在该季度全部公司上做 ridge 拟合） | `asset_embeddings.evaluators.rv` · `asset_embeddings/scripts/eval/rv.py` · `RVEvalConfig` |
| **RC**——收益共动（下一季度收益对嵌入的横截面预测性 $R^2$） | `asset_embeddings.evaluators.rc` · `asset_embeddings/scripts/eval/rc.py` · `RCEvalConfig` |
| **MP**——管理型组合（被遮蔽资产的归一化对数似然） | `asset_embeddings/trainers/mp.py`（`MPBertTrainer` / `MPSimplifiedMLMTrainer`；CLI `asset_embeddings/scripts/eval/mp_bert.py` / `mp_smlm.py`）；`asset_embeddings.evaluators.mp.mp_token_scores` 返回逐词元的归一化对数似然；`MPResult` |

RV 与 RC 默认在样本内拟合；两个配置上的 `scoring: "holdout"` 是另一个独立选项——按公司切分，在多个带种子的切分上于留出公司打分（见下文的 [`holdout` 族](#extensions)，论文未使用的扩展）。时序 RC（`asset_embeddings.evaluators.rc_temporal` · `asset_embeddings/scripts/eval/rc_temporal.py` · `TemporalRCEvalConfig`）是另一个选项：在前一季度拟合的 RC 斜率，或此前各季度斜率的均值，不经重估用于下一季度，嵌入取原始坐标或经 Procrustes 对齐（见 [`rc_temporal` 族](#extensions)）。

### Text-embedding 对照

商用 LLM 基线——嵌入公司*名称*而非持仓——来自在本仓库之外准备的 OpenAI / Cohere / Voyage / Gemini / Qwen / BGE 嵌入，随后通过同一套 RV/RC/MP 评估器打分。

## 实验

实验按论文的顺序排列：先正文各节，再各附录。每一条说明论文在该处提出的问题、分析改变了什么、固定了什么，以及产出这些数字的实验族。

一个**族（family）**是一份 [`fast_generator`](../how-to/generate-configs.md) 模板（展开配置网格），加上一个 `scripts/run/*.sh` 运行脚本，它设定该族的 `runs.experiment` 标签（`EXP=…`）并运行整个网格；如何启动、过滤与重跑一个族，见[运行实验](../how-to/run-experiments.md)。除条目中另行注明者外，标签与族名相同。仅评估的族不训练新模型：它们从另一个角度读取 `main`（或 `timesplit_robust`）的检查点。

全程维度 $d \in \{4,8,10,16,32,64,128\}$。主切分在 2005Q1–2023Q1 上预训练，在 2023Q2–2024Q3 六个季度上评估；50/50 时间切分在 2005Q1–2020Q3 上预训练，在 2020Q4–2024Q3 十六个季度上评估。

### §3 数据

**问题。** 公募基金在 Q2/Q4 披露全部持仓，在 Q1/Q3 只披露前十大，观测到的面板因此在两种制度之间锯齿式交替。要用同一个模型在历史上预训练、再在每个评估季度微调，样本必须在各季度之间保持可比。

**做了什么。** $(10,10)$ 共同持有筛选：股票至少被十个投资者持有、投资者至少持有十只股票时，该持仓才保留；面板按时间切分，前 80 % 的持仓观测用于预训练，其余用于微调与评估（切点 2023Q1）。附录 J 的 50/50 切分由同一流程产出。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| 数据预处理 | 持仓序列、基准数据、行业标签、§7.1 的国有属性标签、80/20 与 50/50 切分；在本仓库之外准备。 | — | — |

### §4 基准表现（RV、RC、MP）

**问题。** 从中国不完整的机构持仓中学到的资产嵌入，能否解释估值、预测收益共动、刻画投资者据以构建组合的相似性？三种架构的排序如何？在 Gabaix 等人的美国证据中，推荐系统领先，BERT 只在被遮蔽资产预测上突出。

**做了什么。** 六个嵌入模型（BERT、W2V、RS-Binary、RS-Ranks、RS-Level0、RS-LevelMin）在七个维度上，于六个评估季度在三项基准上打分，并与公司特征基线比较。对 RV 与 RC，还把 BERT 与申万一级行业哑变量和公司特征组合，共七组回归元，检验嵌入是否含有两者之外的信息。MP 让所有模型经过同一个 BERT 结构的 refine 流程打分，只有嵌入层不同。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **main** | 依次训练 W2V、BERT 预训练、逐季度 BERT 微调，以及四个 RS 变体；打分 RV、RC（含与不含公司特征）与 MP。BERT 遵循上文的[训练设置](#训练设置)。 | `train_main.json` · `eval_main.json` | `train_main.sh` · `eval_main.sh` |
| **industry** | 仅行业哑变量、加公司特征、加 BERT、三者兼有。复用 `main` 嵌入。 | `eval_industry.json` | `eval_industry.sh` |

### §5.1 预训练-微调的作用

**问题。** 预训练-微调范式是否必要？在美国它没有带来提升；但在中国，混合披露规则使观测到的持仓逐季度剧烈变化。

**做了什么。** 在每个维度上比较三种形式的 BERT：完整范式；每个季度从随机初始化训练的同一模型（无预训练，也无 Word2Vec 热启动）；不做季度微调的预训练模型（无微调）。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **no_pt_init**（标签 `pt_strategy_ablation/no_pt_init`） | 无预训练：每个季度的 BERT 从随机初始化出发，嵌入层也不例外；当季的 Word2Vec 模型只提供词表。 | `train_no_pt_init.json` · `eval_no_pt_init.json` | `train_no_pt_init.sh` · `eval_no_pt_init.sh` |
| **static** | 无微调：预训练的 BERT（及 W2V）嵌入在每个评估季度上打分。复用 `main` 检查点。 | `eval_static.json` | `eval_static.sh` |

### §5.2 热启动的作用

**问题。** 既然预训练-微调重要，用 Word2Vec 嵌入（已按共同持有把资产分组）初始化 BERT 的嵌入层是否也重要？

**做了什么。** 在其余一切固定的条件下，比较两个阶段都使用与都不使用 W2V 热启动的 BERT（预训练用合并语料上的 W2V，逐季度微调用该季度的 W2V）。该族运行两个阶段的完整 2×2；论文报告“两阶段都用”与“都不用”的对比，并注明两个阶段互补。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **w2v_init_ablation** | 预训练阶段 × 微调阶段是否热启动的 2×2。 | `train_w2v_init_ablation.json` · `eval_w2v_init_ablation.json` | `train_w2v_init_ablation.sh` · `eval_w2v_init_ablation.sh` |

### §6 资产嵌入与文本嵌入

**问题。** 商用文本嵌入对公司名称编码了行业等公开描述信息。它们能否比得上基于持仓的嵌入？两者在哪里不同？

**做了什么。** 九个文本嵌入模型对每家公司的中文名称做嵌入，按季度 PCA 降到 $d$ 维，在与资产嵌入相同的三项基准上打分。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **text_embedding** | 经 RV、RC、MP 为名称嵌入（OpenAI、Cohere、Voyage、Gemini、Qwen、BGE；在本仓库之外获取并降维）打分。 | `eval_text_embedding.json` | `eval_text_embedding.sh` |

### §7.1 国有属性与资产嵌入的几何结构

**问题。** 仅从持仓学到的嵌入，能否把国有控制的公司与民营公司区分开？这一区分又在多大程度上组织了截面？

**做了什么。** 依据 CSMAR 控股股东文件中实际控制人的性质，把每个公司-年标为国有或民营，该标签在本仓库之外准备。交叉验证的逻辑回归从嵌入中还原该标签，按模型与维度以 AUC 打分，并与 RV 的四个公司特征对照。在国有公司内与民营公司内分别重估 RV 与 RC。每一对公司的嵌入余弦相似度对“同属性”与“同行业”指示变量回归，逐季度系数在六个季度上按 Fama-MacBeth 方式取平均。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **state_ownership** | 六个模型、七个维度的标签探针、以 `SOE` 做 `grouping: "categorical"` 的分组 RV 与 RC、成对相似度回归，外加公司特征基线（171 次运行）。复用 `main` 嵌入。见[标签探针](../how-to/evaluate.md#label-probe)、[分组评估](../how-to/evaluate.md#grouped-evaluation)与[成对相似度](../how-to/evaluate.md#pairwise-similarity)。 | `eval_state_ownership.json` | `eval_state_ownership.sh` |

### §7.4 机构覆盖度与嵌入的信息含量

**问题。** 许多 A 股公司只被少数几家机构披露持有。对这条覆盖稀薄的尾部，嵌入是否仍有信息？

**做了什么。** 每个季度按披露持有的机构数把公司分成四个分位组，在每组内重估 RV 与 RC，并在同一组内重算公司特征基线作为对照。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **coverage** | 六个模型、七个维度按覆盖度分组的 RV 与 RC，分位数分组与固定分箱两种模式，外加公司特征基线。复用 `main` 嵌入。见[评估](../how-to/evaluate.md#grouped-evaluation)。 | `eval_coverage.json` | `eval_coverage.sh` |

### §7.5 资产嵌入需要多少披露？

**问题。** 中国一年两次披露全部持仓，其间只披露较薄的一层。如果一条披露规则支配整个历史，它能隐去多少信息，微调后的嵌入才不再优于全披露下从未微调的嵌入？

**做了什么。** 把披露制度施加于未经筛选持仓的每一个季度，预训练历史与评估季度一视同仁。深度维度上，基金披露按市值排名的前 $N$ 大持仓，同时每只股票持股比例至少 0.3 % 的前十大股东在公司一侧披露；广度维度上，只有固定比例的机构披露。随机对照以独立同分布的抽取保留同样比例的记录。之后不再做共同持有筛选：规则本身就是筛选。三条阶梯上的每一级都在 50/50 切分上训练自己的 W2V 与 BERT 链（默认 $d=16$），并在十六个评估季度上以 RV 与 RC 打分。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **disclosure_regime**（标签 `disclosure_regime/{level}/s{seed}`） | 使用在本仓库之外准备的披露制度面板（深度 `top{N}`、广度 `keep{r}_s{draw}`、随机 `rand{r}_s{draw}`），对每一级与每个训练种子训练一条链，并对微调与从未微调的嵌入打分 RV 与 RC；`DIM=128` 运行 $d=128$ 网格。 | `train_disclosure_regime.json` · `eval_disclosure_regime.json` | `train_disclosure_regime.sh` · `eval_disclosure_regime.sh` |

### 附录 A：估计与计算细节

超参数表与训练曲线描述的是 `main` 族，其 BERT 设置即上文的[训练设置](#训练设置)。

### 附录 B：Skip-gram 与 CBOW

**问题。** Gabaix 等人使用 CBOW，本文使用 skip-gram。这一选择对单独的 W2V 嵌入以及作为 BERT 热启动时是否有影响？

**做了什么。** 其余一切固定，把 W2V 目标在 skip-gram 与 CBOW 之间切换，对 W2V 嵌入和在其上构建的 BERT 都打分。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **w2v_model_ablation** | 比较 CBOW 与 skip-gram，既作为单独的嵌入，也作为 BERT 的热启动。 | `train_w2v_model_ablation.json` · `eval_w2v_model_ablation.json` | `train_w2v_model_ablation.sh` · `eval_w2v_model_ablation.sh` |

### 附录 E：Ridge 惩罚敏感性

**问题。** RV 与 RC 得分以及模型排序是否依赖固定的 ridge 惩罚 $\lambda = 0.5$？

**做了什么。** 在 $[0, 10]$ 上的九个固定惩罚值与季度内 10 折交叉验证下，对主网格重新打分。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **ridge_robust** | 惩罚扫描与 CV 惩罚下的 RV 与 RC。复用 `main` 嵌入。 | `eval_ridge_robust.json` | `eval_ridge_robust.sh` |

### 附录 F：MP 中的下游架构与编码器初始化

**问题。** Gabaix 等人用 BERT 自带的编码器评估 BERT，却用线性头（SMLM）评估 W2V 与 RS。MP 的排序是由下游架构驱动的吗？BERT 热启动的编码器又起了多大作用？

**做了什么。** 五个非 BERT 嵌入分别在 SMLM 与 BERT 结构的 refine 流程下打分（F.1–F.3）。F.4 把 BERT 嵌入与随机初始化的编码器配对，即其他模型所用的同一起点。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **main**（SMLM 行） | SMLM 头下的 MP，与 §4 的 refine 流程行一起运行。 | `eval_main.json` | `eval_main.sh` |
| **mp_bert_random_encoder** | refine 流程中随机编码器下的 BERT 嵌入（F.4）。 | `eval_mp_bert_random_encoder.json` | `eval_mp_bert_random_encoder.sh` |

### 附录 G：遮蔽位置敏感性

**问题。** MP 遮蔽排名第 2 的持仓。模型排序与得分形状是否依赖遮蔽哪一位？

**做了什么。** 对每个模型与维度，在 $k \in \{1,2,3,5,10\}$ 上重估 $\mathrm{MP}_k$。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **mask_position** | 五个遮蔽位置上的 MP，refine 流程与 SMLM 两种下游。 | `eval_mask_position.json` | `eval_mask_position.sh` |

### 附录 H：预测性评估

**问题。** RV 与 MP 在同一季度读取嵌入与目标。第 $t$ 季度的嵌入对第 $t+1$ 季度是否仍有信息？

**做了什么。** Pred-RV 把第 $t+1$ 季度的估值残差对第 $t$ 季度的嵌入回归；Pred-MP 冻结第 $t$ 季度的嵌入，在第 $t+1$ 季度的持仓上 refine。两者都使用主窗口内的五对相邻季度。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **pred** | Pred-RV 与 Pred-MP。复用 `main` 检查点。 | `eval_pred.json` | `eval_pred.sh` |

### 附录 I：文本嵌入稳健性

**问题。** §6 的结论是否依赖于用中文读取公司名称，或依赖于按季度的 PCA？

**做了什么。** 用中文与英文名称、三种降维方式（在合并截面上拟合一次的 PCA、每季度重拟合的 PCA、原生截断），在全部七个维度上重跑文本对照。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **text_embedding** | 即 §6 的族，覆盖两种语言与全部降维方式。 | `eval_text_embedding.json` | `eval_text_embedding.sh` |

### 附录 J：扩展窗口稳健性

**问题。** 以更短的预训练历史为代价换取更长的评估窗口，§4 的排序是否仍然成立？

**做了什么。** 在 50/50 切分上重跑整条流水线，在十六个季度上评估。

| 族 | 作用 | 模板 | 运行脚本 |
|---|---|---|---|
| **timesplit_robust** | 在 50/50 切分上训练与评估。BERT 遵循上文的[训练设置](#训练设置)。 | `train_timesplit_robust.json` · `eval_timesplit_robust.json` | `train_timesplit_robust.sh` · `eval_timesplit_robust.sh` |

## 扩展 { #extensions }

论文未报告的族。它们复用同样的数据、检查点与评估器，作为可选项保留。

| 族 | 它做什么 | 模板 | 脚本 |
|---|---|---|---|
| **rc_temporal** | `main`（2023Q2–2024Q3）与 `timesplit_robust`（2020Q4–2024Q3）网格的时序 RC：在季度 $T-1$ 拟合的 RC 斜率（`previous`）或此前各季度斜率的均值（`expanding`）不经重估用于季度 $T$（跨期），嵌入取原始坐标或经 Procrustes 对齐到拟合季度，分 `fixed` 与 `cv` 两个惩罚档；六个模型 × 七个维度 × 两种对齐、特征基线，以及未经微调的预训练 BERT 与 Word2Vec 嵌入，共 792 次运行。复用 `main` 与 `timesplit_robust` 的预训练与微调嵌入。见[评估](../how-to/evaluate.md#temporal-rc)。 | `eval_rc_temporal.json` | `eval_rc_temporal.sh` |
| **holdout** | `main` 网格的 RV 与 RC 按公司切分评估（`scoring: "holdout"`）：每季度 20 个带种子的 80/20 切分，在留出公司上打分，两个惩罚档背靠背运行，`fixed`（`ridge_alpha` 0.5）与 `cv`（`ridge_alpha` null，在每个训练切分内做 RidgeCV）。每期取值为各切分的中位数，逐切分分数写入 `eval_ridge_splits`。见[评估](../how-to/evaluate.md#holdout-evaluation)。 | `eval_holdout_main.json` | `eval_holdout_main.sh` |
| **contextual** | BERT 上下文（按规模加权）嵌入与静态嵌入行的对比。 | `eval_contextual.json` | `eval_contextual.sh` |
| **timesplit_pred** / **timesplit_static** | 在 50/50 检查点上的预测性评估与无微调评估。 | `eval_timesplit_pred.json` · `eval_timesplit_static.json` | `eval_timesplit_pred.sh` · `eval_timesplit_static.sh` |
| **w2v_model_ablation_timesplit** / **mp_bert_random_encoder_timesplit** | 附录 B 与附录 F.4 在 50/50 切分上的版本。 | `train_w2v_model_ablation_timesplit.json` · `eval_w2v_model_ablation_timesplit.json` · `eval_mp_bert_random_encoder_timesplit.json` | `train_w2v_model_ablation_timesplit.sh` · `eval_w2v_model_ablation_timesplit.sh` · `eval_mp_bert_random_encoder_timesplit.sh` |
| **semiannual_robust** | 仅在 Q2/Q4 季度上训练与评估。 | `train_semiannual_robust.json` · `eval_semiannual_robust.json` | `train_semiannual_robust.sh` · `eval_semiannual_robust.sh` |
| **semiannual_timesplit** | 50/50 切分的 Q2/Q4 子集。 | `train_semiannual_timesplit.json` · `eval_semiannual_timesplit.json` | `train_semiannual_timesplit.sh` · `eval_semiannual_timesplit.sh` |
| **expansion** | 累积预训练：对每个季度 $t$，在 $t$ 之前的全部数据上从零预训练，再在 $t$ 上微调。 | `train_expansion.json` · `eval_expansion.json` | `train_expansion.sh` · `eval_expansion.sh` |
| **sliding_window** N{8,20,40}（标签 `sliding_window/N{8,20,40}`） | 仅在最近 N 个季度（2、5 或 10 年）上预训练。 | `train_sliding_window_N{8,20,40}.json` · `eval_sliding_window_N{8,20,40}.json` | `train_sliding_window.sh` · `eval_sliding_window.sh` |
