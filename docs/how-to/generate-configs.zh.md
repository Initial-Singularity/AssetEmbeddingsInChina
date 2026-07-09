# 生成配置

> 批量配置文件与脚本自动生成工具

## 1. 引言

### 1.1 背景与动机

在机器学习、数据分析或自动化实验中，常常需要**批量生成配置文件**与**批量运行脚本**：

- **超参调优**：训练模型时，需要在不同超参组合下重复实验（如 `dim=128, lr=0.001` vs `dim=256, lr=0.01`）
- **时间序列实验**：需要生成具有不同时间窗口与数据集配置的 JSON 文件，以驱动下游程序
- **批量任务提交**：实验完成后，需要一键生成运行脚本，统一提交到集群或服务器

这些任务**重复而机械**：

- 手工维护 10 个参数组合，可能要复制粘贴 10 个配置文件或脚本
- 一旦参数增加到 3–4 个维度，组合数指数增长，手工维护几乎不可能

### 1.2 设计目标

**核心功能**：

- 通过**模板 + 变量参数**，自动批量生成 JSON 配置文件与脚本文件
- 支持**参数笛卡尔积**、**变量绑定**、**函数生成时间/数字序列**等机制，灵活组合
- 兼顾**命令行友好**（便于直接调用）与**类接口可扩展**（便于开发者定制）

**典型用例**：

1. **机器学习/深度学习实验**：为不同超参组合批量生成配置文件与运行脚本
2. **数据处理流水线**：为不同日期、季度、参数设置批量生成任务配置
3. **自动化批处理**：快速生成成千上万的配置/脚本，避免重复劳动



## 2. 快速上手

### 2.1 示例 1：生成配置文件

假设你有一个模型配置模板 `config_template.json`：

```json
{
  "model": "resnet",
  "dim": "{dim}",
  "lr": "{lr}"
}
```

我们想为不同的 `dim` 与 `lr` 组合生成配置文件。只需运行：

```bash
uv run python -m scripts.tools.fast_generator config \
    -t config_template.json \
    -v '{"dim":[128,256], "lr":[0.001,0.01]}' \
    -o configs/exp_{dim} \
    -n config_{lr}.json
```

### 结果

目录结构：

```
configs/
 ├─ exp_128/
 │   ├─ config_0.001.json
 │   └─ config_0.01.json
 └─ exp_256/
     ├─ config_0.001.json
     └─ config_0.01.json
```

某个文件的内容（`configs/exp_128/config_0.001.json`）：

```json
{
  "model": "resnet",
  "dim": 128,
  "lr": 0.001
}
```



### 2.2 示例 2：用函数生成参数

无需手工写出所有参数值；可以用内置函数自动生成：

```bash
uv run python -m scripts.tools.fast_generator config \
    -t config_template.json \
    -v '{"quarter":"@generate_time_quarters(2020,1,2021,4)"}'
```

这会生成 8 个配置文件，`quarter` 取值为 `2020Q1 … 2021Q4`。



### 2.3 示例 3：生成脚本文件

假设你有一个脚本预设文件 `script_preset.json`：

```json
{
  "output_dir": "scripts",
  "filename": "train.sh",
  "commands": [
    {
      "template": "python train.py --dim {dim} --lr {lr}",
      "variable_params": {"dim":[128,256], "lr":[0.001,0.01]}
    }
  ]
}
```

运行：

```bash
uv run python -m scripts.tools.fast_generator script -f script_preset.json
```

生成 `scripts/train.sh`：

```bash
python train.py --dim 128 --lr 0.001
python train.py --dim 128 --lr 0.01
python train.py --dim 256 --lr 0.001
python train.py --dim 256 --lr 0.01
```

这样即可直接批量提交训练任务。（空行只插在命令块*之间*，不插在同一块的各变体之间。）



## 3. 核心概念

程序的核心思想是：**模板 + 变量参数 + 变量绑定 + 输出控制**

> **核心要素概览**
> - **模板（Template）**：定义生成的基本结构，用占位符标记可变部分
> - **变量参数（Variable Params）**：指定参数取值范围，自动展开为笛卡尔积
> - **变量绑定（Variable Bindings）**：约束参数间关系，避免无效组合
> - **输出控制（Output Control）**：用占位符自定义输出路径与文件名
> - **函数调用（Function Calls）**：通过内置或自定义函数动态生成参数值



### 3.1 模板

模板是生成的“基本骨架”。它可以是：

- JSON 对象（用于配置文件）
- 字符串（用于命令/脚本）

模板中，参数位置可用**占位符**表示：

- `{var}` → 替换为参数值
- `{var:format}` → 按格式替换参数值（支持 Python 格式语法）

示例模板：

```json
{
  "model": "resnet",
  "dim": "{dim}",
  "lr": "{lr:.4f}"
}
```

若参数组合为 `{"dim":128, "lr":0.001}`，生成结果为：

```json
{
  "model": "resnet",
  "dim": 128,
  "lr": 0.0010
}
```



### 3.2 变量参数

变量参数定义哪些取值需要被**枚举并组合**。

- 在命令行中，用 `-v` 传入 JSON 字符串
- 在预设文件中，用 `variable_params` 指定

示例：

```json
{"dim":[64,128], "lr":[0.001,0.01]}
```

将自动展开为 4 个组合：

```
(dim=64, lr=0.001)
(dim=64, lr=0.01)
(dim=128, lr=0.001)
(dim=128, lr=0.01)
```

生成对应的 4 个配置或命令。



### 3.3 变量绑定

有些参数不是“自由组合”，而是“必须绑定”的。例如：

- **错误做法**：`size ∈ {tiny, base}, dim ∈ {64,128}` → 4 个组合，但 `tiny` 可能只应对应 `64`
- **解决办法**：用**变量绑定**显式约束关系：

```json
[
  {"size": "tiny", "dim": 64},
  {"size": "base", "dim": 128}
]
```

这样只会生成 2 个有效组合。

注意：绑定与自由参数可同时使用。

- 绑定关系部分保持不变
- 其他自由参数仍参与组合

示例：

```json
{
  "variable_params": {
    "epochs": [10,20],
    "dim": [64,128],
    "size": ["tiny","base"]
  },
  "variable_bindings": [
    {"size": "tiny", "dim": 64},
    {"size": "base", "dim": 128}
  ]
}
```

生成 4 个组合：

```
(size=tiny, dim=64, epochs=10)
(size=tiny, dim=64, epochs=20)
(size=base, dim=128, epochs=10)
(size=base, dim=128, epochs=20)
```



### 3.4 输出控制

类似地，输出路径与文件名也可用占位符变量自定义。

示例：

```bash
-o configs/{size} -n config_{dim}.json
```

若参数组合为 `{"size":"tiny","dim":64}`，生成的文件为：

```
configs/tiny/config_64.json
```

另一组合 `{"size":"base","dim":128}` 生成：

```
configs/base/config_128.json
```



### 3.5 函数调用

除了手工列出所有取值，变量参数列表也可由已注册的函数动态生成。在字符串中用 `@` 指定函数调用与参数。

语法：

```
"@function_name(arg1, arg2, ...)"
```

示例：

```json
{"quarter": "@generate_time_quarters(2020,1,2021,4)"}
```

→ 生成 `["2020Q1","2020Q2","2020Q3","2020Q4","2021Q1","2021Q2","2021Q3","2021Q4"]`

与绑定或其他参数组合时，会自动展开组合。



## 4. 内置便捷函数

在变量参数（`variable_params`）中，可通过**函数调用**动态生成参数值。语法格式：

```
"@function_name(arg1, arg2, key=value, ...)"
```

函数调用返回的结果会被当作一组参数值，参与组合生成。



### 4.1 时间序列相关函数

**(1) `generate_time_quarters(start_year, start_quarter, end_year, end_quarter, format="%YQ%Q")`**

生成季度序列。

示例：

```json
{"quarter": "@generate_time_quarters(2020,1,2021,4)"}
```

结果：

```
["2020Q1","2020Q2","2020Q3","2020Q4","2021Q1","2021Q2","2021Q3","2021Q4"]
```



**(2) `generate_time_months(start_year, start_month, end_year, end_month, format="%Y-%m")`**

生成月度序列。

示例：

```json
{"month": "@generate_time_months(2020,1,2020,6)"}
```

结果：

```
["2020-01","2020-02","2020-03","2020-04","2020-05","2020-06"]
```



**(3) `generate_date_range(start_date, end_date, freq="D", format="%Y-%m-%d")`**

生成不同频率的日期范围：

- `D` → 日
- `W` → 周
- `M` → 月
- `Q` → 季
- `Y` → 年

示例：

```json
{"date": "@generate_date_range('2020-01-01','2020-01-10',freq='D')"}
```

结果：

```
["2020-01-01","2020-01-02",...,"2020-01-10"]
```



### 4.2 数字序列生成函数

**(1) `range(start, stop, step=1)`**

类似 Python 内置 `range`，生成整数序列。

```json
{"x": "@range(0,5)"}
```

结果：

```
[0,1,2,3,4]
```



**(2) `linspace(start, stop, num)`**

生成等间隔序列（含端点）。

```json
{"x": "@linspace(0,1,5)"}
```

结果：

```
[0.0,0.25,0.5,0.75,1.0]
```



**(3) `logspace(start, stop, num)`**

生成以 10 为底的对数刻度序列。

```json
{"x": "@logspace(0,2,3)"}
```

结果：

```
[1.0, 10.0, 100.0]
```



**(4) `geomspace(start, stop, num)`**

生成从 `start` 到 `stop` 的等比数列。

```json
{"x": "@geomspace(1,100,3)"}
```

结果：

```
[1.0, 10.0, 100.0]
```



### 4.3 列表操作函数

**(1) `repeat(value, times)`**

把一个值重复多次。

```json
{"flag": "@repeat('debug',3)"}
```

结果：

```
["debug","debug","debug"]
```



**(2) `zip_lists(list1, list2, ...)`**

按索引位置把多个列表绑在一起。

```json
{"pair": "@zip_lists([1,2],[10,20])"}
```

结果：

```
[[1,10],[2,20]]
```



**(3) `generate_cross_product(list1, list2, ...)`**

生成多个列表的笛卡尔积。

```json
{"pair": "@generate_cross_product([1,2],[3,4])"}
```

结果：

```
[[1,3],[1,4],[2,3],[2,4]]
```



### 4.4 自定义函数扩展

开发者可通过 `FunctionRegistry.register(name, func)` 注册新函数。

示例：

```python
from scripts.tools.fast_generator import FunctionRegistry

def square_numbers(n):
    return [i*i for i in range(1,n+1)]

registry = FunctionRegistry()
registry.register("squares", square_numbers)

print(registry.call("squares", 5))
# Output: [1,4,9,16,25]
```

然后即可在模板参数中使用：

```json
{"value": "@squares(5)"}
```



## 5. 配置文件生成器（ConfigGenerator）

`ConfigGenerator` 用于基于模板与参数组合批量生成 JSON 配置文件。它是最常用的生成器，适合批量实验与任务自动化场景。



### 5.1 基本原理

1. **模板**：带占位符的 JSON 配置文件
2. **变量参数**：要枚举的参数范围
3. **变量绑定**：参数间的约束关系
4. **输出控制**：生成配置文件的输出目录与文件名模板

程序会自动展开所有参数组合，把 `{var}` 占位符替换为实际值，最终输出一系列 JSON 配置文件。



### 5.2 使用预设文件生成

可通过 `--from-preset` 加载一个 JSON 文件，它定义了全部生成规则。

命令示例：

```bash
uv run python -m scripts.tools.fast_generator config -f configs_preset.json
```

这里的 `configs_preset.json` 是一个任务列表，可包含一个或多个生成任务。例如：



#### 示例 1：AssetW2V Finetune 任务

```json
{
  "name": "AssetW2V Finetune Config Generator",
  "description": "This config generator is used to generate the configs for the AssetW2V finetune model, ranged by embedding dimensions and quarterly data.",
  "template": {
    "pretrained_model": "model/pretrained/AssetW2V/d{dim}/AssetW2V_d{dim}_pretrained.model",
    "embedding_dim": "{dim}",
    "epochs": 10,
    "window": 5,
    "min_count": 1,
    "sg": 1,
    "sample": 1e-3,
    "negative_sample": 10,
    "seed": 42,
    "data_path": "data/processed/MutualFundShareHoldings/finetune/{quarter}.csv",
    "data_format": "csv",
    "id_key": "InvestorID",
    "portfolio_key": "Portfolio",
    "workers": 1,
    "save_folder": "model/finetune/AssetW2V/d{dim}",
    "save_name": "AssetW2V_d{dim}_{quarter}",
    "save_format": ".model"
  },
  "output_dir": "configs/finetune/AssetW2V/d{dim}",
  "filename": "AssetW2V_d{dim}_{quarter}.json",
  "variable_params": {
    "dim": [4, 8, 10, 16, 32, 64, 128],
    "quarter": ["2023Q2","2023Q3","2023Q4","2024Q1","2024Q2","2024Q3"]
  }
}
```

运行后将生成配置文件：

```
configs/finetune/AssetW2V/
 ├─ d4/AssetW2V_d4_2023Q2.json
 ├─ d4/AssetW2V_d4_2023Q3.json
 ...
 ├─ d128/AssetW2V_d128_2024Q3.json
```

其中 `AssetW2V_d4_2023Q2.json` 内容类似于：

```json
{
  "pretrained_model": "model/pretrained/AssetW2V/d4/AssetW2V_d4_pretrained.model",
  "embedding_dim": 4,
  "epochs": 10,
  "window": 5,
  "min_count": 1,
  "sg": 1,
  "sample": 0.001,
  "negative_sample": 10,
  "seed": 42,
  "data_path": "data/processed/MutualFundShareHoldings/finetune/2023Q2.csv",
  "data_format": "csv",
  "id_key": "InvestorID",
  "portfolio_key": "Portfolio",
  "workers": 1,
  "save_folder": "model/finetune/AssetW2V/d4",
  "save_name": "AssetW2V_d4_2023Q2",
  "save_format": ".model"
}
```



#### 示例 2：AssetBERT Finetune 任务（带变量绑定）

在 `AssetBERT` 配置中，`dim`、`intermediate_size`、`max_position_embeddings` 三个参数必须对应。这就需要用**变量绑定（variable_bindings）**。

```json
{
  "name": "AssetBERT finetune Config Generator",
  "description": "This config generator is used to generate the configs for the AssetBERT finetune model, ranged by embedding dimensions and quarterly data.",
  "template": {
    "model": {
      "model_type": "BERT",
      "hidden_size": "{dim}",
      "intermediate_size": "{intermediate_size}",
      "max_position_embeddings": "{max_position_embeddings}"
    },
    "train": {
      "save_folder": "model/finetune/AssetBERT/d{dim}/{quarter}",
      "save_name": "AssetBERT_d{dim}_{quarter}",
      "save_format": ".safetensors"
    }
  },
  "output_dir": "configs/finetune/AssetBERT/d{dim}",
  "filename": "AssetBERT_d{dim}_{quarter}.json",
  "variable_params": {
    "dim": [4, 8, 10, 16, 32, 64, 128],
    "quarter": ["2023Q2","2023Q3","2023Q4","2024Q1","2024Q2","2024Q3"],
    "intermediate_size": [512,512,512,512,512,1024,1024],
    "max_position_embeddings": [64,64,64,64,64,256,256]
  },
  "variable_bindings": [
    {"dim": 4, "intermediate_size": 512, "max_position_embeddings": 64},
    {"dim": 8, "intermediate_size": 512, "max_position_embeddings": 64},
    {"dim": 10, "intermediate_size": 512, "max_position_embeddings": 64},
    {"dim": 16, "intermediate_size": 512, "max_position_embeddings": 64},
    {"dim": 32, "intermediate_size": 512, "max_position_embeddings": 64},
    {"dim": 64, "intermediate_size": 1024, "max_position_embeddings": 256},
    {"dim": 128, "intermediate_size": 1024, "max_position_embeddings": 256}
  ]
}
```

这确保不同的 `dim` 值只与正确的 `intermediate_size` 与 `max_position_embeddings` 匹配，避免无效组合。



## 6. 脚本生成器（ScriptGenerator）

`ScriptGenerator` 用于生成批处理脚本（如 Shell 脚本），支持：

- 单命令模板
- 参数组合展开
- 多个命令块拼接成完整脚本

适合需要一键运行多个训练/测试任务的场景。



### 6.1 基本原理

1. **命令模板**：定义带占位符 `{var}` 的单条命令
2. **变量参数**：参数范围，自动展开为组合
3. **变量绑定**：限制参数关系
4. **多命令块**：多条命令按顺序拼接，构成完整脚本
5. **输出控制**：设置脚本输出路径与文件名



### 6.2 使用预设文件生成

运行命令：

```bash
uv run python -m scripts.tools.fast_generator script -f script_preset.json
```

其中 `script_preset.json` 定义多个命令块：

```json
{
  "name": "Training Config Generator",
  "description": "A generator preset for creating training script configurations.",
  "commands": [
    {
      "template": "#!/bin/bash",
      "variable_params": {},
      "variable_bindings": {}
    },
    {
      "template": "uv run python -m asset_embeddings.scripts.train.w2v --config configs/pretrained/AssetW2V/AssetW2V_d{dim}_pretrained.json --log model/pretrained/AssetW2V/d{dim}/train.log logs/train/pretrained/AssetW2V/d{dim}.log",
      "variable_params": { "dim": [4,8,10,16,32,64,128] },
      "variable_bindings": []
    },
    {
      "template": "uv run python -m asset_embeddings.scripts.train.rs --config configs/finetune/AssetRS/RS_{variant}/d{dim}/AssetRS_{variant}_d{dim}_{quarter}.json --log model/finetune/AssetRS/RS_{variant}/d{dim}/{quarter}/train.log logs/train/finetune/AssetRS/RS_{variant}/d{dim}/{quarter}.log",
      "variable_params": {
        "variant":["Binary","Ranks","Level0","LevelMin"],
        "dim":[4,8,10,16,32,64,128],
        "quarter":["2023Q2","2023Q3","2023Q4","2024Q1","2024Q2","2024Q3"]
      },
      "variable_bindings": []
    },
    {
      "template": "uv run python -m scripts.tools.smtp -t ccarzit@gmail.com -s 'Training Finished' -b 'All pretraining and finetuning model training has been completed.'",
      "variable_params": {},
      "variable_bindings": []
    }
  ],
  "output_dir": "scripts/run",
  "filename": "train.sh"
}
```



### 6.3 生成结果

程序会在 `scripts/run/train.sh`（预设的 `output_dir` + `filename`）生成一份完整脚本。

部分展开的命令（示例）：

```bash
#!/bin/bash
uv run python -m asset_embeddings.scripts.train.w2v --config configs/pretrained/AssetW2V/AssetW2V_d4_pretrained.json --log model/pretrained/AssetW2V/d4/train.log logs/train/pretrained/AssetW2V/d4.log
uv run python -m asset_embeddings.scripts.train.w2v --config configs/pretrained/AssetW2V/AssetW2V_d8_pretrained.json --log model/pretrained/AssetW2V/d8/train.log logs/train/pretrained/AssetW2V/d8.log
...

uv run python -m asset_embeddings.scripts.train.rs --config configs/finetune/AssetRS/RS_Binary/d4/AssetRS_Binary_d4_2023Q2.json --log model/finetune/AssetRS/RS_Binary/d4/2023Q2/train.log logs/train/finetune/AssetRS/RS_Binary/d4/2023Q2.log
uv run python -m asset_embeddings.scripts.train.rs --config configs/finetune/AssetRS/RS_Binary/d4/AssetRS_Binary_d4_2023Q3.json --log model/finetune/AssetRS/RS_Binary/d4/2023Q3/train.log logs/train/finetune/AssetRS/RS_Binary/d4/2023Q3.log
...
uv run python -m scripts.tools.smtp -t ccarzit@gmail.com -s 'Training Finished' -b 'All pretraining and finetuning model training has been completed.'
```



## 7. 进阶特性

除了基本的参数组合与绑定，程序还提供**绑定组识别、校验机制、函数扩展**等进阶特性，以确保生成结果正确而灵活。

### 7.1 多绑定组机制

有时不同参数集彼此绑定、互不影响。程序会自动识别**独立的绑定组**并生成笛卡尔积。

### 示例

```json
{
  "variable_params": {
    "optimizer": ["adam", "sgd"],
    "lr": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128]
  },
  "variable_bindings": [
    {"optimizer": "adam", "lr": 0.001},
    {"optimizer": "adam", "lr": 0.01},
    {"optimizer": "sgd", "lr": 0.1}
  ]
}
```

这里有两个绑定组：

- 组 1：`{"optimizer": "adam", "lr": ...}`
- 组 2：`{"optimizer": "sgd", "lr": 0.1}`

生成组合时，每个绑定组还会与**自由参数 `batch_size`** 做笛卡尔积。

结果：

```
(adam, 0.001, 32)
(adam, 0.001, 64)
(adam, 0.001, 128)
(adam, 0.01, 32)
...
(sgd, 0.1, 128)
```



### 7.2 参数校验机制

程序在生成前执行校验，帮助用户排查错误。

1. **函数调用校验**

   - 检查语法是否正确

   - 检查函数是否已注册

   - 示例：

     ```json
     {"quarter": "@generate_time_quaters(2020,1,2021,4)"}
     ```

     将报错：`Function 'generate_time_quaters' not found. Available functions: generate_time_quarters, ...`

2. **绑定校验**

   - 检查绑定的参数是否在 `variable_params` 中声明

   - 检查绑定值是否属于有效取值

   - 示例：

     ```json
     {
       "variable_params": {"dim":[4,8]},
       "variable_bindings":[{"dim":16}]
     }
     ```

     将报错：`Value 16 for 'dim' not in allowed values [4,8]`



### 7.3 自定义函数扩展

用户可注册自己的便捷函数，让生成器更灵活。

#### 示例：生成指数衰减学习率

```python
from scripts.tools.fast_generator import FunctionRegistry

def exp_decay_lr(start, decay, steps):
    return [start * (decay ** i) for i in range(steps)]

registry = FunctionRegistry()
registry.register("exp_decay_lr", exp_decay_lr)

print(registry.call("exp_decay_lr", 0.1, 0.9, 5))
# Output: [0.1, 0.09000000000000001, 0.08100000000000002, 0.07290000000000002, 0.06561000000000002]
```

然后即可在 JSON 模板中使用：

```json
{"lr": "@exp_decay_lr(0.1,0.9,5)"}
```

生成：

```
[0.1, 0.09, 0.081, 0.0729, 0.06561]
```



### 7.4 动态输出路径与文件名

占位符支持格式化语法，例如：

- `{epoch:03d}` → 补足三位
- `{lr:.4f}` → 格式化为四位小数
- `{size:>5s}` → 右对齐，宽度 5

示例：

```bash
-o configs/{size} -n config_lr{lr:.4f}.json
```

结果：

```
configs/tiny/config_lr0.0010.json
configs/base/config_lr0.0100.json
```



### 7.5 错误处理与调试

- 所有校验错误都会打印详细信息
- 若某个组合生成失败（如非法路径），程序会跳过该组合并继续
- 可通过 `validate_bindings()` 与 `validate_function_calls()` 手动调试

示例：

```python
gen = ConfigGenerator()
gen.set_variable_params(dim=[4,8], lr=[0.001])
gen.set_variable_bindings([{"dim":16}])
print(gen.validate_bindings())
# Output:
# Error in binding 0: Value 16 for 'dim' not in allowed values [4, 8]
# False
```



## 8. 开发者接口与 API 文档

除了命令行工具，本程序还可作为 **Python 库**使用。主要类包括：

- `BaseTemplateGenerator`（抽象基类）
- `ConfigGenerator`（JSON 配置文件生成器）
- `CommandGenerator`（命令字符串生成器）
- `ScriptGenerator`（脚本生成器）
- `FunctionRegistry`（函数注册与调用）



### 8.1 `BaseTemplateGenerator`

抽象基类，定义所有生成器的通用逻辑。

**主要方法**

```python
set_template(template: dict | str)
set_variable_params(**kwargs)
set_variable_bindings(bindings: list[dict])
set_output_dir(path: str)
set_filename(filename_template: str)
replace_template_vars(template, variables: dict) -> Any
validate_bindings() -> bool
validate_function_calls() -> bool
```

**示例**

```python
from scripts.tools.fast_generator import ConfigGenerator

gen = ConfigGenerator()
gen.set_template({"lr": "{lr}", "batch_size": "{bs}"})
gen.set_variable_params(lr=[0.001, 0.01], bs=[32, 64])
gen.set_output_dir("configs")
gen.set_filename("config_lr{lr}_bs{bs}.json")
gen.generate()
```

输出文件：

```
configs/config_lr0.001_bs32.json
configs/config_lr0.001_bs64.json
configs/config_lr0.01_bs32.json
configs/config_lr0.01_bs64.json
```



### 8.2 `ConfigGenerator`

用于生成 JSON 配置文件。

**新增方法**

```python
set_config_template(config: dict)
generate() -> list[str]
```

**示例**

```python
template = {
    "model": "AssetW2V",
    "dim": "{dim}",
    "epochs": 10
}

gen = ConfigGenerator()
gen.set_config_template(template)
gen.set_variable_params(dim=[4, 8, 16])
gen.set_output_dir("configs")
gen.set_filename("AssetW2V_d{dim}.json")
files = gen.generate()

print(files)
# ['configs/AssetW2V_d4.json', 'configs/AssetW2V_d8.json', 'configs/AssetW2V_d16.json']
```



### 8.3 `CommandGenerator`

用于生成命令字符串。

**主要方法**

```python
CommandGenerator(template: str)          # 命令模板是构造函数参数
set_variable_params(**kwargs)
set_variable_bindings(bindings: list[dict])
generate() -> list[str]
```

**示例**

```python
from scripts.tools.fast_generator import CommandGenerator

gen = CommandGenerator("python train.py --lr {lr} --bs {bs}")
gen.set_variable_params(lr=[0.001, 0.01], bs=[32, 64])
commands = gen.generate()

print(commands)
# ['python train.py --lr 0.001 --bs 32',
#  'python train.py --lr 0.001 --bs 64',
#  'python train.py --lr 0.01 --bs 32',
#  'python train.py --lr 0.01 --bs 64']
```



### 8.4 `ScriptGenerator`

用于生成批处理脚本文件。

**主要方法**

```python
add_command(template: str, bindings: list[dict] | None = None, **params)
add_commands(command_dicts: list[dict])
generate() -> list[str]
```

**示例**

```python
from scripts.tools.fast_generator import ScriptGenerator

script_gen = ScriptGenerator()
# 每次 add_command() 调用是一个命令块（各有自己的 params/bindings）。
script_gen.add_command("python train.py --lr {lr} --bs {bs}", lr=[0.001, 0.01], bs=[32])
script_gen.set_output_dir("scripts")
script_gen.set_filename("train.sh")

files = script_gen.generate()
print(files)
# ['scripts/train.sh']
```

生成的 `train.sh`：

```bash
python train.py --lr 0.001 --bs 32
python train.py --lr 0.01 --bs 32
```



### 8.5 `FunctionRegistry`

用于注册与调用自定义函数。

**主要方法**

```python
register(name: str, func: Callable)
call(name: str, *args, **kwargs)
```

**示例**

```python
from scripts.tools.fast_generator import FunctionRegistry

def generate_quarters(start_year, end_year):
    return [f"{y}Q{q}" for y in range(start_year, end_year+1) for q in range(1,5)]

registry = FunctionRegistry()
registry.register("generate_quarters", generate_quarters)

print(registry.call("generate_quarters", 2023, 2024))
# ['2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1', '2024Q2', '2024Q3', '2024Q4']
```

在模板中使用：

```json
{"quarter": "@generate_quarters(2023,2024)"}
```
