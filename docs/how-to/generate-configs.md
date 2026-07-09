# Generate configs

> Automated Batch Configuration and Script Generation Tool

## 1. Introduction

### 1.1 Background and Motivation

In machine learning, data analysis, or automated experiments, there is a frequent need to **batch generate configuration files** and **batch run scripts**:

- **Hyperparameter Tuning**: When training models, experiments need to be repeated under different hyperparameter combinations (e.g., `dim=128, lr=0.001` vs `dim=256, lr=0.01`)
- **Time Series Experiments**: Need to generate JSON files with different time windows and dataset configurations to drive downstream programs
- **Batch Task Submission**: After experiments are completed, need to generate run scripts with one click for unified submission to clusters or servers

These tasks are **repetitive and mechanical**:

- Manually maintaining 10 parameter combinations may require copying and pasting 10 configuration files or scripts
- Once parameters increase to 3-4 dimensions, the number of combinations grows exponentially, making manual maintenance nearly impossible

### 1.2 Design Goals

**Core Functionality**:

- Through **templates + variable parameters**, automatically batch generate JSON configuration files and script files
- Support **parameter Cartesian product**, **variable bindings**, **function-generated time/number series**, and other mechanisms for flexible combination
- Maintain **command-line friendliness** (convenient for direct invocation) and **class interface extensibility** (convenient for developer customization)

**Typical Use Cases**:

1. **Machine Learning/Deep Learning Experiments**: Batch generate configuration files and run scripts for different hyperparameter combinations
2. **Data Processing Pipelines**: Batch generate task configurations for different dates, quarters, and parameter settings
3. **Automated Batch Processing**: Quickly generate thousands of configurations/scripts to avoid repetitive work



## 2. Quick Start

### 2.1 Example 1: Generating Configuration Files

Suppose you have a model configuration template `config_template.json`:

```json
{
  "model": "resnet",
  "dim": "{dim}",
  "lr": "{lr}"
}
```

We want to generate configuration files for different `dim` and `lr` combinations. Simply run:

```bash
uv run python -m scripts.tools.fast_generator config \
    -t config_template.json \
    -v '{"dim":[128,256], "lr":[0.001,0.01]}' \
    -o configs/exp_{dim} \
    -n config_{lr}.json
```

### Result

Directory structure:

```
configs/
 ├─ exp_128/
 │   ├─ config_0.001.json
 │   └─ config_0.01.json
 └─ exp_256/
     ├─ config_0.001.json
     └─ config_0.01.json
```

Content of one file (`configs/exp_128/config_0.001.json`):

```json
{
  "model": "resnet",
  "dim": 128,
  "lr": 0.001
}
```



### 2.2 Example 2: Using Functions to Generate Parameters

No need to manually write out all parameter values; you can use built-in functions to auto-generate:

```bash
uv run python -m scripts.tools.fast_generator config \
    -t config_template.json \
    -v '{"quarter":"@generate_time_quarters(2020,1,2021,4)"}'
```

This will generate 8 configuration files with `quarter` values of `2020Q1 … 2021Q4`.



### 2.3 Example 3: Generating Script Files

Suppose you have a script preset file `script_preset.json`:

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

Run:

```bash
uv run python -m scripts.tools.fast_generator script -f script_preset.json
```

Generates `scripts/train.sh`:

```bash
python train.py --dim 128 --lr 0.001
python train.py --dim 128 --lr 0.01
python train.py --dim 256 --lr 0.001
python train.py --dim 256 --lr 0.01
```

This allows direct batch submission of training tasks. (Blank lines are inserted only *between* command blocks, not between the variants of one block.)



## 3. Core Concepts

The core idea of the program is: **Template + Variable Parameters + Variable Bindings + Output Control**

> **Core Elements Overview**
> - **Template**: Defines the basic structure of generation, using placeholders to mark variable parts
> - **Variable Params**: Specifies the value ranges of parameters, automatically expanded into Cartesian product
> - **Variable Bindings**: Constrains relationships between parameters to avoid invalid combinations
> - **Output Control**: Uses placeholders to customize output paths and filenames
> - **Function Calls**: Dynamically generates parameter values through built-in or custom functions



### 3.1 Template

A template is the "basic skeleton" of generation. It can be:

- JSON object (for configuration files)
- String (for commands/scripts)

In templates, parameter positions can be represented by **placeholders**:

- `{var}` → Replace with parameter value
- `{var:format}` → Replace parameter value according to format (supports Python format syntax)

Example template:

```json
{
  "model": "resnet",
  "dim": "{dim}",
  "lr": "{lr:.4f}"
}
```

If the parameter combination is `{"dim":128, "lr":0.001}`, the generated result is:

```json
{
  "model": "resnet",
  "dim": 128,
  "lr": 0.0010
}
```



### 3.2 Variable Parameters

Variable parameters define which values need to be **enumerated and combined**.

- In command line, pass JSON string with `-v`
- In preset files, specify with `variable_params`

Example:

```json
{"dim":[64,128], "lr":[0.001,0.01]}
```

Will automatically expand to 4 combinations:

```
(dim=64, lr=0.001)
(dim=64, lr=0.01)
(dim=128, lr=0.001)
(dim=128, lr=0.01)
```

Generating corresponding 4 configurations or commands.



### 3.3 Variable Bindings

Some parameters are not "freely combined" but "must be bound".
For example:

- **Wrong approach**: `size ∈ {tiny, base}, dim ∈ {64,128}` → 4 combinations, but `tiny` might only correspond to `64`
- **Solution**: Use **variable bindings** to explicitly constrain relationships:

```json
[
  {"size": "tiny", "dim": 64},
  {"size": "base", "dim": 128}
]
```

This will only generate 2 valid combinations.

Note: Bindings and free parameters can be used simultaneously.

- Bound relationship parts remain unchanged
- Other free parameters still participate in combination

Example:

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

Generates 4 combinations:

```
(size=tiny, dim=64, epochs=10)
(size=tiny, dim=64, epochs=20)
(size=base, dim=128, epochs=10)
(size=base, dim=128, epochs=20)
```



### 3.4 Output Control

Similarly, output paths and filenames can be customized with placeholder variables.

Example:

```bash
-o configs/{size} -n config_{dim}.json
```

If the parameter combination is `{"size":"tiny","dim":64}`, the generated file is:

```
configs/tiny/config_64.json
```

Another combination `{"size":"base","dim":128}` generates:

```
configs/base/config_128.json
```



### 3.5 Function Calls

In addition to manually listing all values, variable parameter lists can also be dynamically generated by registered functions. Use `@` in strings to specify function calls and arguments.

Syntax:

```
"@function_name(arg1, arg2, ...)"
```

Example:

```json
{"quarter": "@generate_time_quarters(2020,1,2021,4)"}
```

→ Generates `["2020Q1","2020Q2","2020Q3","2020Q4","2021Q1","2021Q2","2021Q3","2021Q4"]`

When combined with bindings or other parameters, it will automatically expand combinations.



## 4. Built-in Convenience Functions

In variable parameters (`variable_params`), you can dynamically generate parameter values through **function calls**.
Syntax format:

```
"@function_name(arg1, arg2, key=value, ...)"
```

The result returned by the function call will be treated as a set of parameter values and participate in combination generation.



### 4.1 Time Series Related Functions

**(1) `generate_time_quarters(start_year, start_quarter, end_year, end_quarter, format="%YQ%Q")`**

Generate quarterly sequence.

Example:

```json
{"quarter": "@generate_time_quarters(2020,1,2021,4)"}
```

Result:

```
["2020Q1","2020Q2","2020Q3","2020Q4","2021Q1","2021Q2","2021Q3","2021Q4"]
```



**(2) `generate_time_months(start_year, start_month, end_year, end_month, format="%Y-%m")`**

Generate monthly sequence.

Example:

```json
{"month": "@generate_time_months(2020,1,2020,6)"}
```

Result:

```
["2020-01","2020-02","2020-03","2020-04","2020-05","2020-06"]
```



**(3) `generate_date_range(start_date, end_date, freq="D", format="%Y-%m-%d")`**

Generate date range with different frequencies:

- `D` → Day
- `W` → Week
- `M` → Month
- `Q` → Quarter
- `Y` → Year

Example:

```json
{"date": "@generate_date_range('2020-01-01','2020-01-10',freq='D')"}
```

Result:

```
["2020-01-01","2020-01-02",...,"2020-01-10"]
```



### 4.2 Number Series Generation Functions

**(1) `range(start, stop, step=1)`**

Similar to Python's built-in `range`, generates integer sequence.

```json
{"x": "@range(0,5)"}
```

Result:

```
[0,1,2,3,4]
```



**(2) `linspace(start, stop, num)`**

Generate evenly spaced sequence (inclusive of endpoints).

```json
{"x": "@linspace(0,1,5)"}
```

Result:

```
[0.0,0.25,0.5,0.75,1.0]
```



**(3) `logspace(start, stop, num)`**

Generate logarithmic scale sequence with base 10.

```json
{"x": "@logspace(0,2,3)"}
```

Result:

```
[1.0, 10.0, 100.0]
```



**(4) `geomspace(start, stop, num)`**

Generate geometric progression from `start` to `stop`.

```json
{"x": "@geomspace(1,100,3)"}
```

Result:

```
[1.0, 10.0, 100.0]
```



### 4.3 List Operation Functions

**(1) `repeat(value, times)`**

Repeat a value multiple times.

```json
{"flag": "@repeat('debug',3)"}
```

Result:

```
["debug","debug","debug"]
```



**(2) `zip_lists(list1, list2, ...)`**

Bind multiple lists together by index position.

```json
{"pair": "@zip_lists([1,2],[10,20])"}
```

Result:

```
[[1,10],[2,20]]
```



**(3) `generate_cross_product(list1, list2, ...)`**

Generate Cartesian product of multiple lists.

```json
{"pair": "@generate_cross_product([1,2],[3,4])"}
```

Result:

```
[[1,3],[1,4],[2,3],[2,4]]
```



### 4.4 Custom Function Extension

Developers can register new functions via `FunctionRegistry.register(name, func)`.

Example:

```python
from scripts.tools.fast_generator import FunctionRegistry

def square_numbers(n):
    return [i*i for i in range(1,n+1)]

registry = FunctionRegistry()
registry.register("squares", square_numbers)

print(registry.call("squares", 5))
# Output: [1,4,9,16,25]
```

Then you can use it in template parameters:

```json
{"value": "@squares(5)"}
```



## 5. Configuration File Generator (ConfigGenerator)

`ConfigGenerator` is used to batch generate JSON configuration files based on templates and parameter combinations.
It is the most commonly used generator, suitable for batch experiments and task automation scenarios.



### 5.1 Basic Principle

1. **Template**: A JSON configuration file with placeholders
2. **Variable Parameters**: Parameter ranges to enumerate
3. **Variable Bindings**: Constraint relationships between parameters
4. **Output Control**: Output directory and filename template for generated configuration files

The program will automatically expand all parameter combinations and replace `{var}` placeholders with actual values, finally outputting a series of JSON configuration files.



### 5.2 Generating Using Preset Files

You can load a JSON file via `--from-preset`, which defines all generation rules.

Command example:

```bash
uv run python -m scripts.tools.fast_generator config -f configs_preset.json
```

The `configs_preset.json` here is a task list that can contain one or more generation tasks. For example:



#### Example 1: AssetW2V Finetune Task

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

After running, configuration files will be generated:

```
configs/finetune/AssetW2V/
 ├─ d4/AssetW2V_d4_2023Q2.json
 ├─ d4/AssetW2V_d4_2023Q3.json
 ...
 ├─ d128/AssetW2V_d128_2024Q3.json
```

Where `AssetW2V_d4_2023Q2.json` content is similar to:

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



#### Example 2: AssetBERT Finetune Task (with Variable Bindings)

In the `AssetBERT` configuration, the three parameters `dim`, `intermediate_size`, and `max_position_embeddings` must correspond.
This requires using **variable bindings (variable_bindings)**.

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

This ensures that different `dim` values only match with correct `intermediate_size` and `max_position_embeddings`, avoiding invalid combinations.



## 6. Script Generator (ScriptGenerator)

`ScriptGenerator` is used to generate batch processing scripts (such as Shell scripts), supporting:

- Single command template
- Parameter combination expansion
- Multiple command blocks concatenated into complete script

Suitable for scenarios requiring one-click running of multiple training/testing tasks.



### 6.1 Basic Principle

1. **Command Template**: Defines a single command with placeholders `{var}`
2. **Variable Parameters**: Parameter ranges, automatically expanded into combinations
3. **Variable Bindings**: Limit parameter relationships
4. **Multiple Command Blocks**: Multiple commands concatenated in sequence to form complete script
5. **Output Control**: Set script output path and filename



### 6.2 Generating Using Preset Files

Run command:

```bash
uv run python -m scripts.tools.fast_generator script -f script_preset.json
```

Where `script_preset.json` defines multiple command blocks:

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



### 6.3 Generation Result

The program will generate a complete script at `scripts/run/train.sh` (the preset's `output_dir` + `filename`).

Partially expanded commands (example):

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



## 7. Advanced Features

In addition to basic parameter combination and binding, the program also provides **binding group identification, validation mechanisms, function extension** and other advanced features to ensure correct and flexible generation results.

### 7.1 Multiple Binding Groups Mechanism

Sometimes different parameter sets bind to each other without affecting each other.
The program will automatically identify **independent binding groups** and generate Cartesian products.

### Example

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

There are two binding groups here:

- Group 1: `{"optimizer": "adam", "lr": ...}`
- Group 2: `{"optimizer": "sgd", "lr": 0.1}`

When generating combinations, each binding group will also do Cartesian product with **free parameter `batch_size`**.

Result:

```
(adam, 0.001, 32)
(adam, 0.001, 64)
(adam, 0.001, 128)
(adam, 0.01, 32)
...
(sgd, 0.1, 128)
```



### 7.2 Parameter Validation Mechanism

The program executes validation before generation to help users troubleshoot errors.

1. **Function Call Validation**

   - Check if syntax is correct

   - Check if function is registered

   - Example:

     ```json
     {"quarter": "@generate_time_quaters(2020,1,2021,4)"}
     ```

     Will report error: `Function 'generate_time_quaters' not found. Available functions: generate_time_quarters, ...`

2. **Binding Validation**

   - Check if bound parameters are declared in `variable_params`

   - Check if binding values belong to valid values

   - Example:

     ```json
     {
       "variable_params": {"dim":[4,8]},
       "variable_bindings":[{"dim":16}]
     }
     ```

     Will report error: `Value 16 for 'dim' not in allowed values [4,8]`



### 7.3 Custom Function Extension

Users can register their own convenience functions to make the generator more flexible.

#### Example: Generate Exponentially Decaying Learning Rate

```python
from scripts.tools.fast_generator import FunctionRegistry

def exp_decay_lr(start, decay, steps):
    return [start * (decay ** i) for i in range(steps)]

registry = FunctionRegistry()
registry.register("exp_decay_lr", exp_decay_lr)

print(registry.call("exp_decay_lr", 0.1, 0.9, 5))
# Output: [0.1, 0.09000000000000001, 0.08100000000000002, 0.07290000000000002, 0.06561000000000002]
```

Then you can use it in JSON template:

```json
{"lr": "@exp_decay_lr(0.1,0.9,5)"}
```

Generates:

```
[0.1, 0.09, 0.081, 0.0729, 0.06561]
```



### 7.4 Dynamic Output Paths and Filenames

Placeholders support formatting syntax, such as:

- `{epoch:03d}` → Pad to three digits
- `{lr:.4f}` → Format to four decimal places
- `{size:>5s}` → Right align, width 5

Example:

```bash
-o configs/{size} -n config_lr{lr:.4f}.json
```

Result:

```
configs/tiny/config_lr0.0010.json
configs/base/config_lr0.0100.json
```



### 7.5 Error Handling and Debugging

- All validation errors will print detailed information
- If a combination generation fails (such as illegal path), the program will skip that combination and continue
- You can manually debug via `validate_bindings()` and `validate_function_calls()`

Example:

```python
gen = ConfigGenerator()
gen.set_variable_params(dim=[4,8], lr=[0.001])
gen.set_variable_bindings([{"dim":16}])
print(gen.validate_bindings())
# Output:
# Error in binding 0: Value 16 for 'dim' not in allowed values [4, 8]
# False
```



## 8. Developer Interface and API Documentation

In addition to command-line tools, this program can also be used as a **Python library**.
Main classes include:

- `BaseTemplateGenerator` (abstract base class)
- `ConfigGenerator` (JSON configuration file generator)
- `CommandGenerator` (command string generator)
- `ScriptGenerator` (script generator)
- `FunctionRegistry` (function registration and calling)



### 8.1 `BaseTemplateGenerator`

Abstract base class that defines common logic for all generators.

**Main Methods**

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

**Example**

```python
from scripts.tools.fast_generator import ConfigGenerator

gen = ConfigGenerator()
gen.set_template({"lr": "{lr}", "batch_size": "{bs}"})
gen.set_variable_params(lr=[0.001, 0.01], bs=[32, 64])
gen.set_output_dir("configs")
gen.set_filename("config_lr{lr}_bs{bs}.json")
gen.generate()
```

Output files:

```
configs/config_lr0.001_bs32.json
configs/config_lr0.001_bs64.json
configs/config_lr0.01_bs32.json
configs/config_lr0.01_bs64.json
```



### 8.2 `ConfigGenerator`

Used to generate JSON configuration files.

**New Methods**

```python
set_config_template(config: dict)
generate() -> list[str]
```

**Example**

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

Used to generate command strings.

**Main Methods**

```python
CommandGenerator(template: str)          # the command template is a constructor arg
set_variable_params(**kwargs)
set_variable_bindings(bindings: list[dict])
generate() -> list[str]
```

**Example**

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

Used to generate batch processing script files.

**Main Methods**

```python
add_command(template: str, bindings: list[dict] | None = None, **params)
add_commands(command_dicts: list[dict])
generate() -> list[str]
```

**Example**

```python
from scripts.tools.fast_generator import ScriptGenerator

script_gen = ScriptGenerator()
# Each add_command() call is one command block (its own params/bindings).
script_gen.add_command("python train.py --lr {lr} --bs {bs}", lr=[0.001, 0.01], bs=[32])
script_gen.set_output_dir("scripts")
script_gen.set_filename("train.sh")

files = script_gen.generate()
print(files)
# ['scripts/train.sh']
```

Generated `train.sh`:

```bash
python train.py --lr 0.001 --bs 32
python train.py --lr 0.01 --bs 32
```



### 8.5 `FunctionRegistry`

Used to register and call custom functions.

**Main Methods**

```python
register(name: str, func: Callable)
call(name: str, *args, **kwargs)
```

**Example**

```python
from scripts.tools.fast_generator import FunctionRegistry

def generate_quarters(start_year, end_year):
    return [f"{y}Q{q}" for y in range(start_year, end_year+1) for q in range(1,5)]

registry = FunctionRegistry()
registry.register("generate_quarters", generate_quarters)

print(registry.call("generate_quarters", 2023, 2024))
# ['2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1', '2024Q2', '2024Q3', '2024Q4']
```

Use in template:

```json
{"quarter": "@generate_quarters(2023,2024)"}
```
