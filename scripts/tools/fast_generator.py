import os
import re
import ast
import json
import argparse
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable


# 便捷函数库
def generate_time_quarters(
    start_year: int, start_quarter: int, end_year: int, end_quarter: int, format: str = "%YQ%Q"
) -> List[str]:
    """生成时间季度列表，如2020Q1到2024Q4"""
    quarters = []
    for year in range(start_year, end_year + 1):
        start_q = start_quarter if year == start_year else 1
        end_q = end_quarter if year == end_year else 4
        for quarter in range(start_q, end_q + 1):
            quarters.append(format.replace("%Y", str(year)).replace("%Q", str(quarter)))
    return quarters


def generate_time_months(
    start_year: int, start_month: int, end_year: int, end_month: int, format: str = "%Y-%m"
) -> List[str]:
    """生成时间月份列表，如2020-01到2024-12"""
    months = []
    for year in range(start_year, end_year + 1):
        start_m = start_month if year == start_year else 1
        end_m = end_month if year == end_year else 12
        for month in range(start_m, end_m + 1):
            months.append(format.replace("%Y", str(year)).replace("%m", f"{month:02d}"))
    return months


def generate_date_range(start_date: str, end_date: str, freq: str = "D", format: str = "%Y-%m-%d") -> List[str]:
    """生成日期范围列表，支持不同频率（D=日, W=周, M=月, Q=季度, Y=年）"""
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start

    while current <= end:
        dates.append(current.strftime(format))

        if freq == "D":
            current += timedelta(days=1)
        elif freq == "W":
            current += timedelta(weeks=1)
        elif freq == "M":
            # 月份增加
            month = current.month
            year = current.year
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1
            try:
                current = current.replace(year=year, month=month)
            except ValueError:
                # 处理月末日期问题
                current = current.replace(year=year, month=month, day=1)
        elif freq == "Q":
            # 季度增加（3个月）
            for _ in range(3):
                month = current.month
                year = current.year
                if month == 12:
                    month = 1
                    year += 1
                else:
                    month += 1
                try:
                    current = current.replace(year=year, month=month)
                except ValueError:
                    current = current.replace(year=year, month=month, day=1)
        elif freq == "Y":
            current = current.replace(year=current.year + 1)

    return dates


def generate_cross_product(*lists) -> List[List]:
    """生成多个列表的笛卡尔积"""
    from itertools import product

    return [list(combo) for combo in product(*lists)]


class FunctionRegistry:
    """便捷函数注册器"""

    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._register_builtin_functions()

    def _register_builtin_functions(self):
        """注册内置便捷函数"""
        # 时间相关函数
        self.register("generate_time_quarters", generate_time_quarters)
        self.register("generate_time_months", generate_time_months)
        self.register("generate_date_range", generate_date_range)

        # 数值序列生成函数
        self.register("range", lambda start, stop, step=1: list(range(start, stop, step)))
        self.register("linspace", lambda start, stop, num: [start + i * (stop - start) / (num - 1) for i in range(num)])
        self.register(
            "logspace", lambda start, stop, num: [10 ** (start + i * (stop - start) / (num - 1)) for i in range(num)]
        )
        self.register(
            "geomspace", lambda start, stop, num: [start * (stop / start) ** (i / (num - 1)) for i in range(num)]
        )

        # 列表操作函数
        self.register("repeat", lambda value, times: [value] * times)
        self.register("zip_lists", lambda *lists: [list(item) for item in zip(*lists)])
        self.register("generate_cross_product", generate_cross_product)

    def register(self, name: str, func: Callable):
        """注册新的便捷函数"""
        self._functions[name] = func

    def call(self, name: str, *args, **kwargs) -> Any:
        """调用注册的函数"""
        if name not in self._functions:
            raise ValueError(f"Function '{name}' not found in registry")
        return self._functions[name](*args, **kwargs)

    def list_functions(self) -> List[str]:
        """列出所有可用函数"""
        return list(self._functions.keys())


class BaseTemplateGenerator(ABC):
    """模板生成器基类"""

    def __init__(self):
        self.base_template: Any = {}
        self.variable_params: Dict[str, Any] = {}
        self.variable_bindings: List[Dict[str, Any]] = []
        self.output_dir: str = "output"
        self.filename_template: str = "file"
        self.function_registry = FunctionRegistry()

    def set_template(self, template: Any):
        """设置基础模板"""
        self.base_template = self._copy_template(template)

    @abstractmethod
    def _copy_template(self, template: Any) -> Any:
        """复制模板的抽象方法"""
        pass

    def _parse_function_call(self, value: str) -> Any:
        """
        解析函数调用字符串，支持 @function_name(arg1, arg2, key=value) 格式
        例如: @generate_time_quarters(2020, 1, 2024, 4, format="%YQ%Q")
        """
        if not isinstance(value, str) or not value.startswith("@"):
            return value

        # 移除 @ 前缀
        call_str = value[1:]

        # 解析函数名和参数
        match = re.match(r"(\w+)\((.*)\)$", call_str)
        if not match:
            raise ValueError(f"Invalid function call syntax: {value}")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 解析参数
        args = []
        kwargs = {}

        if args_str:
            # 简单的参数解析（支持数字、字符串、布尔值）

            try:
                # 构造一个临时函数调用来解析参数
                temp_call = f"temp_func({args_str})"
                parsed = ast.parse(temp_call, mode="eval")
                call_node = parsed.body

                # ast.literal_eval handles scalars (int/float/str/bool/None) and
                # container literals (list/tuple/dict) — the latter lets functions
                # like zip_lists / generate_cross_product take list arguments via
                # the @-string syntax. Non-literals (names, calls) raise ValueError.
                for arg in call_node.args:
                    try:
                        args.append(ast.literal_eval(arg))
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Unsupported argument type: {ast.dump(arg)}")

                for keyword in call_node.keywords:
                    try:
                        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Unsupported keyword argument type: {ast.dump(keyword.value)}")

            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse function arguments in '{value}': {e}")

        # 调用函数
        try:
            result = self.function_registry.call(func_name, *args, **kwargs)
            return result
        except Exception as e:
            raise ValueError(f"Error calling function '{func_name}': {e}")

    def _process_variable_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理可变参数，解析其中的函数调用"""
        processed = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("@"):
                result = self._parse_function_call(value)
                # 确保结果是列表（用于参数组合）
                if not isinstance(result, list):
                    result = [result]
                processed[key] = result
            elif isinstance(value, list):
                # 处理列表中的函数调用
                processed_list = []
                for item in value:
                    if isinstance(item, str) and item.startswith("@"):
                        result = self._parse_function_call(item)
                        if isinstance(result, list):
                            processed_list.extend(result)
                        else:
                            processed_list.append(result)
                    else:
                        processed_list.append(item)
                processed[key] = processed_list
            else:
                processed[key] = value
        return processed

    def set_variable_params(self, **params):
        """
        设置可变参数，支持函数调用
        例如: set_variable_params(
            embedding_dim=[4,8,16],
            epochs=[10,20],
            quarter="@generate_time_quarters(2020, 1, 2024, 4)"
        )
        """
        self.variable_params = self._process_variable_params(params)

    def add_variable_binding(self, binding_dict: Dict[str, Any]):
        """
        添加变量绑定关系
        例如: add_variable_binding({"size": "tiny", "dim": 4, "lr": 0.001})
        """
        self.variable_bindings.append(binding_dict)

    def add_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """
        添加多个变量绑定关系
        例如: add_variable_bindings([{"size": "tiny", "dim": 4, "lr": 0.001}, {"size": "base", "dim": 8, "lr": 0.01}])
        """
        self.variable_bindings.extend(binding_dicts)

    def clear_variable_bindings(self):
        """清空所有变量绑定关系"""
        self.variable_bindings = []

    def set_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """设置变量绑定关系"""
        self.variable_bindings = binding_dicts

    def set_output_dir(self, dir_template: str):
        """设置输出目录，支持{var_name}格式"""
        self.output_dir = dir_template

    def set_filename(self, filename_template: str):
        """设置文件名模板，支持{var_name}格式"""
        self.filename_template = filename_template

    def _format_value(self, value: Any, format_spec: str) -> str:
        """
        格式化值，支持 Python format 字符串语法
        例如: {value:.2f}, {value:04d}, {value:>10s}
        """
        if format_spec:
            if isinstance(value, (int, float)):
                return f"{value:{format_spec}}"
            elif isinstance(value, str):
                return f"{value:{format_spec}}"
            else:
                return f"{str(value):{format_spec}}"
        return str(value)

    def replace_template_vars(self, value: Any, replacements: Dict[str, Any]) -> Any:
        """
        替换模板变量，支持 {var_name:format} 格式
        例如: {dim}, {lr:.4f}, {epoch:03d}
        """
        if isinstance(value, str):
            # 匹配所有 {变量名:格式} 或 {变量名} 的模式
            pattern = r"\{([^{}:]+)(?::([^{}]*))?\}"
            matches = list(re.finditer(pattern, value))

            if len(matches) == 1 and matches[0].group(0) == value:
                # 如果整个字符串就是一个占位符，返回原始类型的替换值
                var_name = matches[0].group(1)
                format_spec = matches[0].group(2) or ""

                if var_name in replacements:
                    if format_spec:
                        return self._format_value(replacements[var_name], format_spec)
                    else:
                        return replacements[var_name]
                return value

            # 否则做字符串替换
            result = value
            for match in reversed(matches):  # 从后往前替换，避免位置偏移
                var_name = match.group(1)
                format_spec = match.group(2) or ""

                if var_name in replacements:
                    formatted_value = self._format_value(replacements[var_name], format_spec)
                    result = result[: match.start()] + formatted_value + result[match.end() :]

            return result

        elif isinstance(value, dict):
            return {k: self.replace_template_vars(v, replacements) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.replace_template_vars(item, replacements) for item in value]
        else:
            return value

    def _identify_binding_groups(self) -> List[List[Dict[str, Any]]]:
        """
        识别独立的绑定组
        返回: 绑定组列表，每个绑定组包含共享相同键的绑定
        """
        if not self.variable_bindings:
            return []

        binding_groups = []
        used_bindings = set()

        for i, binding in enumerate(self.variable_bindings):
            if i in used_bindings:
                continue

            # 创建新的绑定组
            group = [binding]
            used_bindings.add(i)
            binding_keys = set(binding.keys())

            # 查找具有相同键集合的其他绑定
            for j, other_binding in enumerate(self.variable_bindings):
                if j > i and j not in used_bindings:
                    if set(other_binding.keys()) == binding_keys:
                        group.append(other_binding)
                        used_bindings.add(j)

            binding_groups.append(group)

        return binding_groups

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """获取所有参数组合，支持多组独立绑定"""
        if self.variable_bindings:
            # 识别独立的绑定组
            binding_groups = self._identify_binding_groups()

            # 收集所有绑定中出现的键
            all_bound_keys = set()
            for group in binding_groups:
                if group:
                    all_bound_keys.update(group[0].keys())

            # 识别自由参数（未在任何绑定中声明的参数）
            free_param_names = [name for name in self.variable_params.keys() if name not in all_bound_keys]

            # 生成自由参数的所有组合
            if free_param_names:
                free_param_values = [
                    self.variable_params[name]
                    if isinstance(self.variable_params[name], list)
                    else [self.variable_params[name]]
                    for name in free_param_names
                ]
                free_combos = [dict(zip(free_param_names, combo)) for combo in product(*free_param_values)]
            else:
                free_combos = [{}]

            # 生成所有绑定组的笛卡尔积
            if binding_groups:
                # 每个绑定组作为一个整体参与笛卡尔积
                binding_group_combos = list(product(*binding_groups))

                # 合并每个笛卡尔积结果中的所有绑定
                merged_binding_combos = []
                for combo_tuple in binding_group_combos:
                    merged = {}
                    for binding in combo_tuple:
                        merged.update(binding)
                    merged_binding_combos.append(merged)

                # 将绑定组合与自由参数组合
                param_combinations = []
                for binding_combo in merged_binding_combos:
                    for free_combo in free_combos:
                        combo = {}
                        combo.update(binding_combo)
                        combo.update(free_combo)
                        param_combinations.append(combo)
            else:
                param_combinations = free_combos
        else:
            # 没有绑定时，使用所有参数的笛卡尔积
            if not self.variable_params:
                param_combinations = [{}]
            else:
                param_names = list(self.variable_params.keys())
                param_values = [
                    self.variable_params[name]
                    if isinstance(self.variable_params[name], list)
                    else [self.variable_params[name]]
                    for name in param_names
                ]
                param_combinations = [dict(zip(param_names, combo)) for combo in product(*param_values)]

        return param_combinations

    def validate_bindings(self) -> bool:
        """
        验证绑定的有效性
        返回 True 如果所有绑定都有效，否则返回 False
        """
        if not self.variable_bindings:
            return True

        is_valid = True

        # 识别绑定组
        binding_groups = self._identify_binding_groups()

        # print(f"Found {len(binding_groups)} binding group(s):")
        for i, group in enumerate(binding_groups):
            # print(f"  Group {i+1}: {len(group)} binding(s) with keys {list(group[0].keys())}")
            pass

        # 验证每个绑定
        for i, binding in enumerate(self.variable_bindings):
            # 检查绑定中的每个键
            for key, value in binding.items():
                # 检查键是否在 variable_params 中定义
                if key not in self.variable_params:
                    print(f"Error in binding {i}: Key '{key}' not defined in variable_params")
                    is_valid = False
                else:
                    # 检查值是否在允许的范围内
                    valid_values = self.variable_params[key]
                    if isinstance(valid_values, list):
                        if value not in valid_values:
                            print(
                                f"Error in binding {i}: Value {value} for '{key}' not in allowed values {valid_values}"
                            )
                            is_valid = False

        return is_valid

    def validate_function_calls(self) -> bool:
        """
        验证变量参数中的函数调用是否有效
        返回 True 如果所有函数调用都有效，否则返回 False
        """
        is_valid = True

        for key, value in self.variable_params.items():
            try:
                # 尝试解析函数调用（如果是的话）
                if isinstance(value, str) and value.startswith("@"):
                    # 测试解析但不执行
                    self._validate_single_function_call(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, str) and item.startswith("@"):
                            try:
                                self._validate_single_function_call(item)
                            except ValueError as e:
                                print(f"Error in variable_params['{key}'][{i}]: {e}")
                                is_valid = False
            except ValueError as e:
                print(f"Error in variable_params['{key}']: {e}")
                is_valid = False

        return is_valid

    def _validate_single_function_call(self, value: str):
        """验证单个函数调用的语法和函数存在性"""
        if not isinstance(value, str) or not value.startswith("@"):
            return

        # 移除 @ 前缀
        call_str = value[1:]

        # 解析函数名和参数
        match = re.match(r"(\w+)\((.*)\)$", call_str)
        if not match:
            raise ValueError(f"Invalid function call syntax: {value}")

        func_name = match.group(1)

        # 检查函数是否存在
        if func_name not in self.function_registry._functions:
            available_funcs = ", ".join(self.function_registry.list_functions())
            raise ValueError(f"Function '{func_name}' not found. Available functions: {available_funcs}")

        # 检查参数语法（不执行）
        args_str = match.group(2).strip()
        if args_str:
            import ast

            try:
                temp_call = f"temp_func({args_str})"
                ast.parse(temp_call, mode="eval")
            except SyntaxError as e:
                raise ValueError(f"Invalid function arguments syntax: {e}")

    def get_binding_info(self) -> str:
        """获取绑定信息的可读描述"""
        if not self.variable_bindings:
            return "No bindings defined"

        binding_groups = self._identify_binding_groups()
        info = []
        info.append(f"Total bindings: {len(self.variable_bindings)}")
        info.append(f"Binding groups: {len(binding_groups)}")

        for i, group in enumerate(binding_groups):
            keys = list(group[0].keys())
            info.append(f"  Group {i+1}: {len(group)} combinations of {keys}")

        # 计算总组合数
        total_combos = len(self._get_param_combinations())
        info.append(f"Total parameter combinations: {total_combos}")

        return "\n".join(info)

    @abstractmethod
    def generate(self):
        """生成所有文件的抽象方法"""
        pass


class ConfigGenerator(BaseTemplateGenerator):
    """JSON 配置文件生成器"""

    def __init__(self):
        super().__init__()
        self.output_dir = "configs"
        self.filename_template = "config.json"

    def _copy_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """复制配置模板"""
        return template.copy()

    def set_config_template(self, config: Dict[str, Any]):
        """设置基础配置模板（向后兼容）"""
        self.set_template(config)

    def generate_configs(self) -> List[str]:
        """生成所有配置文件（向后兼容）"""
        param_combinations = self._get_param_combinations()
        generated_files = []

        for combo in param_combinations:
            # 创建当前配置
            current_config = self.replace_template_vars(self.base_template, combo)

            # 创建输出目录
            current_output_dir = Path(self.replace_template_vars(self.output_dir, combo))
            current_output_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            filename = self.replace_template_vars(self.filename_template, combo)
            filepath = current_output_dir / filename

            # 保存配置文件
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(current_config, f, ensure_ascii=False, indent=4)

            generated_files.append(str(filepath))
            print(f"Generated config: {filepath}")

        return generated_files

    def generate(self) -> List[str]:
        """生成所有配置文件"""
        # 验证函数调用
        if not self.validate_function_calls():
            print("Function call validation failed. Generation aborted.")
            return []

        # 验证绑定
        if not self.validate_bindings():
            print("Binding validation failed. Generation aborted.")
            return []

        param_combinations = self._get_param_combinations()
        generated_files = []

        for combo in param_combinations:
            try:
                # 创建当前配置
                current_config = self.replace_template_vars(self.base_template, combo)

                # 创建输出目录
                current_output_dir = Path(self.replace_template_vars(self.output_dir, combo))
                current_output_dir.mkdir(parents=True, exist_ok=True)

                # 生成文件名
                filename = self.replace_template_vars(self.filename_template, combo)
                filepath = current_output_dir / filename

                # 保存配置文件
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(current_config, f, ensure_ascii=False, indent=4)

                generated_files.append(str(filepath))
                # print(f"Generated config: {filepath}")

            except Exception as e:
                print(f"Error generating config for combination {combo}: {e}")
                continue
        print(
            f"Configuration generation completed. Total files generated: {len(generated_files)} / {len(param_combinations)}"
        )
        return generated_files


class CommandGenerator(BaseTemplateGenerator):
    """单个命令生成器，管理一个命令模板的变量和绑定"""

    def __init__(self, template: str):
        super().__init__()
        self.template = template
        self.variable_params = {}
        self.variable_bindings = []

    def _copy_template(self, template: str) -> str:
        """复制模板"""
        return template

    def add_variable_binding(self, binding_dict: Dict[str, Any]):
        """添加变量绑定关系"""
        self.variable_bindings.append(binding_dict)

    def add_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """添加多个变量绑定关系"""
        self.variable_bindings.extend(binding_dicts)

    def set_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """设置变量绑定关系"""
        self.variable_bindings = binding_dicts

    def clear_variable_bindings(self):
        """清空所有变量绑定关系"""
        self.variable_bindings = []

    def generate(self) -> List[str]:
        """生成所有命令变体"""
        param_combinations = self._get_param_combinations()
        commands = []

        for combo in param_combinations:
            command = self.replace_template_vars(self.template, combo)
            commands.append(command)

        return commands


class ScriptGenerator(BaseTemplateGenerator):
    """脚本文件生成器，支持批处理和Shell脚本"""

    def __init__(self):
        super().__init__()
        self.output_dir = "scripts"
        self.filename_template = "script.sh"
        self.command_generators = []  # 存储命令生成器

    def _copy_template(self, template: str) -> str:
        """复制模板"""
        return template

    def add_command(self, template: str, bindings: Optional[List[Dict[str, Any]]] = None, **params):
        """
        添加带特定 bindings 的命令块
        例如: add_command_with_bindings("python main.py --arg1 {a}", [{"a": 1}, {"a": 2}], epochs=[10, 20])
        """
        command_gen = CommandGenerator(template)
        command_gen.set_variable_params(**params)
        if bindings:
            command_gen.set_variable_bindings(bindings)
        self.command_generators.append(command_gen)

    def add_commands(self, command_dicts: List[Dict[str, Any]]):
        """
        批量添加命令块
        例如: add_commands([{"template":
                                "python main.py --arg1 {a}",
                             "bindings": [{"a": 1}, {"a": 2}],
                             "epochs": [10, 20]}])
        """
        for cmd in command_dicts:
            template = cmd.pop("template", "")
            bindings = cmd.pop("bindings", None)
            params = cmd.pop("params", {})
            self.add_command(template, bindings, **params)

    def clear_commands(self):
        """清空所有命令块"""
        self.command_generators = []

    def generate(self) -> List[str]:
        """生成脚本文件"""
        generated_files = []

        # 创建输出目录
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        filename = self.replace_template_vars(self.filename_template, {})
        filepath = output_path / filename

        # 生成脚本内容
        script_content = []

        # 添加命令块
        for command_gen in self.command_generators:
            # 生成该命令块的所有变体
            commands = command_gen.generate()

            # 添加到脚本内容
            for command in commands:
                script_content.append(command)

            # 在命令块之间添加空行
            script_content.append("")

        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(script_content))

        generated_files.append(str(filepath))
        print(f"Generated script: {filepath}")

        return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="配置和脚本生成器")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 添加子命令
    config_parser = subparsers.add_parser("config", help="生成配置文件")
    config_preset_group = config_parser.add_argument_group("preset", "预设配置选项")
    config_preset_group.add_argument(
        "--from-preset", "-f", type=str, help="JSON 预设文件加载路径，从中加载配置模板和相关信息进行批量生成配置文件"
    )
    # TODO config_preset_group.add_argument('--save-preset', "-s", type=str, help='JSON 预设文件保存路径，用于将当前设置保存为预设文件，便于后续加载使用')
    config_iteractive_group = config_parser.add_argument_group(
        "interactive", "交互式配置选项（与预设配置不能同时指定）"
    )
    config_iteractive_group.add_argument("--template", "-t", type=str, help="JSON 配置模板字符串或文件路径")
    config_iteractive_group.add_argument(
        "--variable-params",
        "-v",
        type=str,
        help='可变参数，JSON 字符串格式，类型为Dict[str, List[Any]]，例如 \'{"dim": [4,8,16], "lr": [0.001,0.01]}\'',
    )
    config_iteractive_group.add_argument(
        "--variable-bindings",
        "-b",
        type=str,
        help='变量绑定关系，JSON 字符串格式，类型为List[Dict[str, Any]]，例如 \'[{"size":"tiny","dim":4},{"size":"base","dim":8}]\'',
    )
    config_iteractive_group.add_argument(
        "--output-dir", "-o", type=str, default="configs", help="输出目录，支持{var_name}格式，默认configs"
    )
    config_iteractive_group.add_argument(
        "--filename", "-n", type=str, default="config.json", help="文件名模板，支持{var_name}格式，默认config.json"
    )

    script_parser = subparsers.add_parser("script", help="生成脚本文件")
    script_preset_group = script_parser.add_argument_group("config", "配置选项")
    script_preset_group.add_argument("--from-preset", "-f", type=str, help="JSON 配置模板文件路径")

    # TODO script_interactive_group = script_parser.add_argument_group("interactive", "交互式配置选项（与预设配置不能同时指定）")
    # script_interactive_group.add_argument('--add-command', "-c", type=str, action='append', help='可变参数，JSON 字符串格式，类型为Dict[str, List[Any]]，例如 \'{"dim": [4,8,16], "lr": [0.001,0.01]}\'')
    # script_interactive_group.add_argument('--output-dir', "-o", type=str, default="scripts", help='输出目录,默认scripts/')
    # script_interactive_group.add_argument('--filename', "-n", type=str, default="script.py", help='文件名模板，默认script.py')

    args = parser.parse_args()

    if args.command == "config":
        if args.from_preset:
            with open(args.from_preset, "r", encoding="utf-8") as f:
                presets: List[Dict[str, Any]] = json.load(f)
            for preset in presets:
                generator = ConfigGenerator()
                generator.set_template(preset.get("template", {}))
                # 处理可能包含函数调用的变量参数
                var_params = preset.get("variable_params", {})
                generator.set_variable_params(**var_params)
                generator.set_variable_bindings(preset.get("variable_bindings", []))
                generator.set_output_dir(preset.get("output_dir", "configs"))
                generator.set_filename(preset.get("filename", "config.json"))
                generator.generate()
        else:
            if not args.template:
                print("Error: --template is required when not using --from-preset")
                exit(1)
            # 解析模板
            if os.path.isfile(args.template):
                with open(args.template, "r", encoding="utf-8") as f:
                    template = json.load(f)
            else:
                try:
                    template = json.loads(args.template)
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON template - {e}")
                    exit(1)
            generator = ConfigGenerator()
            generator.set_template(template)
            # 解析可变参数
            if args.variable_params:
                try:
                    var_params = json.loads(args.variable_params)
                    if not isinstance(var_params, dict):
                        raise ValueError("variable_params must be a JSON object (Dict[str, List[Any]])")
                    generator.set_variable_params(**var_params)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error: Invalid variable_params - {e}")
                    exit(1)
            # 解析变量绑定
            if args.variable_bindings:
                try:
                    var_bindings = json.loads(args.variable_bindings)
                    if not isinstance(var_bindings, list):
                        raise ValueError("variable_bindings must be a JSON array (List[Dict[str, Any]])")
                    generator.set_variable_bindings(var_bindings)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error: Invalid variable_bindings - {e}")
                    exit(1)
            generator.set_output_dir(args.output_dir)
            generator.set_filename(args.filename)
            generator.generate()
    elif args.command == "script":
        with open(args.from_preset, "r", encoding="utf-8") as f:
            preset: Dict[str, Any] = json.load(f)
        generator = ScriptGenerator()
        generator.set_output_dir(preset.get("output_dir", "scripts"))
        generator.set_filename(preset.get("filename", "script.sh"))
        commands = preset.get("commands", [])
        for cmd in commands:
            # 创建副本以避免修改原始数据
            cmd_copy = cmd.copy()
            template = cmd_copy.pop("template", "")
            bindings = cmd_copy.pop("variable_bindings", None)
            params = cmd_copy.pop("variable_params", {})
            generator.add_command(template, bindings, **params)
        generator.generate()
