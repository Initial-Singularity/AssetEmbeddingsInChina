"""Template-based batch generation of JSON config files and runner scripts from variable bindings.

A template (a JSON object or a command string) carries ``{name}`` / ``{name:format}`` placeholders; variable
parameters are expanded into their Cartesian product, variable bindings pin parameters that must move together,
and ``@function(...)`` strings produce parameter lists from registered helpers. Runs as
``python -m scripts.tools.fast_generator config|script``.
"""

import os
import re
import ast
import json
import argparse
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable


def generate_time_quarters(
    start_year: int, start_quarter: int, end_year: int, end_quarter: int, format: str = "%YQ%Q"
) -> List[str]:
    """List the quarters from start to end inclusive, such as ``2020Q1`` to ``2024Q4``.

    Args:
        start_year: First year.
        start_quarter: First quarter (1-4) of the first year.
        end_year: Last year.
        end_quarter: Last quarter (1-4) of the last year.
        format: Label pattern; ``%Y`` is replaced by the year and ``%Q`` by the quarter number.

    Returns:
        Quarter labels in chronological order.
    """
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
    """List the months from start to end inclusive, such as ``2020-01`` to ``2024-12``.

    Args:
        start_year: First year.
        start_month: First month (1-12) of the first year.
        end_year: Last year.
        end_month: Last month (1-12) of the last year.
        format: Label pattern; ``%Y`` is replaced by the year and ``%m`` by the zero-padded month.

    Returns:
        Month labels in chronological order.
    """
    months = []
    for year in range(start_year, end_year + 1):
        start_m = start_month if year == start_year else 1
        end_m = end_month if year == end_year else 12
        for month in range(start_m, end_m + 1):
            months.append(format.replace("%Y", str(year)).replace("%m", f"{month:02d}"))
    return months


def generate_date_range(start_date: str, end_date: str, freq: str = "D", format: str = "%Y-%m-%d") -> List[str]:
    """List the dates from start to end inclusive at a daily, weekly, monthly, quarterly or yearly step.

    Args:
        start_date: First date as ``YYYY-MM-DD``.
        end_date: Last date as ``YYYY-MM-DD``.
        freq: Step size: ``D`` day, ``W`` week, ``M`` month, ``Q`` quarter, ``Y`` year.
        format: ``strftime`` pattern for the labels.

    Returns:
        Date labels in chronological order.
    """
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
                # the day does not exist in the target month (Jan 31 -> Feb): fall back to its first day
                current = current.replace(year=year, month=month, day=1)
        elif freq == "Q":
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
    """Return the Cartesian product of the given lists, one list per combination.

    Args:
        *lists: The lists to combine.

    Returns:
        Every combination as a list, in ``itertools.product`` order.
    """
    from itertools import product

    return [list(combo) for combo in product(*lists)]


class FunctionRegistry:
    """Name-to-callable registry behind the ``@function(...)`` syntax of variable parameters.

    The built-in entries are the time helpers of this module (``generate_time_quarters``, ``generate_time_months``,
    ``generate_date_range``), the numeric sequences ``range``, ``linspace``, ``logspace`` and ``geomspace``, and the
    list helpers ``repeat``, ``zip_lists`` and ``generate_cross_product``.
    """

    def __init__(self):
        self._functions: Dict[str, Callable] = {}
        self._register_builtin_functions()

    def _register_builtin_functions(self):
        self.register("generate_time_quarters", generate_time_quarters)
        self.register("generate_time_months", generate_time_months)
        self.register("generate_date_range", generate_date_range)

        self.register("range", lambda start, stop, step=1: list(range(start, stop, step)))
        self.register("linspace", lambda start, stop, num: [start + i * (stop - start) / (num - 1) for i in range(num)])
        self.register(
            "logspace", lambda start, stop, num: [10 ** (start + i * (stop - start) / (num - 1)) for i in range(num)]
        )
        self.register(
            "geomspace", lambda start, stop, num: [start * (stop / start) ** (i / (num - 1)) for i in range(num)]
        )

        self.register("repeat", lambda value, times: [value] * times)
        self.register("zip_lists", lambda *lists: [list(item) for item in zip(*lists)])
        self.register("generate_cross_product", generate_cross_product)

    def register(self, name: str, func: Callable):
        """Register ``func`` under ``name``, replacing any earlier entry with that name.

        Args:
            name: The name used after ``@`` in a variable-parameter string.
            func: The callable to run.
        """
        self._functions[name] = func

    def call(self, name: str, *args, **kwargs) -> Any:
        """Run the registered function ``name`` with the given arguments.

        Args:
            name: A registered function name.
            *args: Positional arguments passed through.
            **kwargs: Keyword arguments passed through.

        Returns:
            Whatever the function returns.

        Raises:
            ValueError: If no function is registered under ``name``.
        """
        if name not in self._functions:
            raise ValueError(f"Function '{name}' not found in registry")
        return self._functions[name](*args, **kwargs)

    def list_functions(self) -> List[str]:
        """Return the registered function names in registration order."""
        return list(self._functions.keys())


class BaseTemplateGenerator(ABC):
    """Shared machinery of the generators: template, variable parameters, variable bindings and output naming.

    Subclasses decide how a template is copied and what ``generate`` writes. Variable parameters expand into their
    Cartesian product; variable bindings with the same key set form one group whose members are alternatives, and
    the groups multiply with each other and with the free (unbound) parameters.
    """

    def __init__(self):
        self.base_template: Any = {}
        self.variable_params: Dict[str, Any] = {}
        self.variable_bindings: List[Dict[str, Any]] = []
        self.output_dir: str = "output"
        self.filename_template: str = "file"
        self.function_registry = FunctionRegistry()

    def set_template(self, template: Any):
        """Store a copy of ``template`` as the base template.

        Args:
            template: A JSON-like object or a string with ``{name}`` placeholders.
        """
        self.base_template = self._copy_template(template)

    @abstractmethod
    def _copy_template(self, template: Any) -> Any:
        pass

    def _parse_function_call(self, value: str) -> Any:
        """Evaluate an ``@name(arg, ..., key=value)`` string through the registry; other values pass through.

        Arguments must be Python literals (scalars, lists, tuples, dicts); names and nested calls are rejected.

        Args:
            value: The variable-parameter value.

        Returns:
            The function's result, or ``value`` itself when it is not an ``@`` string.

        Raises:
            ValueError: On malformed syntax, a non-literal argument, an unknown function or a failing call.
        """
        if not isinstance(value, str) or not value.startswith("@"):
            return value

        call_str = value[1:]

        match = re.match(r"(\w+)\((.*)\)$", call_str)
        if not match:
            raise ValueError(f"Invalid function call syntax: {value}")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        args = []
        kwargs = {}

        if args_str:
            try:
                # wrap the argument list in a dummy call so ast separates positional from keyword arguments
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

        try:
            result = self.function_registry.call(func_name, *args, **kwargs)
            return result
        except Exception as e:
            raise ValueError(f"Error calling function '{func_name}': {e}")

    def _process_variable_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``@`` calls in the values; a call inside a list splices its result into that list."""
        processed = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("@"):
                result = self._parse_function_call(value)
                # a scalar result must still be a list: the Cartesian product iterates over it
                if not isinstance(result, list):
                    result = [result]
                processed[key] = result
            elif isinstance(value, list):
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
        """Replace the variable parameters, resolving ``@function(...)`` strings into value lists.

        Args:
            **params: Parameter name to a list of values, a single value, or an ``@`` call string.
        """
        self.variable_params = self._process_variable_params(params)

    def add_variable_binding(self, binding_dict: Dict[str, Any]):
        """Append one binding, a dict of parameter values that are only ever used together.

        Args:
            binding_dict: Parameter name to the value it takes in this binding.
        """
        self.variable_bindings.append(binding_dict)

    def add_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """Append several bindings.

        Args:
            binding_dicts: Bindings as accepted by ``add_variable_binding``.
        """
        self.variable_bindings.extend(binding_dicts)

    def clear_variable_bindings(self):
        """Remove every binding."""
        self.variable_bindings = []

    def set_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """Replace the bindings with ``binding_dicts``.

        Args:
            binding_dicts: Bindings as accepted by ``add_variable_binding``.
        """
        self.variable_bindings = binding_dicts

    def set_output_dir(self, dir_template: str):
        """Set the output directory; ``{name}`` placeholders are filled per combination.

        Args:
            dir_template: Directory path, possibly with placeholders.
        """
        self.output_dir = dir_template

    def set_filename(self, filename_template: str):
        """Set the file name; ``{name}`` placeholders are filled per combination.

        Args:
            filename_template: File name, possibly with placeholders.
        """
        self.filename_template = filename_template

    def _format_value(self, value: Any, format_spec: str) -> str:
        """Render ``value`` with a Python format spec (``.2f``, ``04d``, ``>10s``); non-scalars are str()-ed first."""
        if format_spec:
            if isinstance(value, (int, float)):
                return f"{value:{format_spec}}"
            elif isinstance(value, str):
                return f"{value:{format_spec}}"
            else:
                return f"{str(value):{format_spec}}"
        return str(value)

    def replace_template_vars(self, value: Any, replacements: Dict[str, Any]) -> Any:
        """Fill ``{name}`` and ``{name:format}`` placeholders in a string, recursing through dicts and lists.

        A string that is exactly one placeholder is replaced by the value itself, so ``"{dim}"`` becomes the
        integer ``128`` rather than ``"128"``; a format spec always yields a string. Placeholders whose name is not
        in ``replacements`` are left untouched.

        Args:
            value: A string, dict, list or scalar.
            replacements: Parameter name to value.

        Returns:
            The filled copy; scalars other than strings are returned as they are.
        """
        if isinstance(value, str):
            pattern = r"\{([^{}:]+)(?::([^{}]*))?\}"
            matches = list(re.finditer(pattern, value))

            if len(matches) == 1 and matches[0].group(0) == value:
                # a whole-string placeholder keeps the value's own type unless a format spec forces a string
                var_name = matches[0].group(1)
                format_spec = matches[0].group(2) or ""

                if var_name in replacements:
                    if format_spec:
                        return self._format_value(replacements[var_name], format_spec)
                    else:
                        return replacements[var_name]
                return value

            result = value
            for match in reversed(matches):  # right to left so the earlier match offsets stay valid
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
        """Group the bindings by key set: bindings with identical keys are alternatives within one group."""
        if not self.variable_bindings:
            return []

        binding_groups = []
        used_bindings = set()

        for i, binding in enumerate(self.variable_bindings):
            if i in used_bindings:
                continue

            group = [binding]
            used_bindings.add(i)
            binding_keys = set(binding.keys())

            for j, other_binding in enumerate(self.variable_bindings):
                if j > i and j not in used_bindings:
                    if set(other_binding.keys()) == binding_keys:
                        group.append(other_binding)
                        used_bindings.add(j)

            binding_groups.append(group)

        return binding_groups

    def _get_param_combinations(self) -> List[Dict[str, Any]]:
        """Enumerate the parameter combinations: product of the binding groups times the free parameters."""
        if self.variable_bindings:
            binding_groups = self._identify_binding_groups()

            all_bound_keys = set()
            for group in binding_groups:
                if group:
                    all_bound_keys.update(group[0].keys())

            free_param_names = [name for name in self.variable_params.keys() if name not in all_bound_keys]

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

            if binding_groups:
                # a group takes part in the product as one unit: its members are alternatives, not factors
                binding_group_combos = list(product(*binding_groups))

                merged_binding_combos = []
                for combo_tuple in binding_group_combos:
                    merged = {}
                    for binding in combo_tuple:
                        merged.update(binding)
                    merged_binding_combos.append(merged)

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
        """Check that every bound key is a variable parameter and every bound value is among its allowed values.

        Each violation is printed; with no bindings the check passes.

        Returns:
            True if all bindings are valid, False otherwise.
        """
        if not self.variable_bindings:
            return True

        is_valid = True

        binding_groups = self._identify_binding_groups()

        for i, binding in enumerate(self.variable_bindings):
            for key, value in binding.items():
                if key not in self.variable_params:
                    print(f"Error in binding {i}: Key '{key}' not defined in variable_params")
                    is_valid = False
                else:
                    valid_values = self.variable_params[key]
                    if isinstance(valid_values, list):
                        if value not in valid_values:
                            print(
                                f"Error in binding {i}: Value {value} for '{key}' not in allowed values {valid_values}"
                            )
                            is_valid = False

        return is_valid

    def validate_function_calls(self) -> bool:
        """Check the syntax and function names of every ``@`` call in the variable parameters without running them.

        Each violation is printed.

        Returns:
            True if all calls are well formed and registered, False otherwise.
        """
        is_valid = True

        for key, value in self.variable_params.items():
            try:
                if isinstance(value, str) and value.startswith("@"):
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
        """Raise ValueError if ``value`` is an ``@`` call with bad syntax or an unregistered name; never calls it."""
        if not isinstance(value, str) or not value.startswith("@"):
            return

        call_str = value[1:]

        match = re.match(r"(\w+)\((.*)\)$", call_str)
        if not match:
            raise ValueError(f"Invalid function call syntax: {value}")

        func_name = match.group(1)

        if func_name not in self.function_registry._functions:
            available_funcs = ", ".join(self.function_registry.list_functions())
            raise ValueError(f"Function '{func_name}' not found. Available functions: {available_funcs}")

        args_str = match.group(2).strip()
        if args_str:
            import ast

            try:
                temp_call = f"temp_func({args_str})"
                ast.parse(temp_call, mode="eval")
            except SyntaxError as e:
                raise ValueError(f"Invalid function arguments syntax: {e}")

    def get_binding_info(self) -> str:
        """Describe the bindings: their count, the groups with their key sets, and the resulting combination count.

        Returns:
            A multi-line summary, or ``"No bindings defined"``.
        """
        if not self.variable_bindings:
            return "No bindings defined"

        binding_groups = self._identify_binding_groups()
        info = []
        info.append(f"Total bindings: {len(self.variable_bindings)}")
        info.append(f"Binding groups: {len(binding_groups)}")

        for i, group in enumerate(binding_groups):
            keys = list(group[0].keys())
            info.append(f"  Group {i+1}: {len(group)} combinations of {keys}")

        total_combos = len(self._get_param_combinations())
        info.append(f"Total parameter combinations: {total_combos}")

        return "\n".join(info)

    @abstractmethod
    def generate(self):
        """Write every output and return what was produced."""
        pass


class ConfigGenerator(BaseTemplateGenerator):
    """Generator that writes one JSON config file per parameter combination.

    Defaults to ``configs/config.json``; set the directory and file name templates to spread the files out.
    """

    def __init__(self):
        super().__init__()
        self.output_dir = "configs"
        self.filename_template = "config.json"

    def _copy_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        return template.copy()

    def set_config_template(self, config: Dict[str, Any]):
        """Set the base template; older name of ``set_template``, kept for callers that still use it.

        Args:
            config: The JSON object with placeholders.
        """
        self.set_template(config)

    def generate_configs(self) -> List[str]:
        """Write every config file without the validation and per-file error handling of ``generate``.

        Older entry point kept for callers that still use it; each written path is printed.

        Returns:
            The paths of the written files.
        """
        param_combinations = self._get_param_combinations()
        generated_files = []

        for combo in param_combinations:
            current_config = self.replace_template_vars(self.base_template, combo)

            current_output_dir = Path(self.replace_template_vars(self.output_dir, combo))
            current_output_dir.mkdir(parents=True, exist_ok=True)

            filename = self.replace_template_vars(self.filename_template, combo)
            filepath = current_output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(current_config, f, ensure_ascii=False, indent=4)

            generated_files.append(str(filepath))
            print(f"Generated config: {filepath}")

        return generated_files

    def generate(self) -> List[str]:
        """Validate the function calls and bindings, then write one config file per combination.

        A combination that fails (an unwritable path, say) is reported and skipped; the others are still
        written. Validation failure aborts before anything is written.

        Returns:
            The paths of the written files; empty when validation failed.
        """
        if not self.validate_function_calls():
            print("Function call validation failed. Generation aborted.")
            return []

        if not self.validate_bindings():
            print("Binding validation failed. Generation aborted.")
            return []

        param_combinations = self._get_param_combinations()
        generated_files = []

        for combo in param_combinations:
            try:
                current_config = self.replace_template_vars(self.base_template, combo)

                current_output_dir = Path(self.replace_template_vars(self.output_dir, combo))
                current_output_dir.mkdir(parents=True, exist_ok=True)

                filename = self.replace_template_vars(self.filename_template, combo)
                filepath = current_output_dir / filename

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(current_config, f, ensure_ascii=False, indent=4)

                generated_files.append(str(filepath))

            except Exception as e:
                print(f"Error generating config for combination {combo}: {e}")
                continue
        print(
            f"Configuration generation completed. Total files generated: {len(generated_files)} / {len(param_combinations)}"
        )
        return generated_files


class CommandGenerator(BaseTemplateGenerator):
    """Generator for one command template with its own variable parameters and bindings.

    ``generate`` yields one command line per parameter combination.
    """

    def __init__(self, template: str):
        super().__init__()
        self.template = template
        self.variable_params = {}
        self.variable_bindings = []

    def _copy_template(self, template: str) -> str:
        return template

    def add_variable_binding(self, binding_dict: Dict[str, Any]):
        """Append one binding.

        Args:
            binding_dict: Parameter name to the value it takes in this binding.
        """
        self.variable_bindings.append(binding_dict)

    def add_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """Append several bindings.

        Args:
            binding_dicts: Bindings as accepted by ``add_variable_binding``.
        """
        self.variable_bindings.extend(binding_dicts)

    def set_variable_bindings(self, binding_dicts: List[Dict[str, Any]]):
        """Replace the bindings with ``binding_dicts``.

        Args:
            binding_dicts: Bindings as accepted by ``add_variable_binding``.
        """
        self.variable_bindings = binding_dicts

    def clear_variable_bindings(self):
        """Remove every binding."""
        self.variable_bindings = []

    def generate(self) -> List[str]:
        """Fill the command template for every parameter combination.

        Returns:
            The command lines, in combination order.
        """
        param_combinations = self._get_param_combinations()
        commands = []

        for combo in param_combinations:
            command = self.replace_template_vars(self.template, combo)
            commands.append(command)

        return commands


class ScriptGenerator(BaseTemplateGenerator):
    """Generator that writes one shell script from a sequence of command blocks.

    Each block is a `CommandGenerator`; its expanded command lines are written in order, with a blank line after
    every block. Defaults to ``scripts/script.sh``.
    """

    def __init__(self):
        super().__init__()
        self.output_dir = "scripts"
        self.filename_template = "script.sh"
        self.command_generators = []

    def _copy_template(self, template: str) -> str:
        return template

    def add_command(self, template: str, bindings: Optional[List[Dict[str, Any]]] = None, **params):
        """Append a command block with its own variable parameters and bindings.

        Args:
            template: Command line with ``{name}`` placeholders.
            bindings: Bindings for this block, or None for none.
            **params: Variable parameters for this block, as for ``set_variable_params``.
        """
        command_gen = CommandGenerator(template)
        command_gen.set_variable_params(**params)
        if bindings:
            command_gen.set_variable_bindings(bindings)
        self.command_generators.append(command_gen)

    def add_commands(self, command_dicts: List[Dict[str, Any]]):
        """Append several command blocks given as dicts with the keys ``template``, ``bindings`` and ``params``.

        The keys are popped from each dict, so the dicts are modified in place; a missing key means an empty
        template, no bindings or no parameters.

        Args:
            command_dicts: One dict per block.
        """
        for cmd in command_dicts:
            template = cmd.pop("template", "")
            bindings = cmd.pop("bindings", None)
            params = cmd.pop("params", {})
            self.add_command(template, bindings, **params)

    def clear_commands(self):
        """Remove every command block."""
        self.command_generators = []

    def generate(self) -> List[str]:
        """Write the script: the expanded lines of every block, a blank line after each block.

        Returns:
            A one-element list with the path of the written script.
        """
        generated_files = []

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = self.replace_template_vars(self.filename_template, {})
        filepath = output_path / filename

        script_content = []

        for command_gen in self.command_generators:
            commands = command_gen.generate()

            for command in commands:
                script_content.append(command)

            script_content.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(script_content))

        generated_files.append(str(filepath))
        print(f"Generated script: {filepath}")

        return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config and script generator")
    subparsers = parser.add_subparsers(dest="command", help="subcommand")

    config_parser = subparsers.add_parser("config", help="generate config files")
    config_preset_group = config_parser.add_argument_group("preset", "preset options")
    config_preset_group.add_argument(
        "--from-preset",
        "-f",
        type=str,
        help="JSON preset file to load; its templates, variable parameters, bindings and output settings drive the batch",
    )
    config_preset_group.add_argument(
        "--set-variable",
        "-V",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a variable_params entry of every preset (repeatable); VALUE is parsed as a Python literal, e.g. -V dim=128",
    )
    config_iteractive_group = config_parser.add_argument_group(
        "interactive", "interactive options (not combinable with a preset)"
    )
    config_iteractive_group.add_argument(
        "--template", "-t", type=str, help="JSON config template, as a string or a file path"
    )
    config_iteractive_group.add_argument(
        "--variable-params",
        "-v",
        type=str,
        help='variable parameters as a JSON object of type Dict[str, List[Any]], e.g. \'{"dim": [4,8,16], "lr": [0.001,0.01]}\'',
    )
    config_iteractive_group.add_argument(
        "--variable-bindings",
        "-b",
        type=str,
        help='variable bindings as a JSON array of type List[Dict[str, Any]], e.g. \'[{"size":"tiny","dim":4},{"size":"base","dim":8}]\'',
    )
    config_iteractive_group.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="configs",
        help="output directory, {var_name} placeholders allowed (default: configs)",
    )
    config_iteractive_group.add_argument(
        "--filename",
        "-n",
        type=str,
        default="config.json",
        help="file name template, {var_name} placeholders allowed (default: config.json)",
    )

    script_parser = subparsers.add_parser("script", help="generate a script file")
    script_preset_group = script_parser.add_argument_group("config", "preset options")
    script_preset_group.add_argument("--from-preset", "-f", type=str, help="JSON preset file with the command blocks")

    args = parser.parse_args()

    if args.command == "config":
        if args.from_preset:
            with open(args.from_preset, "r", encoding="utf-8") as f:
                presets: List[Dict[str, Any]] = json.load(f)
            for preset in presets:
                generator = ConfigGenerator()
                generator.set_template(preset.get("template", {}))
                var_params = preset.get("variable_params", {})
                for item in args.set_variable:
                    name, sep, raw = item.partition("=")
                    if not name or not sep:
                        raise SystemExit(f"--set-variable expects NAME=VALUE, got {item!r}")
                    try:
                        var_params[name] = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        var_params[name] = raw
                generator.set_variable_params(**var_params)
                generator.set_variable_bindings(preset.get("variable_bindings", []))
                generator.set_output_dir(preset.get("output_dir", "configs"))
                generator.set_filename(preset.get("filename", "config.json"))
                generator.generate()
        else:
            if not args.template:
                print("Error: --template is required when not using --from-preset")
                exit(1)
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
            if args.variable_params:
                try:
                    var_params = json.loads(args.variable_params)
                    if not isinstance(var_params, dict):
                        raise ValueError("variable_params must be a JSON object (Dict[str, List[Any]])")
                    generator.set_variable_params(**var_params)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error: Invalid variable_params - {e}")
                    exit(1)
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
            # pop() below would otherwise strip the keys from the loaded preset itself
            cmd_copy = cmd.copy()
            template = cmd_copy.pop("template", "")
            bindings = cmd_copy.pop("variable_bindings", None)
            params = cmd_copy.pop("variable_params", {})
            generator.add_command(template, bindings, **params)
        generator.generate()
