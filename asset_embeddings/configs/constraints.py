"""Constraint classes that `ConfigField` and `Config` use to validate configuration values.

A field constraint (`Constraint`) judges one value and runs on every assignment and again in
`Config.validate()`; a cross-field constraint (`CrossConstraint`) judges a whole config and runs only
in `validate()`. A field opts in explicitly through `ConfigField(constraint=...)`; the default
`NoConstraint()` checks nothing.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Union, List, Any, Optional, Type, get_origin

if TYPE_CHECKING:
    from .base import Config
    from .container import ConfigContainer


class BaseConstraint(ABC):
    """Marker base shared by `Constraint` and `CrossConstraint`.

    The two branches take different arguments and are not interchangeable; the base only
    guarantees the three-method shape, gives `isinstance` one type to test, and supplies a
    fallback `__repr__`.
    """

    @abstractmethod
    def validate(self, *args, **kwargs):
        """Return True when the constraint holds."""
        pass

    @abstractmethod
    def check(self, *args, **kwargs):
        """Validate and raise `ConstraintError` on failure."""
        pass

    @abstractmethod
    def get_error_message(self, *args, **kwargs):
        """Return the message that `check` raises with."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Constraint(BaseConstraint):
    """Base class for constraints on a single field value.

    Subclasses implement `validate` and `get_error_message`; `&` and `|` compose constraints into
    `IntersectionConstraints` and `UnionConstraints`.
    """

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Return True when the value satisfies this constraint.

        Args:
            value: The value to validate.

        Returns:
            True if the value is acceptable, False otherwise.
        """
        pass

    @abstractmethod
    def get_error_message(self, field_name: str, value: Any) -> str:
        """Describe why the value fails, for the `ConstraintError` that `check` raises.

        Args:
            field_name: Name of the field being validated.
            value: The value that failed.

        Returns:
            The error message.
        """
        pass

    def check(self, field_name: str, value: Any) -> None:
        """Validate the value and raise on failure.

        Any exception escaping `validate` is wrapped too, so callers see one error type.

        Args:
            field_name: Name of the field being validated.
            value: The value to validate.

        Raises:
            ConstraintError: If the value fails or `validate` itself raises.
        """
        try:
            if not self.validate(value):
                raise ConstraintError(self.get_error_message(field_name, value))
        except Exception as e:
            if isinstance(e, ConstraintError):
                raise ConstraintError(e.message)
            raise ConstraintError(f"Unexpected error occurred while validating '{field_name}': {e}")

    def __and__(self, other: "Constraint") -> "IntersectionConstraints":
        return IntersectionConstraints(self, other)

    def __or__(self, other: "Constraint") -> "UnionConstraints":
        return UnionConstraints(self, other)


class ConstraintError(Exception):
    """Raised by `check` when a value or config fails a constraint.

    Attributes:
        message: The failure description.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TypeConstraint(Constraint):
    """Requires the value to be an instance of one type.

    Attributes:
        expected_type: The type tested with `isinstance`.
    """

    def __init__(self, expected_type: Type):
        self.expected_type = expected_type

    def validate(self, value: Any) -> bool:
        """Return True when the value is an instance of the expected type."""
        # A subscripted generic (List[str]) is not a class, so isinstance needs its origin;
        # element types therefore go unchecked.
        origin = get_origin(self.expected_type)
        if origin is not None:
            return isinstance(value, origin)

        return isinstance(value, self.expected_type)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """Name the expected and the actual type."""
        if hasattr(self.expected_type, "__name__"):
            type_name = self.expected_type.__name__
        else:
            type_name = str(self.expected_type)
        return f"Field '{field_name}' expected {type_name}, got {type(value).__name__}"

    def __repr__(self) -> str:
        return f"TypeConstraint(expected_type={self.expected_type})"


class RangeConstraint(Constraint):
    """Requires a numeric value to lie in the closed interval `[min_val, max_val]`.

    Attributes:
        min_val: Lower bound (inclusive), or None for no lower bound.
        max_val: Upper bound (inclusive), or None for no upper bound.
    """

    def __init__(self, min_val: Union[int, float, None], max_val: Union[int, float, None]):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value: Any) -> bool:
        """Return True when the value lies within the bounds; False when both bounds are None."""
        if self.min_val is not None and self.max_val is not None:
            return self.min_val <= value <= self.max_val
        elif self.min_val is not None:
            return self.min_val <= value
        elif self.max_val is not None:
            return value <= self.max_val
        return False

    def get_error_message(self, field_name: str, value: Any) -> str:
        """State the bounds and the offending value."""
        return f"Field '{field_name}' must be between {self.min_val} and {self.max_val}, got {value}"

    def __repr__(self) -> str:
        return f"RangeConstraint(min_val={self.min_val}, max_val={self.max_val})"


class ChoiceConstraint(Constraint):
    """Requires the value to be one of a fixed set of choices.

    Attributes:
        choices: The allowed values.
    """

    def __init__(self, choices: List[Any]):
        self.choices = choices

    def validate(self, value: Any) -> bool:
        """Return True when the value is one of the choices."""
        return value in self.choices

    def get_error_message(self, field_name: str, value: Any) -> str:
        """List the choices and the offending value."""
        return f"Field '{field_name}' must be one of {self.choices}, got {value}"

    def __repr__(self) -> str:
        return f"ChoiceConstraint(choices={self.choices})"


class ChoicesConstraint(Constraint):
    """Requires a list or tuple whose every element is one of a fixed set of choices.

    The element-wise analogue of `ChoiceConstraint`; an empty sequence passes.

    Attributes:
        choices: The allowed element values.
    """

    def __init__(self, choices: List[Any]):
        self.choices = choices

    def validate(self, value: Any) -> bool:
        """Return True when the value is a list or tuple whose elements are all choices."""
        if not isinstance(value, (list, tuple)):
            return False
        return all(v in self.choices for v in value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """Name the wrong container type, or list the elements outside the choices."""
        if not isinstance(value, (list, tuple)):
            return f"Field '{field_name}' must be a list/tuple, " f"got {type(value).__name__}"
        bad = [v for v in value if v not in self.choices]
        return f"Field '{field_name}' contains invalid elements {bad}; " f"allowed values are {self.choices}"

    def __repr__(self) -> str:
        return f"ChoicesConstraint(choices={self.choices})"


class EqualityConstraint(Constraint):
    """Requires the value to equal one expected value under `==`.

    Attributes:
        expected_value: The value to equal.
    """

    def __init__(self, expected_value: Any):
        self.expected_value = expected_value

    def validate(self, value: Any) -> bool:
        """Return True when the value equals the expected value."""
        return value == self.expected_value

    def get_error_message(self, field_name: str, value: Any) -> str:
        """State the expected and the offending value."""
        return f"Field '{field_name}' must be equal to {self.expected_value}, got {value}"

    def __repr__(self) -> str:
        return f"EqualityConstraint(expected_value={self.expected_value})"


class LengthConstraint(Constraint):
    """Requires `len(value)` to lie in the closed interval `[min_length, max_length]`.

    The signature mirrors `RangeConstraint`; `LengthConstraint(N, N)` checks an exact length.

    Attributes:
        min_length: Lower bound (inclusive), or None for no lower bound.
        max_length: Upper bound (inclusive), or None for no upper bound.
    """

    def __init__(self, min_length: Optional[int], max_length: Optional[int]):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any) -> bool:
        """Return True when the length lies within the bounds; False when both bounds are None.

        Raises:
            TypeError: If the value has no `len()`; `check` wraps it into a `ConstraintError`.
        """
        n = len(value)
        if self.min_length is not None and self.max_length is not None:
            return self.min_length <= n <= self.max_length
        if self.min_length is not None:
            return self.min_length <= n
        if self.max_length is not None:
            return n <= self.max_length
        return False

    def get_error_message(self, field_name: str, value: Any) -> str:
        """State the length bound that was violated and the actual length."""
        n = len(value)
        if self.min_length is not None and self.max_length is not None:
            if self.min_length == self.max_length:
                return f"Field '{field_name}' must have length {self.min_length}, got {n}"
            return f"Field '{field_name}' must have length between {self.min_length} " f"and {self.max_length}, got {n}"
        if self.min_length is not None:
            return f"Field '{field_name}' must have length >= {self.min_length}, got {n}"
        if self.max_length is not None:
            return f"Field '{field_name}' must have length <= {self.max_length}, got {n}"
        return f"Field '{field_name}' length constraint has no bounds set"

    def __repr__(self) -> str:
        return f"LengthConstraint(min_length={self.min_length}, " f"max_length={self.max_length})"


class LambdaConstraint(Constraint):
    """Delegates validation to a caller-supplied predicate.

    Attributes:
        func: Predicate taking the value and returning True when it is acceptable.
        error_message: A fixed message, a callable of `(field_name, value)` producing one, or None
            for a generic message.
    """

    def __init__(self, func: Callable[[Any], bool], error_message: Union[str, Callable[[str, Any], str], None] = None):
        self.func = func
        self.error_message = error_message

    def validate(self, value: Any) -> bool:
        """Return the predicate's verdict on the value."""
        return self.func(value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """Return the configured message, calling it if it is a callable, or a generic one."""
        if self.error_message:
            if callable(self.error_message):
                return self.error_message(field_name, value)
            return self.error_message
        return f"Field '{field_name}' failed custom validation {self.func}, got {value}"

    def __repr__(self) -> str:
        return f"LambdaConstraint(func={self.func}, error_message={self.error_message})"


class NoneConstraint(Constraint):
    """Requires the value to be None."""

    def validate(self, value: Any) -> bool:
        """Return True when the value is None."""
        return value is None

    def get_error_message(self, field_name: str, value: Any) -> str:
        """State that the field must be None."""
        return f"Field '{field_name}' must be None, got {value}"

    def __repr__(self) -> str:
        return "NoneConstraint()"


class NoConstraint(Constraint):
    """Accepts every value; the default constraint of a `ConfigField`."""

    def validate(self, value: Any) -> bool:
        """Return True for any value."""
        return True

    def get_error_message(self, field_name: str, value: Any) -> str:
        """Return an empty message; this constraint never fails."""
        return ""

    def __repr__(self) -> str:
        return "NoConstraint()"


class OptionalConstraint(Constraint):
    """Accepts None, and otherwise defers to the wrapped constraint.

    Attributes:
        constraint: The constraint applied to non-None values.
    """

    def __init__(self, constraint: Constraint):
        self.constraint = constraint

    def validate(self, value: Any) -> bool:
        """Return True when the value is None or satisfies the wrapped constraint."""
        if value is None:
            return True
        return self.constraint.validate(value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """Return the wrapped constraint's message."""
        return self.constraint.get_error_message(field_name, value)

    def __repr__(self) -> str:
        return f"OptionalConstraint(constraint={self.constraint})"


class InverseConstraint(Constraint):
    """Accepts exactly the values the wrapped constraint rejects.

    Attributes:
        constraint: The constraint being negated.
    """

    def __init__(self, constraint: Constraint):
        self.constraint = constraint

    def validate(self, value: Any) -> bool:
        """Return True when the value fails the wrapped constraint."""
        return not self.constraint.validate(value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """Name the wrapped constraint the value must not satisfy."""
        return f"Field '{field_name}' must not satisfy the constraint: {self.constraint}, got {value}"

    def __repr__(self) -> str:
        return f"InverseConstraint(constraint={self.constraint})"


class IntersectionConstraints(Constraint):
    """Requires every wrapped constraint to hold; what `&` builds.

    Attributes:
        constraints: The constraints that must all hold.
    """

    def __init__(self, *constraints: Constraint):
        self.constraints = constraints

    def validate(self, value: Any) -> bool:
        """Return True when the value satisfies all wrapped constraints."""
        return all(constraint.validate(value) for constraint in self.constraints)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """List the constraints that must all hold."""
        return f"Field '{field_name}' must satisfy all of the constraints: {self.constraints}, got {value}"

    def __repr__(self) -> str:
        return f"IntersectionConstraints(constraints={self.constraints})"


class UnionConstraints(Constraint):
    """Requires at least one wrapped constraint to hold; what `|` builds.

    Attributes:
        constraints: The constraints of which one must hold.
    """

    def __init__(self, *constraints: Constraint):
        self.constraints = constraints

    def validate(self, value: Any) -> bool:
        """Return True when the value satisfies at least one wrapped constraint."""
        return any(constraint.validate(value) for constraint in self.constraints)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """List the constraints of which one must hold."""
        return f"Field '{field_name}' must satisfy at least one of the constraints: {self.constraints}, got {value}"

    def __repr__(self) -> str:
        return f"UnionConstraints(constraints={self.constraints})"


class CrossConstraint(BaseConstraint):
    """Base class for invariants that span several fields of a `Config` or `ConfigContainer`.

    A sibling of `Constraint`, not a subclass: it receives the whole config, not one value.
    Subclasses implement `validate`, `get_error_message` and the `name` property; `&` and `|`
    compose cross-constraints.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name used in error messages and by `ConfigContainer.remove_cross_constraint`."""
        pass

    @abstractmethod
    def validate(self, config: "Config | ConfigContainer") -> bool:
        """Return True when the invariant holds on the config."""
        pass

    @abstractmethod
    def get_error_message(self, config: "Config | ConfigContainer") -> str:
        """Describe why the config violates the invariant."""
        pass

    def check(self, config: "Config | ConfigContainer") -> None:
        """Validate the config and raise on failure.

        Mirrors `Constraint.check`: a False verdict and an exception escaping `validate` both
        surface as `ConstraintError`.

        Args:
            config: The `Config` or `ConfigContainer` to validate.

        Raises:
            ConstraintError: If the invariant fails or `validate` itself raises.
        """
        try:
            if not self.validate(config):
                raise ConstraintError(self.get_error_message(config))
        except Exception as e:
            if isinstance(e, ConstraintError):
                raise ConstraintError(e.message)
            raise ConstraintError(f"Unexpected error in cross-constraint '{self.name}': {e}")

    def __and__(self, other: "CrossConstraint") -> "CrossIntersectionConstraints":
        if not isinstance(other, CrossConstraint):
            raise TypeError(
                f"Cannot compose CrossConstraint with {type(other).__name__}; " f"use & only between CrossConstraints"
            )
        return CrossIntersectionConstraints(self, other)

    def __or__(self, other: "CrossConstraint") -> "CrossUnionConstraints":
        if not isinstance(other, CrossConstraint):
            raise TypeError(
                f"Cannot compose CrossConstraint with {type(other).__name__}; " f"use | only between CrossConstraints"
            )
        return CrossUnionConstraints(self, other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class LambdaCrossConstraint(CrossConstraint):
    """Cross-constraint whose invariant is a caller-supplied predicate on the config.

    An unnamed instance gets a name derived from its `id()`, so two anonymous constraints never
    collide in error messages or `remove_cross_constraint`.
    """

    def __init__(self, func: Callable, error: str, name: str = ""):
        self._func = func
        self._error = error
        self._name = name or f"LambdaCrossConstraint@{id(self):x}"

    @property
    def name(self) -> str:
        """The given name, or the generated one."""
        return self._name

    def validate(self, config: "Config | ConfigContainer") -> bool:
        """Return the predicate's verdict on the config."""
        return self._func(config)

    def get_error_message(self, config: "Config | ConfigContainer") -> str:
        """Prefix the configured error text with the constraint name."""
        return f"CrossConstraint '{self._name}' failed: {self._error}"


class MutualExclusionConstraint(CrossConstraint):
    """Requires at most one of the named fields to be non-None."""

    def __init__(self, *fields: str, name: str = ""):
        self._fields = fields
        self._name = name or f"mutual_exclusion({', '.join(fields)})"

    @property
    def name(self) -> str:
        """The given name, or one listing the fields."""
        return self._name

    def validate(self, config: "Config | ConfigContainer") -> bool:
        """Return True when at most one of the fields is set."""
        non_none = sum(1 for f in self._fields if getattr(config, f) is not None)
        return non_none <= 1

    def get_error_message(self, config: "Config | ConfigContainer") -> str:
        """List the fields that are set at the same time."""
        set_fields = [f for f in self._fields if getattr(config, f) is not None]
        return (
            f"CrossConstraint '{self._name}' failed: "
            f"at most one of {list(self._fields)} may be set, "
            f"but got: {set_fields}"
        )


class MethodCrossConstraint(CrossConstraint):
    """Cross-constraint built by the metaclass from a method marked with `@cross_validate`.

    Behaves like `LambdaCrossConstraint`; it is a separate class so that decorator-declared
    constraints stay distinguishable from lambdas in `isinstance` checks and `repr` output.
    """

    def __init__(self, func: Callable, error: str, name: str):
        self._func = func
        self._error = error
        self._name = name

    @property
    def name(self) -> str:
        """The name given to `@cross_validate`, or the method name."""
        return self._name

    def validate(self, config: "Config | ConfigContainer") -> bool:
        """Call the wrapped method with the config as `self`."""
        return self._func(config)

    def get_error_message(self, config: "Config | ConfigContainer") -> str:
        """Prefix the configured error text with the constraint name."""
        return f"CrossConstraint '{self._name}' failed: {self._error}"


class CrossIntersectionConstraints(CrossConstraint):
    """Requires every wrapped cross-constraint to hold; what `&` builds."""

    def __init__(self, *constraints: CrossConstraint):
        self._constraints = constraints
        self._name = " & ".join(c.name for c in constraints)

    @property
    def name(self) -> str:
        """The member names joined with ` & `."""
        return self._name

    def validate(self, config: "Config | ConfigContainer") -> bool:
        """Return True when all members hold."""
        return all(c.validate(config) for c in self._constraints)

    def get_error_message(self, config: "Config | ConfigContainer") -> str:
        """List the members that fail."""
        failed = [c.name for c in self._constraints if not c.validate(config)]
        return f"CrossConstraint '{self._name}' failed: " f"the following sub-constraints failed: {failed}"

    def check(self, config: "Config | ConfigContainer") -> None:
        """Validate the config, evaluating each member once.

        The base `check` would call `validate` and then `get_error_message`, running every
        member predicate twice.

        Args:
            config: The `Config` or `ConfigContainer` to validate.

        Raises:
            ConstraintError: If any member fails or raises.
        """
        try:
            failed = [c for c in self._constraints if not c.validate(config)]
            if failed:
                failed_names = [c.name for c in failed]
                raise ConstraintError(
                    f"CrossConstraint '{self._name}' failed: " f"the following sub-constraints failed: {failed_names}"
                )
        except Exception as e:
            if isinstance(e, ConstraintError):
                raise ConstraintError(e.message)
            raise ConstraintError(f"Unexpected error in cross-constraint '{self.name}': {e}")


class CrossUnionConstraints(CrossConstraint):
    """Requires at least one wrapped cross-constraint to hold; what `|` builds."""

    def __init__(self, *constraints: CrossConstraint):
        self._constraints = constraints
        self._name = " | ".join(c.name for c in constraints)

    @property
    def name(self) -> str:
        """The member names joined with ` | `."""
        return self._name

    def validate(self, config: "Config | ConfigContainer") -> bool:
        """Return True when at least one member holds."""
        return any(c.validate(config) for c in self._constraints)

    def get_error_message(self, config: "Config | ConfigContainer") -> str:
        """State that no member held."""
        return f"CrossConstraint '{self._name}' failed: " f"none of the sub-constraints were satisfied"


def cross_validate(error: Optional[str] = None, name: Optional[str] = None) -> Callable:
    """Mark a config method as a cross-field validator.

    The method takes `self` and returns True when the invariant holds; the metaclass wraps it into
    a `MethodCrossConstraint` when the class is created.

    Args:
        error: Text of the failure message; empty when omitted.
        name: Constraint name; the method name when omitted.

    Returns:
        The decorator that marks the method.
    """

    def decorator(method):
        method._is_cross_validator = True
        method._cross_error = error if error is not None else ""
        method._cross_name = name or method.__name__
        return method

    return decorator


def collect_cross_constraints(cls: type) -> list:
    """Gather the cross-constraints declared on a class and its ancestors, base-first.

    Reads `_cross_constraints` and `@cross_validate`-marked methods from each class's own
    `__dict__` along the reversed MRO; attribute lookup is avoided so a diamond ancestor is not
    collected twice. Called by `ConfigMeta` and `ConfigContainerMeta`.

    Args:
        cls: The class whose MRO is walked.

    Returns:
        The cross-constraints, parents before children.
    """
    all_cc: list = []
    for base in reversed(cls.__mro__):
        if "_cross_constraints" in base.__dict__:
            all_cc.extend(base.__dict__["_cross_constraints"])
        for attr_name, attr_val in base.__dict__.items():
            if getattr(attr_val, "_is_cross_validator", False):
                all_cc.append(
                    MethodCrossConstraint(
                        func=attr_val,
                        error=attr_val._cross_error,
                        name=attr_val._cross_name,
                    )
                )
    return all_cc
