"""A comprehensive constraint validation module for data validation and type checking.

This module provides a flexible framework for defining and applying validation constraints
to data fields. It includes various built-in constraint types such as type checking,
range validation, choice validation, and custom lambda constraints. Constraints can
be combined using logical operations (AND, OR) and can be made optional or inverted.

Typical usage example:

    # Create individual constraints
    int_constraint = TypeConstraint(int)
    range_constraint = RangeConstraint(0, 100)

    # Combine constraints
    combined = int_constraint & range_constraint

    # Validate values
    try:
        combined.check("score", 85)  # Valid
        combined.check("score", 150)  # Raises ConstraintError
    except ConstraintError as e:
        print(e.message)
"""

from abc import ABC, abstractmethod
from typing import Callable, Union, List, Any, Optional, Type, get_origin


class BaseConstraint(ABC):
    """Marker ABC shared by Constraint and CrossConstraint.

    NOT a polymorphic interface — callers must know which branch they hold.
    Provides: (1) structural guarantee via @abstractmethod,
    (2) isinstance type marker, (3) shared __repr__ fallback.
    """

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abstractmethod
    def check(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_error_message(self, *args, **kwargs):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Constraint(BaseConstraint):
    """Abstract base class for single-field constraint types.

    This class defines the interface that all single-field constraints must
    implement, including validation logic and error message generation. It also
    provides operator overloading for combining constraints using logical
    AND (&) and OR (|).

    Examples:
        >>> class CustomConstraint(Constraint):
        ...     def validate(self, value):
        ...         return isinstance(value, str) and len(value) > 0
        ...     def get_error_message(self, field_name, value):
        ...         return f"Field '{field_name}' must be a non-empty string"
    """

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validates whether a value satisfies this constraint.

        Args:
            value: The value to validate against this constraint.

        Returns:
            True if the value satisfies the constraint, False otherwise.
        """
        pass

    @abstractmethod
    def get_error_message(self, field_name: str, value: Any) -> str:
        """Generates an error message for constraint validation failures.

        Args:
            field_name: The name of the field being validated.
            value: The value that failed validation.

        Returns:
            A descriptive error message explaining why validation failed.
        """
        pass

    def check(self, field_name: str, value: Any) -> None:
        """Validates a value and raises ConstraintError if validation fails.

        This method performs validation and handles any exceptions that occur
        during the validation process, ensuring consistent error reporting.

        Args:
            field_name: The name of the field being validated.
            value: The value to validate.

        Raises:
            ConstraintError: If the value fails validation or if an unexpected
                error occurs during validation.

        Examples:
            >>> constraint = TypeConstraint(int)
            >>> constraint.check("age", 25)  # No exception
            >>> constraint.check("age", "25")  # Raises ConstraintError
        """
        try:
            if not self.validate(value):
                raise ConstraintError(self.get_error_message(field_name, value))
        except Exception as e:
            if isinstance(e, ConstraintError):
                raise ConstraintError(e.message)
            raise ConstraintError(f"Unexpected error occurred while validating '{field_name}': {e}")

    def __and__(self, other: "Constraint") -> "IntersectionConstraints":
        """Creates an intersection constraint using the & operator.

        Args:
            other: Another constraint to combine with this one.

        Returns:
            An IntersectionConstraints object that requires both constraints
            to be satisfied.

        Examples:
            >>> type_constraint = TypeConstraint(int)
            >>> range_constraint = RangeConstraint(0, 100)
            >>> combined = type_constraint & range_constraint
        """
        return IntersectionConstraints(self, other)

    def __or__(self, other: "Constraint") -> "UnionConstraints":
        """Creates a union constraint using the | operator.

        Args:
            other: Another constraint to combine with this one.

        Returns:
            A UnionConstraints object that requires at least one constraint
            to be satisfied.

        Examples:
            >>> str_constraint = TypeConstraint(str)
            >>> int_constraint = TypeConstraint(int)
            >>> either = str_constraint | int_constraint
        """
        return UnionConstraints(self, other)


class ConstraintError(Exception):
    """Exception raised when constraint validation fails.

    This exception is raised by constraints when validation fails or when
    unexpected errors occur during the validation process.

    Attributes:
        message: A descriptive error message explaining the validation failure.

    Examples:
        >>> try:
        ...     constraint.check("field", invalid_value)
        ... except ConstraintError as e:
        ...     print(f"Validation failed: {e.message}")
    """

    def __init__(self, message: str):
        """Initializes a ConstraintError with the given message.

        Args:
            message: A descriptive error message explaining the validation failure.
        """
        super().__init__(message)
        self.message = message


class TypeConstraint(Constraint):
    """Constraint that validates values against a specific type.

    This constraint ensures that values are instances of the specified type.
    It uses isinstance() for type checking, which supports inheritance.

    Attributes:
        expected_type: The type that values must be instances of.
    """

    def __init__(self, expected_type: Type):
        """Initializes a TypeConstraint with the expected type.

        Args:
            expected_type: The type that values must be instances of.
        """
        self.expected_type = expected_type

    def validate(self, value: Any) -> bool:
        """Validates that the value is an instance of the expected type.

        Args:
            value: The value to validate.

        Returns:
            True if the value is an instance of expected_type, False otherwise.
        """
        # Handle generic types (e.g., List[str], Dict[str, int])
        origin = get_origin(self.expected_type)
        if origin is not None:
            # For generic types, check against the origin type (e.g., list for List[str])
            return isinstance(value, origin)

        # Handle regular types
        return isinstance(value, self.expected_type)

    def get_error_message(self, field_name: str, value: Any) -> str:
        # Handle generic types which don't have __name__ attribute
        if hasattr(self.expected_type, "__name__"):
            type_name = self.expected_type.__name__
        else:
            type_name = str(self.expected_type)
        return f"Field '{field_name}' expected {type_name}, got {type(value).__name__}"

    def __repr__(self) -> str:
        return f"TypeConstraint(expected_type={self.expected_type})"


class RangeConstraint(Constraint):
    """Constraint that validates numeric values within a specified range.

    This constraint ensures that numeric values fall within the specified closed interval [min_val, max_val]. Either boundary can be None to create open-ended ranges.

    Attributes:
        min_val: The minimum allowed value (inclusive), or None for no lower bound.
        max_val: The maximum allowed value (inclusive), or None for no upper bound.
    """

    def __init__(self, min_val: Union[int, float, None], max_val: Union[int, float, None]):
        """Initializes a RangeConstraint with the specified bounds.

        Args:
            min_val: The minimum allowed value (inclusive), or None for no lower bound.
            max_val: The maximum allowed value (inclusive), or None for no upper bound.
        """
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value: Any) -> bool:
        """Validates that the value falls within the specified range.

        Args:
            value: The numeric value to validate.

        Returns:
            True if the value is within the range, False otherwise.
            Returns False if both min_val and max_val are None.
        """
        if self.min_val is not None and self.max_val is not None:
            return self.min_val <= value <= self.max_val
        elif self.min_val is not None:
            return self.min_val <= value
        elif self.max_val is not None:
            return value <= self.max_val
        return False

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must be between {self.min_val} and {self.max_val}, got {value}"

    def __repr__(self) -> str:
        return f"RangeConstraint(min_val={self.min_val}, max_val={self.max_val})"


class ChoiceConstraint(Constraint):
    """Constraint that validates values against a list of allowed choices.

    This constraint ensures that values are members of a predefined set of allowed values. It uses the 'in' operator for membership testing.

    Attributes:
        choices: A list of allowed values.
    """

    def __init__(self, choices: List[Any]):
        """Initializes a ChoiceConstraint with the allowed choices.

        Args:
            choices: A list of values that are considered valid.
        """
        self.choices = choices

    def validate(self, value: Any) -> bool:
        """Validates that the value is one of the allowed choices.

        Args:
            value: The value to validate.

        Returns:
            True if the value is in the choices list, False otherwise.
        """
        return value in self.choices

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must be one of {self.choices}, got {value}"

    def __repr__(self) -> str:
        return f"ChoiceConstraint(choices={self.choices})"


class ChoicesConstraint(Constraint):
    """Constraint that validates every element of a list/tuple against allowed choices.

    The list-analogue of ChoiceConstraint: membership is tested element-wise.
    Empty sequences are accepted (no element violates the constraint).

    Attributes:
        choices: A list of values that every element must be drawn from.
    """

    def __init__(self, choices: List[Any]):
        self.choices = choices

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)):
            return False
        return all(v in self.choices for v in value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        if not isinstance(value, (list, tuple)):
            return f"Field '{field_name}' must be a list/tuple, " f"got {type(value).__name__}"
        bad = [v for v in value if v not in self.choices]
        return f"Field '{field_name}' contains invalid elements {bad}; " f"allowed values are {self.choices}"

    def __repr__(self) -> str:
        return f"ChoicesConstraint(choices={self.choices})"


class EqualityConstraint(Constraint):
    """Constraint that validates values for equality with a specific value.

    This constraint ensures that values are exactly equal to a predetermined expected value using the == operator.

    Attributes:
        expected_value: The exact value that inputs must equal.
    """

    def __init__(self, expected_value: Any):
        """Initializes an EqualityConstraint with the expected value.

        Args:
            expected_value: The exact value that inputs must equal.
        """
        self.expected_value = expected_value

    def validate(self, value: Any) -> bool:
        """Validates that the value equals the expected value.

        Args:
            value: The value to validate.

        Returns:
            True if the value equals the expected value, False otherwise.
        """
        return value == self.expected_value

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must be equal to {self.expected_value}, got {value}"

    def __repr__(self) -> str:
        return f"EqualityConstraint(expected_value={self.expected_value})"


class LengthConstraint(Constraint):
    """Constraint that validates the length of sequences or collections.

    Validates that ``len(value)`` falls within the closed interval
    ``[min_length, max_length]``. Either bound can be ``None`` for an
    open-ended range. Signature mirrors ``RangeConstraint`` so both follow
    the same min/max convention — use ``LengthConstraint(N, N)`` for an
    exact-length check.

    Attributes:
        min_length: Minimum allowed length (inclusive), or None for no lower bound.
        max_length: Maximum allowed length (inclusive), or None for no upper bound.
    """

    def __init__(self, min_length: Optional[int], max_length: Optional[int]):
        """Initializes a LengthConstraint with the specified length bounds.

        Args:
            min_length: Minimum allowed length (inclusive), or None for no lower bound.
            max_length: Maximum allowed length (inclusive), or None for no upper bound.
        """
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any) -> bool:
        """Validates that ``len(value)`` falls within the configured bounds.

        Args:
            value: The object to validate (must support ``len()``).

        Returns:
            True if the length is within the range, False otherwise.
            Returns False if both bounds are None.

        Raises:
            TypeError: If the value doesn't support ``len()`` (handled by check()).
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
    """Constraint that uses a custom function for validation logic.

    This constraint allows for arbitrary validation logic by accepting a
    callable that takes a value and returns a boolean. It also supports
    custom error message generation.

    Attributes:
        func: A callable that takes a value and returns True if valid.
        error_message: Either a static string or a callable that generates error messages.
    """

    def __init__(self, func: Callable[[Any], bool], error_message: Union[str, Callable[[str, Any], str], None] = None):
        """Initializes a LambdaConstraint with validation function and error message.

        Args:
            func: A callable that takes a value and returns True if the value is valid, False otherwise.
            error_message: Either a static error message string, a callable that takes (field_name, value) and returns a string, or None to use a default message.
        """
        self.func = func
        self.error_message = error_message

    def validate(self, value: Any) -> bool:
        """Validates the value using the provided function.

        Args:
            value: The value to validate.

        Returns:
            The result of calling self.func(value).
        """
        return self.func(value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        """Generates an error message for validation failures.

        Args:
            field_name: The name of the field being validated.
            value: The value that failed validation.

        Returns:
            An error message based on the configured error_message attribute.
        """
        if self.error_message:
            if callable(self.error_message):
                return self.error_message(field_name, value)
            return self.error_message
        return f"Field '{field_name}' failed custom validation {self.func}, got {value}"

    def __repr__(self) -> str:
        return f"LambdaConstraint(func={self.func}, error_message={self.error_message})"


class NoneConstraint(Constraint):
    """Constraint that validates values to ensure they are None.

    This constraint only accepts None values and rejects all other values.
    It's useful for fields that must be explicitly unset or null.
    """

    def validate(self, value: Any) -> bool:
        """Validates that the value is None.

        Args:
            value: The value to validate.

        Returns:
            True if the value is None, False otherwise.
        """
        return value is None

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must be None, got {value}"

    def __repr__(self) -> str:
        return "NoneConstraint()"


class NoConstraint(Constraint):
    """Constraint that accepts any value without validation.

    This constraint always returns True for any input value. It's useful as a placeholder or default constraint when no validation is needed.
    """

    def validate(self, value: Any) -> bool:
        """Accepts any value without validation.

        Args:
            value: The value to validate (ignored).

        Returns:
            Always returns True.
        """
        return True

    def get_error_message(self, field_name: str, value: Any) -> str:
        return ""

    def __repr__(self) -> str:
        return "NoConstraint()"


# ============================================================================
# 约束容器 (Constraint Containers)
# ============================================================================


class OptionalConstraint(Constraint):
    """Constraint wrapper that makes another constraint optional.

    This constraint accepts None values or values that satisfy the wrapped constraint. It's useful for making fields optional while still applying validation when values are provided.

    Attributes:
        constraint: The wrapped constraint to apply to non-None values.

    Examples:
        >>> # Optional integer constraint
        >>> optional_int = OptionalConstraint(TypeConstraint(int))
        >>> optional_int.validate(None)  # True
        >>> optional_int.validate(42)  # True
        >>> optional_int.validate("42")  # False

        >>> # Optional range constraint
        >>> optional_range = OptionalConstraint(RangeConstraint(0, 100))
        >>> optional_range.validate(None)  # True
        >>> optional_range.validate(50)  # True
        >>> optional_range.validate(150)  # False
    """

    def __init__(self, constraint: Constraint):
        """Initializes an OptionalConstraint with a wrapped constraint.

        Args:
            constraint: The constraint to apply to non-None values.
        """
        self.constraint = constraint

    def validate(self, value: Any) -> bool:
        """Validates that the value is None or satisfies the wrapped constraint.

        Args:
            value: The value to validate.

        Returns:
            True if the value is None or satisfies the wrapped constraint,
            False otherwise.
        """
        if value is None:
            return True
        return self.constraint.validate(value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        return self.constraint.get_error_message(field_name, value)

    def __repr__(self) -> str:
        return f"OptionalConstraint(constraint={self.constraint})"


class InverseConstraint(Constraint):
    """Constraint wrapper that inverts another constraint's validation logic.

    This constraint accepts values that do NOT satisfy the wrapped constraint.
    It's useful for excluding certain values or types.

    Attributes:
        constraint: The wrapped constraint whose logic will be inverted.

    Examples:
        >>> # Not a string constraint
        >>> not_string = InverseConstraint(TypeConstraint(str))
        >>> not_string.validate(42)  # True
        >>> not_string.validate("hello")  # False

        >>> # Not in range constraint
        >>> not_in_range = InverseConstraint(RangeConstraint(0, 10))
        >>> not_in_range.validate(15)  # True
        >>> not_in_range.validate(5)  # False
    """

    def __init__(self, constraint: Constraint):
        """Initializes an InverseConstraint with a wrapped constraint.

        Args:
            constraint: The constraint whose validation logic will be inverted.
        """
        self.constraint = constraint

    def validate(self, value: Any) -> bool:
        """Validates that the value does NOT satisfy the wrapped constraint.

        Args:
            value: The value to validate.

        Returns:
            True if the value does NOT satisfy the wrapped constraint,
            False otherwise.
        """
        return not self.constraint.validate(value)

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must not satisfy the constraint: {self.constraint}, got {value}"

    def __repr__(self) -> str:
        return f"InverseConstraint(constraint={self.constraint})"


class IntersectionConstraints(Constraint):
    """Constraint that requires ALL of multiple constraints to be satisfied.

    This constraint implements logical AND operation across multiple constraints.
    A value is valid only if it satisfies every single constraint in the collection.

    Attributes:
        constraints: A tuple of constraints that must all be satisfied.

    Examples:
        >>> # Must be int AND in range [0, 100]
        >>> int_and_range = IntersectionConstraints(
        ...     TypeConstraint(int),
        ...     RangeConstraint(0, 100)
        ... )
        >>> int_and_range.validate(50)  # True
        >>> int_and_range.validate(150)  # False (out of range)
        >>> int_and_range.validate("50")  # False (wrong type)

        >>> # Can also be created using & operator
        >>> same_constraint = TypeConstraint(int) & RangeConstraint(0, 100)
    """

    def __init__(self, *constraints: Constraint):
        """Initializes an IntersectionConstraints with multiple constraints.

        Args:
            *constraints: Variable number of constraints that must all be satisfied.
        """
        self.constraints = constraints

    def validate(self, value: Any) -> bool:
        """Validates that the value satisfies ALL constraints.

        Args:
            value: The value to validate.

        Returns:
            True if the value satisfies all constraints, False if any
            constraint is not satisfied.
        """
        return all(constraint.validate(value) for constraint in self.constraints)

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must satisfy all of the constraints: {self.constraints}, got {value}"

    def __repr__(self) -> str:
        return f"IntersectionConstraints(constraints={self.constraints})"


class UnionConstraints(Constraint):
    """Constraint that requires ANY of multiple constraints to be satisfied.

    This constraint implements logical OR operation across multiple constraints.
    A value is valid if it satisfies at least one constraint in the collection.

    Attributes:
        constraints: A tuple of constraints where at least one must be satisfied.

    Examples:
        >>> # Must be string OR int
        >>> string_or_int = UnionConstraints(
        ...     TypeConstraint(str),
        ...     TypeConstraint(int)
        ... )
        >>> string_or_int.validate("hello")  # True
        >>> string_or_int.validate(42)  # True
        >>> string_or_int.validate([1, 2, 3])  # False

        >>> # Can also be created using | operator
        >>> same_constraint = TypeConstraint(str) | TypeConstraint(int)
    """

    def __init__(self, *constraints: Constraint):
        """Initializes a UnionConstraints with multiple constraints.

        Args:
            *constraints: Variable number of constraints where at least one
                         must be satisfied.
        """
        self.constraints = constraints

    def validate(self, value: Any) -> bool:
        """Validates that the value satisfies at least one constraint.

        Args:
            value: The value to validate.

        Returns:
            True if the value satisfies at least one constraint,
            False if no constraints are satisfied.
        """
        return any(constraint.validate(value) for constraint in self.constraints)

    def get_error_message(self, field_name: str, value: Any) -> str:
        return f"Field '{field_name}' must satisfy at least one of the constraints: {self.constraints}, got {value}"

    def __repr__(self) -> str:
        return f"UnionConstraints(constraints={self.constraints})"


# ============================================================================
# Cross-field constraints (CrossConstraint branch)
# ============================================================================


class CrossConstraint(BaseConstraint):
    """Abstract base class for cross-field constraint types.

    Validates invariants that span multiple fields on a Config or
    ConfigContainer instance.  Sibling of Constraint — NOT a subclass.

    Subclasses must implement validate(), get_error_message(), and provide
    a ``name`` property used in error messages and for dynamic removal.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this cross-constraint (used in error messages)."""
        pass

    @abstractmethod
    def validate(self, config) -> bool:
        """Return True if the cross-field invariant holds on *config*."""
        pass

    @abstractmethod
    def get_error_message(self, config) -> str:
        """Return a human-readable error message when validation fails."""
        pass

    def check(self, config) -> None:
        """Validate and raise ConstraintError on failure.

        Mirrors the error-handling pattern of Constraint.check():
        - validate() returning False → ConstraintError with error message
        - unexpected exception → ConstraintError wrapping the original
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
    """General-purpose cross-constraint using a callable predicate."""

    def __init__(self, func: Callable, error: str, name: str = ""):
        self._func = func
        self._error = error
        self._name = name or f"LambdaCrossConstraint@{id(self):x}"

    @property
    def name(self) -> str:
        return self._name

    def validate(self, config) -> bool:
        return self._func(config)

    def get_error_message(self, config) -> str:
        return f"CrossConstraint '{self._name}' failed: {self._error}"


class MutualExclusionConstraint(CrossConstraint):
    """Asserts that at most one of the named fields is non-None."""

    def __init__(self, *fields: str, name: str = ""):
        self._fields = fields
        self._name = name or f"mutual_exclusion({', '.join(fields)})"

    @property
    def name(self) -> str:
        return self._name

    def validate(self, config) -> bool:
        non_none = sum(1 for f in self._fields if getattr(config, f) is not None)
        return non_none <= 1

    def get_error_message(self, config) -> str:
        set_fields = [f for f in self._fields if getattr(config, f) is not None]
        return (
            f"CrossConstraint '{self._name}' failed: "
            f"at most one of {list(self._fields)} may be set, "
            f"but got: {set_fields}"
        )


class MethodCrossConstraint(CrossConstraint):
    """Wraps an unbound function from a class body (produced by @cross_validate).

    Structurally similar to LambdaCrossConstraint, but kept as a separate class
    to distinguish decorator-originated constraints from user-defined lambdas
    in isinstance checks, repr output, and debugging.
    """

    def __init__(self, func: Callable, error: str, name: str):
        self._func = func
        self._error = error
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def validate(self, config) -> bool:
        return self._func(config)

    def get_error_message(self, config) -> str:
        return f"CrossConstraint '{self._name}' failed: {self._error}"


class CrossIntersectionConstraints(CrossConstraint):
    """All wrapped CrossConstraints must pass (logical AND)."""

    def __init__(self, *constraints: CrossConstraint):
        self._constraints = constraints
        self._name = " & ".join(c.name for c in constraints)

    @property
    def name(self) -> str:
        return self._name

    def validate(self, config) -> bool:
        return all(c.validate(config) for c in self._constraints)

    def get_error_message(self, config) -> str:
        failed = [c.name for c in self._constraints if not c.validate(config)]
        return f"CrossConstraint '{self._name}' failed: " f"the following sub-constraints failed: {failed}"

    def check(self, config) -> None:
        """Single-pass check: evaluate each sub-constraint once."""
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
    """At least one wrapped CrossConstraint must pass (logical OR)."""

    def __init__(self, *constraints: CrossConstraint):
        self._constraints = constraints
        self._name = " | ".join(c.name for c in constraints)

    @property
    def name(self) -> str:
        return self._name

    def validate(self, config) -> bool:
        return any(c.validate(config) for c in self._constraints)

    def get_error_message(self, config) -> str:
        return f"CrossConstraint '{self._name}' failed: " f"none of the sub-constraints were satisfied"


# ============================================================================
# @cross_validate decorator
# ============================================================================


def cross_validate(error=None, name=None):
    """Mark a method as a cross-field validator.

    The method should accept ``self`` (the Config/ConfigContainer) and
    return True when the invariant holds.  The metaclass wraps it into
    a MethodCrossConstraint at class-creation time.
    """

    def decorator(method):
        method._is_cross_validator = True
        method._cross_error = error if error is not None else ""
        method._cross_name = name or method.__name__
        return method

    return decorator


# ============================================================================
# Metaclass helper
# ============================================================================


def collect_cross_constraints(cls) -> list:
    """Collect cross-constraints from *cls*'s MRO for metaclass use.

    Walks ``reversed(cls.__mro__)``, reading ``_cross_constraints`` and
    ``@cross_validate``-marked methods from each class's ``__dict__`` only
    (no inherited attribute lookup).  Returns a flat list ordered
    base-first, child-last.

    Called by both ``ConfigMeta.__new__`` and ``ConfigContainerMeta.__new__``.
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
