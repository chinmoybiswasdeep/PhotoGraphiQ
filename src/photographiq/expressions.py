"""Small, non-evaluating expression trees for parameters and classical signals."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_OPS: dict[str, Callable[..., Any]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "pow": operator.pow,
    "neg": operator.neg,
    "sin": math.sin,
    "cos": math.cos,
    "atan2": math.atan2,
    "exp": math.exp,
}


@dataclass(frozen=True)
class Expr:
    """Immutable arithmetic expression tree with explicit parameter and outcome dependencies.

    Args:
        op (object): Op as described by this object’s contract.
        args (object): Args as described by this object’s contract.

    Raises:
        ValueError: Expression must evaluate to a finite real scalar.
    """

    op: str
    args: tuple[Any, ...]

    def __add__(self, x):
        return Expr("add", (self, expression(x)))

    def __radd__(self, x):
        return expression(x) + self

    def __sub__(self, x):
        return Expr("sub", (self, expression(x)))

    def __rsub__(self, x):
        return expression(x) - self

    def __mul__(self, x):
        return Expr("mul", (self, expression(x)))

    def __rmul__(self, x):
        return expression(x) * self

    def __truediv__(self, x):
        return Expr("div", (self, expression(x)))

    def __rtruediv__(self, x):
        return expression(x) / self

    def __pow__(self, x):
        return Expr("pow", (self, expression(x)))

    def __rpow__(self, x):
        return expression(x) ** self

    def __neg__(self):
        return Expr("neg", (self,))

    @property
    def dependencies(self) -> frozenset:
        """Classical/quantum dependencies required before executing this object.

        Returns:
            result (object): Dependency keys or command DAG, according to the owning object.
        """
        if self.op == "outcome":
            return frozenset((self.args[0],))
        return frozenset().union(*(a.dependencies for a in self.args if isinstance(a, Expr)))

    @property
    def parameters(self) -> frozenset[str]:
        """Names of external scalar parameters required by this object.

        Returns:
            result (frozenset): External parameter names.
        """
        if self.op == "parameter":
            return frozenset((self.args[0],))
        return frozenset().union(*(a.parameters for a in self.args if isinstance(a, Expr)))

    def evaluate(self, records: Mapping, parameters: Mapping[str, float]) -> float:
        """Resolve the expression using declared classical records and externally bound parameters.

        Args:
            records (dict): Previously produced classical records.
            parameters (dict): Externally bound scalar parameter values.

        Returns:
            result (float): Finite resolved scalar.

        Raises:
            ValueError: Expression must evaluate to a finite real scalar.
        """
        if self.op == "constant":
            value = self.args[0]
        elif self.op == "parameter":
            value = parameters[self.args[0]]
        elif self.op == "outcome":
            value = records[self.args[0]]
            if len(self.args) == 2:
                value = value[self.args[1]]
        else:
            if self.op not in _OPS:
                raise ValueError(f"Unknown expression operation: {self.op}")
            value = _OPS[self.op](*(a.evaluate(records, parameters) for a in self.args))
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Expression must evaluate to a finite real scalar")
        return value


def Parameter(name: str) -> Expr:
    """An external real scalar bound at execution time."""
    if not isinstance(name, str) or not name:
        raise ValueError("Parameter name must be a nonempty string")
    return Expr("parameter", (name,))


def Outcome(key, component: int | None = None) -> Expr:
    """A prior outcome or signal; vector outcomes require a component."""
    return Expr("outcome", (key,) if component is None else (key, component))


def expression(value) -> Expr:
    """Convert a finite real scalar to a constant expression or preserve an existing Expr.

    Args:
        value (object): Finite real value or supported expression.

    Raises:
        ValueError: Constants must be finite.
    """
    if isinstance(value, Expr):
        return value
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Constants must be finite")
    return Expr("constant", (value,))


@dataclass(frozen=True)
class CallableExpression:
    """Runtime-only callable with mandatory, access-enforced dependencies."""

    function: Callable
    dependencies: frozenset

    def __post_init__(self):
        object.__setattr__(self, "dependencies", frozenset(self.dependencies))

    @property
    def parameters(self):
        """Names of external scalar parameters required by this object.

        Returns:
            result (frozenset): External parameter names.
        """
        return frozenset()

    def evaluate(self, records, parameters):
        # Undeclared access raises KeyError, even if that result exists.
        """Resolve the expression using declared classical records and externally bound parameters.

        Args:
            records (dict): Previously produced classical records.
            parameters (dict): Externally bound scalar parameter values.

        Returns:
            result (float): Finite resolved scalar.
        """
        return expression(self.function({k: records[k] for k in self.dependencies})).evaluate(
            {}, {}
        )


def dependencies(value) -> frozenset:
    """Classical/quantum dependencies required before executing this object.

    Args:
        value (object): Finite real value or supported expression.

    Returns:
        object (object): Dependency keys or command DAG, according to the owning object.
    """
    return getattr(value, "dependencies", frozenset())


def resolve(value, records, parameters) -> float:
    """Evaluate an expression or real constant; reject undeclared Python callables.

    Args:
        value (object): Finite real value or supported expression.
        records (dict): Previously produced classical records.
        parameters (dict): Externally bound scalar parameter values.

    Raises:
        TypeError: Wrap callables in CallableExpression with declared dependencies.
    """
    if isinstance(value, (Expr, CallableExpression)):
        return value.evaluate(records, parameters)
    if callable(value):
        raise TypeError("Wrap callables in CallableExpression with declared dependencies")
    return expression(value).evaluate({}, {})


def sin(value):
    """Build the sine of an expression in radians.

    Args:
        value (object): Finite real value or supported expression.
    """
    return Expr("sin", (expression(value),))


def cos(value):
    """Build the cosine of an expression in radians.

    Args:
        value (object): Finite real value or supported expression.
    """
    return Expr("cos", (expression(value),))


def exp(value):
    """Build the exponential of an expression.

    Args:
        value (object): Finite real value or supported expression.
    """
    return Expr("exp", (expression(value),))


def atan2(y, x):
    """Build a quadrant-aware angle expression from y and x.

    Args:
        y (object): Y as described by this object’s contract.
        x (object): X as described by this object’s contract.
    """
    return Expr("atan2", (expression(y), expression(x)))
