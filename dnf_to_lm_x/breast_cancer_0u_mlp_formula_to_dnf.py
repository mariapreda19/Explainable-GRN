from __future__ import annotations

import os
import random
import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Union, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

U_VALUE = 1.0
U = Fraction(1)
ZERO = Fraction(0)


def set_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def F(x: Union[int, float, str, Fraction]) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, float):
        return Fraction(str(float(x)))
    return Fraction(x)


def frac_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{float(q):.10f}".rstrip("0").rstrip(".")


class Formula:
    def to_string(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_string()


class Bottom(Formula):
    def to_string(self) -> str:
        return "⊥"


class Top(Formula):
    def to_string(self) -> str:
        return "⊤"


class SymbolRef(Formula):
    def __init__(self, name: str):
        self.name = name

    def to_string(self) -> str:
        return self.name


class Diamond(Formula):
    def __init__(self, coeff: float, child: Formula):
        self.coeff = float(coeff)
        self.child = child

    def to_string(self) -> str:
        c = f"{self.coeff:.6f}".rstrip("0").rstrip(".")
        if c == "-0":
            c = "0"
        return f"◇_{c}({self.child})"


class Not(Formula):
    def __init__(self, child: Formula):
        self.child = child

    def to_string(self) -> str:
        return f"¬({self.child})"


class Oplus(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def to_string(self) -> str:
        return f"({self.left} ⊕ {self.right})"


class Odot(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def to_string(self) -> str:
        return f"({self.left} ⊙ {self.right})"


class And(Formula):
    def __init__(self, *children: Formula):
        self.children = list(children)

    def to_string(self) -> str:
        if len(self.children) == 0:
            return "⊤"
        if len(self.children) == 1:
            return str(self.children[0])
        return "(" + " ∧ ".join(str(c) for c in self.children) + ")"


class Or(Formula):
    def __init__(self, *children: Formula):
        self.children = list(children)

    def to_string(self) -> str:
        if len(self.children) == 0:
            return "⊥"
        if len(self.children) == 1:
            return str(self.children[0])
        return "(" + " ∨ ".join(str(c) for c in self.children) + ")"


class ClipAffine(Formula):

    def __init__(self, terms, bias: float = 0.0, u: float = U_VALUE):
        self.terms = [(float(coeff), child) for coeff, child in terms]
        self.bias = float(bias)
        self.u = float(u)

    def to_string(self) -> str:
        parts = []
        for coeff, child in self.terms:
            if abs(coeff) < 1e-12:
                continue
            c = f"{coeff:.6f}".rstrip("0").rstrip(".")
            if c == "-0":
                continue
            parts.append(f"{c}·{child}")
        if abs(self.bias) >= 1e-12:
            b = f"{self.bias:.6f}".rstrip("0").rstrip(".")
            if b != "-0":
                parts.append(b)
        inner = " + ".join(parts) if parts else "0"
        u_txt = f"{self.u:.6f}".rstrip("0").rstrip(".")
        return f"clip_[0,{u_txt}]({inner})"


def _weights_to_text(weights, max_items: int = 8) -> str:
    ws = list(weights)
    shown = ws[:max_items]
    body = ", ".join(f"{float(w):.4f}" for w in shown)
    if len(ws) > max_items:
        body += ", ..."
    return body


class SoftMinFormula(Formula):

    def __init__(self, children, weights, mu: float, u: float = U_VALUE, eps: float = 1e-8):
        self.children = list(children)
        self.weights = [float(w) for w in weights]
        self.mu = float(mu)
        self.u = float(u)
        self.eps = float(eps)

    def to_string(self) -> str:
        args = ", ".join(str(c) for c in self.children)
        return f"smin_mu={self.mu:g}[w={_weights_to_text(self.weights)}]({args})"


class SoftMaxFormula(Formula):

    def __init__(self, children, weights, mu: float, u: float = U_VALUE, eps: float = 1e-8):
        self.children = list(children)
        self.weights = [float(w) for w in weights]
        self.mu = float(mu)
        self.u = float(u)
        self.eps = float(eps)

    def to_string(self) -> str:
        args = ", ".join(str(c) for c in self.children)
        return f"smax_mu={self.mu:g}[w={_weights_to_text(self.weights)}]({args})"


def _logsumexp_np(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    m = float(np.max(values))
    return float(m + np.log(np.sum(np.exp(values - m))))


def _formula_key(formula: Formula) -> str:
    return formula.to_string()


def _merge_duplicate_soft_children(children: Sequence[Formula], weights: Sequence[float]):
    merged = {}
    order = []
    for child, weight in zip(children, weights):
        key = _formula_key(child)
        if key not in merged:
            merged[key] = [child, 0.0]
            order.append(key)
        merged[key][1] += float(weight)
    out_children = [merged[k][0] for k in order]
    out_weights = [merged[k][1] for k in order]
    return out_children, out_weights


def _select_compact_gate_indices(gate_row, gate_threshold: float = 0.10, top_k: Optional[int] = 2,
                                min_keep: int = 1):
    g = np.asarray(gate_row, dtype=float)
    if g.ndim != 1:
        raise ValueError("gate_row must be one-dimensional")

    order = list(np.argsort(-g))
    if top_k is None:
        idxs = [int(i) for i in order if g[i] >= gate_threshold]
    else:
        idxs = [int(i) for i in order[:max(int(top_k), int(min_keep))] if g[i] >= gate_threshold]

    if len(idxs) < min_keep:
        for i in order:
            if int(i) not in idxs:
                idxs.append(int(i))
            if len(idxs) >= min_keep:
                break

    return sorted(idxs)


def compact_formula_length(symbolic_defs) -> int:
    return sum(len(name) + 4 + len(str(formula)) for name, formula in symbolic_defs)


def expanded_formula_length(finals: Dict[str, Formula]) -> int:
    return sum(len(name) + 4 + len(str(formula)) for name, formula in finals.items())


def print_symbolic_definitions(symbolic_defs, title="SYMBOLIC DEFINITIONS", only_outputs=False):
    print(f"\n{title}")
    rows = symbolic_defs
    if only_outputs:
        rows = [(name, formula) for name, formula in symbolic_defs if name.startswith("y")]
    for name, formula in rows:
        print(f"{name} := {formula}")


def print_final_output_formulas(finals: Dict[str, Formula], title="FINAL OUTPUT FORMULAS") -> None:
    print(f"\n{title}")
    for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
        print(f"{name} := {finals[name]}")


def write_final_output_formulas(f, finals: Dict[str, Formula], title="Final output formulas") -> None:
    f.write(f"\n{title}\n")
    for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
        f.write(f"{name} := {finals[name]}\n")


def print_binary_decision_rule(finals: Dict[str, Formula], target_names=None) -> None:
    if "y1" not in finals or "y2" not in finals:
        return
    negative = "class 0" if target_names is None else str(target_names[0])
    positive = "class 1" if target_names is None else str(target_names[1])
    print("\nCompact decision rule")
    print(f"Predict {positive} iff y2 >= y1; otherwise predict {negative}.")
    print(f"y1 = {finals['y1']}")
    print(f"y2 = {finals['y2']}")


def write_binary_decision_rule(f, finals: Dict[str, Formula], target_names=None) -> None:
    if "y1" not in finals or "y2" not in finals:
        return
    negative = "class 0" if target_names is None else str(target_names[0])
    positive = "class 1" if target_names is None else str(target_names[1])
    f.write("\nCompact decision rule\n")
    f.write(f"Predict {positive} iff y2 >= y1; otherwise predict {negative}.\n")
    f.write(f"y1 = {finals['y1']}\n")
    f.write(f"y2 = {finals['y2']}\n")


def simplify(formula: Formula) -> Formula:
    if isinstance(formula, (Bottom, Top, SymbolRef)):
        return formula

    if isinstance(formula, Diamond):
        child = simplify(formula.child)
        if abs(formula.coeff) < 1e-12:
            return Bottom()
        return Diamond(formula.coeff, child)

    if isinstance(formula, Not):
        child = simplify(formula.child)
        if isinstance(child, Bottom):
            return Top()
        if isinstance(child, Top):
            return Bottom()
        if isinstance(child, Not):
            return simplify(child.child)
        return Not(child)

    if isinstance(formula, Oplus):
        left = simplify(formula.left)
        right = simplify(formula.right)
        if isinstance(left, Bottom):
            return right
        if isinstance(right, Bottom):
            return left
        return Oplus(left, right)

    if isinstance(formula, Odot):
        left = simplify(formula.left)
        right = simplify(formula.right)
        if isinstance(left, Bottom) or isinstance(right, Bottom):
            return Bottom()
        if isinstance(left, Top):
            return right
        if isinstance(right, Top):
            return left
        return Odot(left, right)

    if isinstance(formula, And):
        children = []
        for c in formula.children:
            c = simplify(c)
            if isinstance(c, Bottom):
                return Bottom()
            if not isinstance(c, Top):
                children.append(c)
        if len(children) == 0:
            return Top()
        if len(children) == 1:
            return children[0]
        return And(*children)

    if isinstance(formula, Or):
        children = []
        for c in formula.children:
            c = simplify(c)
            if isinstance(c, Top):
                return Top()
            if not isinstance(c, Bottom):
                children.append(c)
        if len(children) == 0:
            return Bottom()
        if len(children) == 1:
            return children[0]
        return Or(*children)

    if isinstance(formula, ClipAffine):
        terms = [(coeff, simplify(child)) for coeff, child in formula.terms if abs(coeff) >= 1e-12]
        if len(terms) == 0:
            value = min(float(formula.u), max(0.0, float(formula.bias)))
            if abs(value) < 1e-12:
                return Bottom()
            if abs(value - float(formula.u)) < 1e-12:
                return Top()
            return Diamond(value / float(formula.u), Top())
        return ClipAffine(terms, bias=formula.bias, u=formula.u)

    if isinstance(formula, SoftMinFormula):
        children = [simplify(c) for c in formula.children]
        children, weights = _merge_duplicate_soft_children(children, formula.weights)
        if len(children) == 1:
            return children[0]
        return SoftMinFormula(children, weights, mu=formula.mu, u=formula.u, eps=formula.eps)

    if isinstance(formula, SoftMaxFormula):
        children = [simplify(c) for c in formula.children]
        children, weights = _merge_duplicate_soft_children(children, formula.weights)
        if len(children) == 1:
            return children[0]
        return SoftMaxFormula(children, weights, mu=formula.mu, u=formula.u, eps=formula.eps)

    return formula


def linear_combination(L, eps: float = 1e-12) -> Formula:
    if len(L) == 0:
        return Bottom()

    if all(r <= eps for (r, _) in L):
        return Bottom()

    k = None
    for idx, (r, _) in enumerate(L):
        if r > eps:
            k = idx
            break

    if k is None:
        return Bottom()

    r_k, item_k = L[k]

    if item_k == 0:
        psi = Diamond(r_k, Top())
    else:
        psi = Diamond(r_k, item_k)

    if len(L) == 1:
        return simplify(psi)

    Lnew = L[:k] + L[k + 1:]
    neg_Lnew = [(-r_j, item_j) for (r_j, item_j) in Lnew]

    phi = linear_combination(Lnew, eps=eps)
    chi = linear_combination(neg_Lnew, eps=eps)

    return simplify(Odot(Oplus(phi, psi), Not(chi)))


def build_definition_map(symbolic_defs) -> Dict[str, Formula]:
    return {name: formula for name, formula in symbolic_defs}


def expand_formula_all(formula: Formula, def_map: Dict[str, Formula], seen: Optional[set] = None) -> Formula:
    if seen is None:
        seen = set()

    if isinstance(formula, SymbolRef):
        if formula.name not in def_map:
            return formula
        if formula.name in seen:
            raise ValueError(f"Cyclic symbolic reference: {formula.name}")
        return expand_formula_all(def_map[formula.name], def_map, seen | {formula.name})

    if isinstance(formula, (Bottom, Top)):
        return formula

    if isinstance(formula, Diamond):
        return simplify(Diamond(formula.coeff, expand_formula_all(formula.child, def_map, seen)))

    if isinstance(formula, Not):
        return simplify(Not(expand_formula_all(formula.child, def_map, seen)))

    if isinstance(formula, Oplus):
        return simplify(Oplus(
            expand_formula_all(formula.left, def_map, seen),
            expand_formula_all(formula.right, def_map, seen),
        ))

    if isinstance(formula, Odot):
        return simplify(Odot(
            expand_formula_all(formula.left, def_map, seen),
            expand_formula_all(formula.right, def_map, seen),
        ))

    if isinstance(formula, And):
        return simplify(And(*(expand_formula_all(c, def_map, seen) for c in formula.children)))

    if isinstance(formula, Or):
        return simplify(Or(*(expand_formula_all(c, def_map, seen) for c in formula.children)))

    if isinstance(formula, ClipAffine):
        return simplify(ClipAffine(
            [(coeff, expand_formula_all(child, def_map, seen)) for coeff, child in formula.terms],
            bias=formula.bias,
            u=formula.u,
        ))

    if isinstance(formula, SoftMinFormula):
        return simplify(SoftMinFormula(
            [expand_formula_all(c, def_map, seen) for c in formula.children],
            formula.weights,
            mu=formula.mu,
            u=formula.u,
            eps=formula.eps,
        ))

    if isinstance(formula, SoftMaxFormula):
        return simplify(SoftMaxFormula(
            [expand_formula_all(c, def_map, seen) for c in formula.children],
            formula.weights,
            mu=formula.mu,
            u=formula.u,
            eps=formula.eps,
        ))

    raise TypeError(type(formula))


def get_fully_expanded_formulas(symbolic_defs) -> Dict[str, Formula]:
    def_map = build_definition_map(symbolic_defs)
    finals = {}
    for name, formula in symbolic_defs:
        if name.startswith("y"):
            finals[name] = simplify(expand_formula_all(formula, def_map))
    return finals


def parse_optional_int_env(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "off", "false", "exact"}:
        return None
    return int(raw)


def _round_float_for_formula(x: float, decimals: Optional[int]) -> float:
    x = float(x)
    if decimals is None:
        return x
    y = round(x, int(decimals))
    if abs(y) < 1e-12:
        return 0.0
    return float(y)


def quantize_formula_for_dnf(formula: Formula, decimals: Optional[int] = 0, coeff_abs_threshold: float = 0.0) -> Formula:
    if isinstance(formula, (Bottom, Top, SymbolRef)):
        return formula

    if isinstance(formula, ClipAffine):
        new_terms = []
        for coeff, child in formula.terms:
            q = _round_float_for_formula(coeff, decimals)
            if abs(q) <= float(coeff_abs_threshold):
                continue
            new_terms.append((q, quantize_formula_for_dnf(child, decimals, coeff_abs_threshold)))
        new_bias = _round_float_for_formula(formula.bias, decimals)
        return simplify(ClipAffine(new_terms, bias=new_bias, u=formula.u))

    if isinstance(formula, Diamond):
        return simplify(Diamond(
            _round_float_for_formula(formula.coeff, decimals),
            quantize_formula_for_dnf(formula.child, decimals, coeff_abs_threshold),
        ))

    if isinstance(formula, Not):
        return simplify(Not(quantize_formula_for_dnf(formula.child, decimals, coeff_abs_threshold)))

    if isinstance(formula, Oplus):
        return simplify(Oplus(
            quantize_formula_for_dnf(formula.left, decimals, coeff_abs_threshold),
            quantize_formula_for_dnf(formula.right, decimals, coeff_abs_threshold),
        ))

    if isinstance(formula, Odot):
        return simplify(Odot(
            quantize_formula_for_dnf(formula.left, decimals, coeff_abs_threshold),
            quantize_formula_for_dnf(formula.right, decimals, coeff_abs_threshold),
        ))

    if isinstance(formula, And):
        return simplify(And(*(quantize_formula_for_dnf(c, decimals, coeff_abs_threshold) for c in formula.children)))

    if isinstance(formula, Or):
        return simplify(Or(*(quantize_formula_for_dnf(c, decimals, coeff_abs_threshold) for c in formula.children)))

    if isinstance(formula, (SoftMinFormula, SoftMaxFormula)):
        raise TypeError("DNF approximation requires piecewise-affine formulas; use compact_top_k=1.")

    raise TypeError(type(formula))


def quantize_finals_for_dnf(
        finals: Dict[str, Formula],
        decimals: Optional[int] = 0,
        coeff_abs_threshold: float = 0.0,
) -> Dict[str, Formula]:
    return {
        name: simplify(quantize_formula_for_dnf(formula, decimals, coeff_abs_threshold))
        for name, formula in finals.items()
    }


def eval_formula(formula: Formula, env: Dict[str, float], u: float = U_VALUE) -> float:
    if isinstance(formula, Bottom):
        return 0.0
    if isinstance(formula, Top):
        return float(u)
    if isinstance(formula, SymbolRef):
        return float(env[formula.name])
    if isinstance(formula, Diamond):
        return float(formula.coeff) * eval_formula(formula.child, env, u=u)
    if isinstance(formula, Not):
        return float(u) - eval_formula(formula.child, env, u=u)
    if isinstance(formula, Oplus):
        return min(float(u), eval_formula(formula.left, env, u=u) + eval_formula(formula.right, env, u=u))
    if isinstance(formula, Odot):
        return max(0.0, eval_formula(formula.left, env, u=u) + eval_formula(formula.right, env, u=u) - float(u))
    if isinstance(formula, And):
        if not formula.children:
            return float(u)
        return min(eval_formula(c, env, u=u) for c in formula.children)
    if isinstance(formula, Or):
        if not formula.children:
            return 0.0
        return max(eval_formula(c, env, u=u) for c in formula.children)
    if isinstance(formula, ClipAffine):
        value = formula.bias + sum(coeff * eval_formula(child, env, u=u) for coeff, child in formula.terms)
        return min(float(formula.u), max(0.0, float(value)))
    if isinstance(formula, SoftMinFormula):
        xs = np.array([eval_formula(c, env, u=u) for c in formula.children], dtype=float)
        w = np.asarray(formula.weights, dtype=float).clip(min=formula.eps)
        logw = np.log(w)
        y = -(_logsumexp_np(-formula.mu * xs + logw) - _logsumexp_np(logw)) / formula.mu
        return min(float(formula.u), max(0.0, float(y)))
    if isinstance(formula, SoftMaxFormula):
        xs = np.array([eval_formula(c, env, u=u) for c in formula.children], dtype=float)
        w = np.asarray(formula.weights, dtype=float).clip(min=formula.eps)
        logw = np.log(w)
        y = (_logsumexp_np(formula.mu * xs + logw) - _logsumexp_np(logw)) / formula.mu
        return min(float(formula.u), max(0.0, float(y)))
    raise TypeError(type(formula))


def eval_expr(expr, env, u=U_VALUE):
    if isinstance(expr, Formula):
        env_float = {k: float(v) for k, v in env.items()}
        return eval_formula(expr, env_float, u=float(u))

    u_frac = F(u)

    if "Var" in globals() and isinstance(expr, Var):
        return env[expr.name]

    if "Const" in globals() and isinstance(expr, Const):
        return expr.value

    if "Neg" in globals() and isinstance(expr, Neg):
        return u_frac - eval_expr(expr.sub, env, u=u_frac)

    if "OPlusExpr" in globals() and isinstance(expr, OPlusExpr):
        return min(u_frac, eval_expr(expr.left, env, u=u_frac) + eval_expr(expr.right, env, u=u_frac))

    if "ODotExpr" in globals() and isinstance(expr, ODotExpr):
        return max(ZERO, eval_expr(expr.left, env, u=u_frac) + eval_expr(expr.right, env, u=u_frac) - u_frac)

    if "SMul" in globals() and isinstance(expr, SMul):
        return expr.scalar * eval_expr(expr.sub, env, u=u_frac)

    if "Add" in globals() and isinstance(expr, Add):
        return eval_expr(expr.left, env, u=u_frac) + eval_expr(expr.right, env, u=u_frac)

    if "Sub" in globals() and isinstance(expr, Sub):
        return eval_expr(expr.left, env, u=u_frac) - eval_expr(expr.right, env, u=u_frac)

    if "Mul" in globals() and isinstance(expr, Mul):
        scalar = getattr(expr, "scalar", None)
        sub = getattr(expr, "sub", None)
        if scalar is not None and sub is not None:
            return scalar * eval_expr(sub, env, u=u_frac)
        return eval_expr(expr.left, env, u=u_frac) * eval_expr(expr.right, env, u=u_frac)

    if "Min" in globals() and isinstance(expr, Min):
        return min(eval_expr(expr.left, env, u=u_frac), eval_expr(expr.right, env, u=u_frac))

    if "Max" in globals() and isinstance(expr, Max):
        return max(eval_expr(expr.left, env, u=u_frac), eval_expr(expr.right, env, u=u_frac))

    raise TypeError(type(expr))



@dataclass
class DNFAffine:
    coef: Dict[str, Fraction] = field(default_factory=dict)
    const: Fraction = Fraction(0, 1)

    def __add__(self, other: "DNFAffine") -> "DNFAffine":
        keys = set(self.coef) | set(other.coef)
        out: Dict[str, Fraction] = {}
        for k in keys:
            v = self.coef.get(k, ZERO) + other.coef.get(k, ZERO)
            if v != 0:
                out[k] = v
        return DNFAffine(out, self.const + other.const)

    def __sub__(self, other: "DNFAffine") -> "DNFAffine":
        keys = set(self.coef) | set(other.coef)
        out: Dict[str, Fraction] = {}
        for k in keys:
            v = self.coef.get(k, ZERO) - other.coef.get(k, ZERO)
            if v != 0:
                out[k] = v
        return DNFAffine(out, self.const - other.const)

    def __mul__(self, scalar: Union[int, float, str, Fraction]) -> "DNFAffine":
        s = F(scalar)
        out: Dict[str, Fraction] = {}
        for k, v in self.coef.items():
            w = v * s
            if w != 0:
                out[k] = w
        return DNFAffine(out, self.const * s)

    __rmul__ = __mul__

    def key(self):
        return (tuple(sorted(self.coef.items())), self.const)

    def eval(self, point: Dict[str, Fraction]) -> Fraction:
        total = self.const
        for var, a in self.coef.items():
            total += a * point[var]
        return total

    def is_const(self, c: Optional[Fraction] = None) -> bool:
        if self.coef:
            return False
        return True if c is None else self.const == c

    def cube_minmax(self, variables: List[str]) -> Tuple[Fraction, Fraction]:
        mn = self.const
        mx = self.const
        for var in variables:
            a = self.coef.get(var, ZERO)
            if a >= 0:
                mx += a * U
            else:
                mn += a * U
        return mn, mx

    def pretty(self) -> str:
        parts: List[str] = []
        for var in sorted(self.coef):
            a = self.coef[var]
            if a == 1:
                parts.append(var)
            elif a == -1:
                parts.append(f"-{var}")
            else:
                parts.append(f"{frac_str(a)}·{var}")

        if self.const != 0 or not parts:
            parts.append(frac_str(self.const))

        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s


@dataclass
class DNFInequality:
    expr: DNFAffine

    @staticmethod
    def le(lhs: DNFAffine, rhs: DNFAffine) -> "DNFInequality":
        return DNFInequality(lhs - rhs)

    @staticmethod
    def ge(lhs: DNFAffine, rhs: DNFAffine) -> "DNFInequality":
        return DNFInequality(rhs - lhs)

    def key(self):
        return self.expr.key()


@dataclass
class DNFRegion:
    constraints: List[DNFInequality] = field(default_factory=list)

    def add(self, ineq: DNFInequality) -> "DNFRegion":
        return DNFRegion(self.constraints + [ineq])

    def extend(self, other: "DNFRegion") -> "DNFRegion":
        return DNFRegion(self.constraints + other.constraints)

    def dedup(self) -> "DNFRegion":
        seen = set()
        out: List[DNFInequality] = []
        for c in self.constraints:
            k = c.key()
            if k not in seen:
                seen.add(k)
                out.append(c)
        return DNFRegion(out)


@dataclass
class DNFPiece:
    region: DNFRegion
    affine: DNFAffine


@dataclass
class DNFClipFactor:
    affine: DNFAffine

    def key(self):
        return self.affine.key()

    def pretty(self, variables: List[str], drop_lower_clip: bool = False) -> str:
        mn, mx = self.affine.cube_minmax(variables)
        a = self.affine.pretty()
        u_text = frac_str(U)
        need_lower = (mn < ZERO) and (not drop_lower_clip)
        need_upper = mx > U

        if not need_lower and not need_upper:
            return a
        if need_lower and not need_upper:
            return f"({a} ∨ 0)"
        if not need_lower and need_upper:
            return f"({a} ∧ {u_text})"
        return f"(({a} ∨ 0) ∧ {u_text})"


@dataclass
class DNFMeetTerm:
    factors: List[DNFClipFactor]

    def pretty(self, variables: List[str], drop_lower_clip: bool = False) -> str:
        if not self.factors:
            return frac_str(U)

        has_nonnegative_factor = any(f.affine.cube_minmax(variables)[0] >= ZERO for f in self.factors)
        texts = [
            f.pretty(variables, drop_lower_clip=(drop_lower_clip or has_nonnegative_factor))
            for f in self.factors
        ]
        return " ∧ ".join(texts)


@dataclass
class DNForm:
    clauses: List[DNFMeetTerm]

    def pretty(self, variables: List[str]) -> str:
        if not self.clauses:
            return "0"

        drop_lower_clip = len(self.clauses) > 1
        clause_texts: List[str] = []
        for c in self.clauses:
            txt = c.pretty(variables, drop_lower_clip=drop_lower_clip)
            if txt != "0":
                clause_texts.append(txt)

        if not clause_texts:
            return "0"
        return " ∨ ".join(clause_texts)


def _dnf_collect_variables(formula: Formula) -> List[str]:
    out = set()

    def rec(f: Formula) -> None:
        if isinstance(f, SymbolRef):
            out.add(f.name)
        elif isinstance(f, (Bottom, Top)):
            return
        elif isinstance(f, Diamond):
            rec(f.child)
        elif isinstance(f, Not):
            rec(f.child)
        elif isinstance(f, (Oplus, Odot)):
            rec(f.left)
            rec(f.right)
        elif isinstance(f, (And, Or)):
            for c in f.children:
                rec(c)
        elif isinstance(f, ClipAffine):
            for _, child in f.terms:
                rec(child)
        elif isinstance(f, (SoftMinFormula, SoftMaxFormula)):
            raise ValueError(
                "DNF conversion is only available for piecewise-affine formulas; soft-min/max must be compacted or crispified first.")
        else:
            raise TypeError(type(f))

    rec(formula)
    return sorted(out)


def _dnf_dedup_pieces(pieces: List[DNFPiece]) -> List[DNFPiece]:
    seen = set()
    out: List[DNFPiece] = []
    for p in pieces:
        region = p.region.dedup()
        key = (tuple(c.key() for c in region.constraints), p.affine.key())
        if key not in seen:
            seen.add(key)
            out.append(DNFPiece(region, p.affine))
    return out


def _dnf_piecewise_affine(formula: Formula) -> List[DNFPiece]:
    top = DNFAffine({}, U)
    zero = DNFAffine({}, ZERO)

    def combine_product(left: List[DNFPiece], right: List[DNFPiece]):
        for p, q in itertools.product(left, right):
            yield p, q, p.region.extend(q.region)

    def rec(f: Formula) -> List[DNFPiece]:
        if isinstance(f, Bottom):
            return [DNFPiece(DNFRegion(), zero)]
        if isinstance(f, Top):
            return [DNFPiece(DNFRegion(), top)]
        if isinstance(f, SymbolRef):
            return [DNFPiece(DNFRegion(), DNFAffine({f.name: F(1)}, ZERO))]
        if isinstance(f, Diamond):
            return [DNFPiece(p.region, F(f.coeff) * p.affine) for p in rec(f.child)]
        if isinstance(f, Not):
            return [DNFPiece(p.region, top - p.affine) for p in rec(f.child)]

        if isinstance(f, Oplus):
            out: List[DNFPiece] = []
            for p, q, base in combine_product(rec(f.left), rec(f.right)):
                s = p.affine + q.affine
                out.append(DNFPiece(base.add(DNFInequality.le(s, top)), s))
                out.append(DNFPiece(base.add(DNFInequality.ge(s, top)), top))
            return _dnf_dedup_pieces(out)

        if isinstance(f, Odot):
            out: List[DNFPiece] = []
            for p, q, base in combine_product(rec(f.left), rec(f.right)):
                s = p.affine + q.affine
                out.append(DNFPiece(base.add(DNFInequality.le(s, top)), zero))
                out.append(DNFPiece(base.add(DNFInequality.ge(s, top)), s - top))
            return _dnf_dedup_pieces(out)

        if isinstance(f, And):
            if not f.children:
                return [DNFPiece(DNFRegion(), top)]
            pieces = rec(f.children[0])
            for child in f.children[1:]:
                child_pieces = rec(child)
                out: List[DNFPiece] = []
                for p, q, base in combine_product(pieces, child_pieces):
                    diff = p.affine - q.affine
                    out.append(DNFPiece(base.add(DNFInequality.le(diff, zero)), p.affine))
                    out.append(DNFPiece(base.add(DNFInequality.ge(diff, zero)), q.affine))
                pieces = _dnf_dedup_pieces(out)
            return pieces

        if isinstance(f, Or):
            if not f.children:
                return [DNFPiece(DNFRegion(), zero)]
            pieces = rec(f.children[0])
            for child in f.children[1:]:
                child_pieces = rec(child)
                out: List[DNFPiece] = []
                for p, q, base in combine_product(pieces, child_pieces):
                    diff = p.affine - q.affine
                    out.append(DNFPiece(base.add(DNFInequality.ge(diff, zero)), p.affine))
                    out.append(DNFPiece(base.add(DNFInequality.le(diff, zero)), q.affine))
                pieces = _dnf_dedup_pieces(out)
            return pieces

        if isinstance(f, ClipAffine):
            child_piece_lists = [rec(child) for _, child in f.terms]
            if not child_piece_lists:
                value = min(U, max(ZERO, F(f.bias)))
                return [DNFPiece(DNFRegion(), DNFAffine({}, value))]

            out: List[DNFPiece] = []
            for combo in itertools.product(*child_piece_lists):
                region = DNFRegion()
                affine = DNFAffine({}, F(f.bias))
                for (coeff, _), child_piece in zip(f.terms, combo):
                    region = region.extend(child_piece.region)
                    affine = affine + F(coeff) * child_piece.affine

                out.append(DNFPiece(region.add(DNFInequality.le(affine, zero)), zero))
                mid_region = region.add(DNFInequality.ge(affine, zero)).add(DNFInequality.le(affine, top))
                out.append(DNFPiece(mid_region, affine))
                out.append(DNFPiece(region.add(DNFInequality.ge(affine, top)), top))
            return _dnf_dedup_pieces(out)

        if isinstance(f, (SoftMinFormula, SoftMaxFormula)):
            raise ValueError(
                "DNF conversion is only available for piecewise-affine formulas; soft-min/max must be compacted or crispified first.")

        raise TypeError(type(f))

    return _dnf_dedup_pieces(rec(formula))


def _dnf_cube_constraints(variables: List[str]) -> List[DNFInequality]:
    out: List[DNFInequality] = []
    for v in variables:
        out.append(DNFInequality.le(DNFAffine({v: F(1)}, ZERO), DNFAffine({}, U)))
        out.append(DNFInequality.le(DNFAffine({v: F(-1)}, ZERO), DNFAffine({}, ZERO)))
    return out


def _dnf_region_with_cube(region: DNFRegion, variables: List[str]) -> DNFRegion:
    return DNFRegion(region.constraints + _dnf_cube_constraints(variables)).dedup()


def _dnf_solve_linear_system_fraction(A: List[List[Fraction]], b: List[Fraction]):
    n = len(A)
    if n == 0:
        return []

    M = [row[:] + [rhs] for row, rhs in zip(A, b)]
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return None
        M[row], M[pivot] = M[pivot], M[row]
        piv = M[row][col]
        for j in range(col, n + 1):
            M[row][j] /= piv
        for r in range(n):
            if r == row:
                continue
            factor = M[r][col]
            if factor != 0:
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[row][j]
        row += 1
    return [M[i][n] for i in range(n)]


def _dnf_satisfies_region(point: Dict[str, Fraction], region: DNFRegion) -> bool:
    return all(c.expr.eval(point) <= 0 for c in region.constraints)


def _dnf_poly_vertices(region: DNFRegion, variables: List[str]) -> List[Dict[str, Fraction]]:
    full = _dnf_region_with_cube(region, variables)
    cons = full.constraints
    n = len(variables)

    if n == 0:
        return [{}] if _dnf_satisfies_region({}, full) else []

    vertices: List[Dict[str, Fraction]] = []
    seen = set()
    for idxs in itertools.combinations(range(len(cons)), n):
        A: List[List[Fraction]] = []
        b: List[Fraction] = []
        for i in idxs:
            expr = cons[i].expr
            A.append([expr.coef.get(v, ZERO) for v in variables])
            b.append(-expr.const)
        sol = _dnf_solve_linear_system_fraction(A, b)
        if sol is None:
            continue
        point = {v: sol[k] for k, v in enumerate(variables)}
        if not _dnf_satisfies_region(point, full):
            continue
        key = tuple(point[v] for v in variables)
        if key not in seen:
            seen.add(key)
            vertices.append(point)

    if not vertices:
        for bits in itertools.product([ZERO, U], repeat=n):
            point = {v: bits[i] for i, v in enumerate(variables)}
            if _dnf_satisfies_region(point, full):
                key = tuple(point[v] for v in variables)
                if key not in seen:
                    seen.add(key)
                    vertices.append(point)

    return vertices


def _dnf_region_is_empty(region: DNFRegion, variables: List[str]) -> bool:
    return len(_dnf_poly_vertices(region, variables)) == 0


def _dnf_affine_minmax_on_region(aff: DNFAffine, region: DNFRegion, variables: List[str]) -> Tuple[Fraction, Fraction]:
    verts = _dnf_poly_vertices(region, variables)
    if not verts:
        raise ValueError("Empty region encountered")
    vals = [aff.eval(v) for v in verts]
    return min(vals), max(vals)


def _dnf_unique_affines(items: List[DNFAffine]) -> List[DNFAffine]:
    seen = set()
    out: List[DNFAffine] = []
    for a in items:
        k = a.key()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def _dnf_simplify_meet(term: DNFMeetTerm, variables: List[str]) -> DNFMeetTerm:
    uniq: Dict[Tuple, DNFClipFactor] = {}
    for f in term.factors:
        uniq[f.key()] = f

    factors = list(uniq.values())
    keep: List[DNFClipFactor] = []
    for i, f in enumerate(factors):
        mn, _ = f.affine.cube_minmax(variables)
        if mn >= U:
            continue

        redundant = False
        for j, g in enumerate(factors):
            if i == j:
                continue
            diff = f.affine - g.affine
            mn_diff, _ = diff.cube_minmax(variables)
            if mn_diff >= ZERO:
                redundant = True
                break
        if not redundant:
            keep.append(f)
    return DNFMeetTerm(keep)


def _dnf_simplify(dnf: DNForm, variables: List[str]) -> DNForm:
    seen = set()
    out: List[DNFMeetTerm] = []
    for c in dnf.clauses:
        sc = _dnf_simplify_meet(c, variables)
        key = tuple(sorted(f.key() for f in sc.factors))
        if key not in seen:
            seen.add(key)
            out.append(sc)
    return DNForm(out)


def _dnf_clause_from_piece(target: DNFPiece, components: List[DNFAffine], variables: List[str]) -> Optional[
    DNFMeetTerm]:
    L = target.affine
    if L.is_const(ZERO):
        return None
    if L.is_const(U):
        return None

    factors: List[DNFClipFactor] = []
    for A in components:
        diff = A - L
        mn, _ = _dnf_affine_minmax_on_region(diff, target.region, variables)
        if mn >= ZERO:
            if A.is_const(U):
                continue
            factors.append(DNFClipFactor(A))

    if not any(f.affine.key() == L.key() for f in factors):
        factors.append(DNFClipFactor(L))

    return _dnf_simplify_meet(DNFMeetTerm(factors), variables)


def formula_to_dnf(formula: Formula) -> Tuple[DNForm, List[str], List[DNFPiece]]:
    variables = _dnf_collect_variables(formula)
    raw_pieces = _dnf_piecewise_affine(formula)
    pieces = [p for p in raw_pieces if not _dnf_region_is_empty(p.region, variables)]

    if not pieces:
        return DNForm([]), variables, []

    components = _dnf_unique_affines([p.affine for p in pieces])
    if not any(not a.is_const() for a in components):
        if any(a.is_const(U) for a in components):
            return DNForm([DNFMeetTerm([])]), variables, pieces
        constants = [a for a in components if not a.is_const(ZERO)]
        if constants:
            return DNForm([DNFMeetTerm([DNFClipFactor(constants[0])])]), variables, pieces
        return DNForm([]), variables, pieces

    clauses: List[DNFMeetTerm] = []
    for p in pieces:
        clause = _dnf_clause_from_piece(p, components, variables)
        if clause is None:
            continue
        is_zero = any(f.affine.is_const(ZERO) for f in clause.factors)
        if not is_zero:
            clauses.append(clause)

    return _dnf_simplify(DNForm(clauses), variables), variables, pieces


def eval_dnf(dnf: DNForm, env: Dict[str, Union[float, Fraction]]) -> float:
    env_frac = {k: (v if isinstance(v, Fraction) else F(float(v))) for k, v in env.items()}

    def clip(q: Fraction) -> Fraction:
        return min(U, max(ZERO, q))

    def eval_factor(f: DNFClipFactor) -> Fraction:
        return clip(f.affine.eval(env_frac))

    if not dnf.clauses:
        return 0.0
    values = []
    for clause in dnf.clauses:
        if not clause.factors:
            values.append(U)
        else:
            values.append(min(eval_factor(f) for f in clause.factors))
    return float(max(values))


def dnf_equivalence_on_grid(formula: Formula, dnf: DNForm, variables: List[str],
                            step: Fraction = Fraction(1, 4)) -> bool:
    if len(variables) > 3:
        return True
    n = int(U / step)
    values = [step * k for k in range(n + 1)]
    for tup in itertools.product(values, repeat=len(variables)):
        env_frac = {variables[i]: tup[i] for i in range(len(variables))}
        env_float = {k: float(v) for k, v in env_frac.items()}
        lhs = eval_formula(formula, env_float, u=float(U))
        rhs = eval_dnf(dnf, env_frac)
        if abs(lhs - rhs) > 1e-9:
            print("DNF mismatch", env_float, "formula=", lhs, "dnf=", rhs)
            return False
    return True


def dnf_text_for_formula(formula: Formula, verify_step: Optional[Fraction] = None) -> Tuple[str, bool, int]:
    dnf, variables, pieces = formula_to_dnf(formula)
    ok = True if verify_step is None else dnf_equivalence_on_grid(formula, dnf, variables, step=verify_step)
    return dnf.pretty(variables), ok, len(pieces)


def dnf_outputs_for_X(dnfs: Dict[str, DNForm], X, feature_names) -> np.ndarray:
    names = sorted(dnfs.keys(), key=lambda name: int(name[1:]) if name.startswith("y") and name[1:].isdigit() else name)
    out = []
    for row in X:
        env = {feature_names[i]: float(row[i]) for i in range(len(feature_names))}
        out.append([eval_dnf(dnfs[name], env) for name in names])
    return np.array(out, dtype=float)


def metrics_from_score_matrix(y_true, score_matrix: np.ndarray) -> Dict[str, float]:
    score_matrix = np.asarray(score_matrix, dtype=float)
    preds = np.argmax(score_matrix, axis=1)
    probs = softmax_numpy(score_matrix)[:, 1] if score_matrix.shape[1] > 1 else score_matrix[:, 0]
    return compute_metrics(y_true, preds, probs)


def score_matrix_audit(reference_scores: np.ndarray, candidate_scores: np.ndarray) -> Dict[str, float]:
    reference_scores = np.asarray(reference_scores, dtype=float)
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    ref_pred = np.argmax(reference_scores, axis=1)
    cand_pred = np.argmax(candidate_scores, axis=1)
    diff = candidate_scores - reference_scores
    return {
        "Disagree": int(np.sum(cand_pred != ref_pred)),
        "Fidelity": float(np.mean(cand_pred == ref_pred)),
        "ScoreMSE": float(np.mean(diff ** 2)),
        "ScoreMaxErr": float(np.max(np.abs(diff))),
        "Unique": int(len(np.unique(np.round(candidate_scores, 6), axis=0))),
    }


def print_all_metrics_table(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print("\nMETRICS FOR ALL SCORE SOURCES — test split")
    print("{:<28} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "Source", "Accuracy", "Precision", "Recall", "F1", "AUC"
    ))
    print("-" * 86)
    for label, m in rows:
        print("{:<28} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>12.8f}".format(
            label, m["Accuracy"], m["Precision"], m["Recall"], m["F1-score"], m["AUC"]
        ))


def print_score_audit_table(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print("\nSCORE-LEVEL AUDIT VS KES — test split")
    print("{:<28} {:>10} {:>10} {:>14} {:>14} {:>10}".format(
        "Source", "Disagree", "Fidelity", "ScoreMSE", "ScoreMaxErr", "Unique"
    ))
    print("-" * 92)
    for label, a in rows:
        print("{:<28} {:>10d} {:>10.6f} {:>14.10f} {:>14.8f} {:>10d}".format(
            label, int(a["Disagree"]), a["Fidelity"], a["ScoreMSE"], a["ScoreMaxErr"], int(a["Unique"])
        ))


def _pairwise_rank_audit(reference_score: np.ndarray, candidate_score: np.ndarray) -> Dict[str, int]:
    ref = np.asarray(reference_score, dtype=float)
    cand = np.asarray(candidate_score, dtype=float)
    inversions = 0
    changed_ties = 0
    n = len(ref)
    for i in range(n):
        for j in range(i + 1, n):
            sr = np.sign(ref[i] - ref[j])
            sc = np.sign(cand[i] - cand[j])
            if sr == 0 or sc == 0:
                if sr != sc:
                    changed_ties += 1
            elif sr != sc:
                inversions += 1
    return {"RankInv": int(inversions), "TieChanges": int(changed_ties)}


def metric_audit_row(label: str, y_true, scores: np.ndarray, reference_scores: np.ndarray,) -> Tuple[str, Dict[str, float]]:
    scores = np.asarray(scores, dtype=float)
    reference_scores = np.asarray(reference_scores, dtype=float)

    m = metrics_from_score_matrix(y_true, scores)
    pred = np.argmax(scores, axis=1)
    ref_pred = np.argmax(reference_scores, axis=1)

    probs = softmax_numpy(scores)[:, 1] if scores.shape[1] > 1 else scores[:, 0]
    ref_probs = softmax_numpy(reference_scores)[:, 1] if reference_scores.shape[1] > 1 else reference_scores[:, 0]

    diff = scores - reference_scores
    prob_diff = probs - ref_probs
    margin = scores[:, 1] - scores[:, 0] if scores.shape[1] > 1 else scores[:, 0]
    ref_margin = reference_scores[:, 1] - reference_scores[:, 0] if reference_scores.shape[1] > 1 else reference_scores[
        :, 0]
    margin_diff = margin - ref_margin
    rank = _pairwise_rank_audit(ref_probs, probs)

    row = {
        **m,
        "Disagree": int(np.sum(pred != ref_pred)),
        "Fidelity": float(np.mean(pred == ref_pred)),
        "ScoreRMSE": float(np.sqrt(np.mean(diff ** 2))),
        "ScoreMaxErr": float(np.max(np.abs(diff))),
        "ProbRMSE": float(np.sqrt(np.mean(prob_diff ** 2))),
        "ProbMaxErr": float(np.max(np.abs(prob_diff))),
        "MarginRMSE": float(np.sqrt(np.mean(margin_diff ** 2))),
        "MarginMaxErr": float(np.max(np.abs(margin_diff))),
        "RankInv": int(rank["RankInv"]),
        "TieChanges": int(rank["TieChanges"]),
        "Unique": int(len(np.unique(np.round(scores, 6), axis=0))),
    }
    return label, row


def print_metric_audit_table(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print("\nALL MODELS / FORMULAS — classification + score audit on test split")
    print("{:<24} {:>8} {:>8} {:>8} {:>8} {:>10} {:>9} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}".format(
        "Source", "Acc", "Prec", "Rec", "F1", "AUC", "Disagr", "ScoreRMSE", "ScoreMax", "ProbRMSE", "ProbMax",
        "RankInv", "Unique"
    ))
    print("-" * 146)
    for label, m in rows:
        print(
            "{:<24} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.8f} {:>9d} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>8d} {:>8d}".format(
                label,
                m["Accuracy"], m["Precision"], m["Recall"], m["F1-score"], m["AUC"],
                int(m["Disagree"]), m["ScoreRMSE"], m["ScoreMaxErr"], m["ProbRMSE"], m["ProbMaxErr"],
                int(m["RankInv"]), int(m["Unique"])
            ))


def print_pairwise_score_difference_table(named_scores: List[Tuple[str, np.ndarray]]) -> None:
    print("\nPAIRWISE SCORE INPUT CHECK — max |score_A - score_B| on test split")
    print("{:<24} {:<24} {:>14}".format("A", "B", "max_abs_diff"))
    print("-" * 66)
    for i in range(len(named_scores)):
        for j in range(i + 1, len(named_scores)):
            a_name, a = named_scores[i]
            b_name, b = named_scores[j]
            print("{:<24} {:<24} {:>14.10f}".format(
                a_name, b_name, float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            ))




class CReLU_U(nn.Module):
    def __init__(self, u=U_VALUE):
        super().__init__()
        self.u = u

    def forward(self, x):
        return torch.clamp(x, 0.0, self.u)


class SmoothMinLayer(nn.Module):
    def __init__(self, in_dim, out_dim, u=U_VALUE, mu=4.0, temperature=1.0, dropout_p=0.0, eps=1e-8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.u = u
        self.mu = mu
        self.temperature = temperature
        self.dropout_p = dropout_p
        self.eps = eps
        self.selector_logits = nn.Parameter(torch.empty(out_dim, in_dim))
        self.act = CReLU_U(u=u)
        self.drop = nn.Dropout(dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.selector_logits, mean=1.5, std=0.15)

    def gate_values(self):
        return torch.sigmoid(self.selector_logits / self.temperature)

    def forward(self, x):
        w = self.gate_values()
        x_exp = x.unsqueeze(1)
        logw = torch.log(w.clamp_min(self.eps)).unsqueeze(0)
        num = torch.logsumexp(-self.mu * x_exp + logw, dim=-1)
        den = torch.logsumexp(logw, dim=-1).squeeze(0)
        y = -(num - den.unsqueeze(0)) / self.mu
        y = self.act(y)
        if self.training and self.dropout_p > 0:
            y = self.drop(y)
        return y

    def extract_connectivity(self, selector_threshold=0.5):
        with torch.no_grad():
            gate = self.gate_values().detach().cpu().numpy()
        groups = []
        for j in range(self.out_dim):
            idxs = [i for i in range(self.in_dim) if gate[j, i] >= selector_threshold]
            if len(idxs) == 0:
                idxs = [int(np.argmax(gate[j]))]
            groups.append(idxs)
        return groups


class SmoothMaxLayer(SmoothMinLayer):
    def forward(self, x):
        w = self.gate_values()
        x_exp = x.unsqueeze(1)
        logw = torch.log(w.clamp_min(self.eps)).unsqueeze(0)
        num = torch.logsumexp(self.mu * x_exp + logw, dim=-1)
        den = torch.logsumexp(logw, dim=-1).squeeze(0)
        y = (num - den.unsqueeze(0)) / self.mu
        y = self.act(y)
        if self.training and self.dropout_p > 0:
            y = self.drop(y)
        return y


class ClassKESPath(nn.Module):
    def __init__(self, input_dim=2, num_hyperplanes=6, min_layer_dims=(4,), max_layer_dims=(2, 1),
                 u=U_VALUE, mu=4.0, temperature=1.0, dropout_p=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_hyperplanes = num_hyperplanes
        self.min_layer_dims = list(min_layer_dims)
        self.max_layer_dims = list(max_layer_dims)
        self.u = u

        if len(self.max_layer_dims) == 0 or self.max_layer_dims[-1] != 1:
            raise ValueError("Ultimul strat max trebuie sa aiba exact 1 neuron.")

        self.linear = nn.Linear(input_dim, num_hyperplanes)
        self.linear_act = CReLU_U(u=u)

        self.min_layers = nn.ModuleList()
        prev_dim = num_hyperplanes
        for out_dim in self.min_layer_dims:
            self.min_layers.append(
                SmoothMinLayer(prev_dim, out_dim, u=u, mu=mu, temperature=temperature, dropout_p=dropout_p))
            prev_dim = out_dim

        self.max_layers = nn.ModuleList()
        for out_dim in self.max_layer_dims:
            self.max_layers.append(
                SmoothMaxLayer(prev_dim, out_dim, u=u, mu=mu, temperature=temperature, dropout_p=dropout_p))
            prev_dim = out_dim

        self.score_bias = nn.Parameter(torch.zeros(1))
        self.output_act = CReLU_U(u=u)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)
        for layer in self.min_layers:
            layer.reset_parameters()
        for layer in self.max_layers:
            layer.reset_parameters()
        nn.init.constant_(self.score_bias, 0.0)

    def forward(self, x):
        h = self.linear_act(self.linear(x))
        for layer in self.min_layers:
            h = layer(h)
        for layer in self.max_layers:
            h = layer(h)
        score = self.output_act(h.squeeze(-1) + self.score_bias)
        return score

    def extract_symbolic_path(
            self,
            class_idx,
            feature_names=None,
            eps=1e-8,
            selector_threshold=0.5,
            extraction_mode="exact",
            compact_gate_threshold=0.10,
            compact_top_k=2,
            compact_min_keep=1,
            verbose=False,
    ):
        symbolic_defs = []
        input_refs = [SymbolRef(f"x{i + 1}") for i in range(self.input_dim)] if feature_names is None else [
            SymbolRef(str(n)) for n in feature_names]

        W = self.linear.weight.detach().cpu().numpy()
        b = self.linear.bias.detach().cpu().numpy()

        prev_refs = []
        iterator = range(self.num_hyperplanes)
        if verbose and tqdm is not None:
            iterator = tqdm(iterator, desc=f"Class {class_idx + 1} hyperplanes", leave=True)

        for j in iterator:
            name = f"c{class_idx + 1}_p{j + 1}"
            terms = [(float(W[j, i]), input_refs[i]) for i in range(self.input_dim)]
            formula = simplify(ClipAffine(terms, bias=float(b[j]), u=self.u))
            symbolic_defs.append((name, formula))
            prev_refs.append(SymbolRef(name))

        if extraction_mode not in {"exact", "compact", "crisp"}:
            raise ValueError("extraction_mode must be one of: 'exact', 'compact', 'crisp'")

        for layer_idx, layer in enumerate(self.min_layers, start=1):
            current_refs = []
            if extraction_mode == "crisp":
                groups = layer.extract_connectivity(selector_threshold=selector_threshold)
                for j, idxs in enumerate(groups):
                    name = f"c{class_idx + 1}_m{layer_idx}_{j + 1}"
                    formula = simplify(And(*(prev_refs[i] for i in idxs)))
                    symbolic_defs.append((name, formula))
                    current_refs.append(SymbolRef(name))
            else:
                with torch.no_grad():
                    gate = layer.gate_values().detach().cpu().numpy()
                for j in range(layer.out_dim):
                    name = f"c{class_idx + 1}_m{layer_idx}_{j + 1}"
                    if extraction_mode == "compact":
                        idxs = _select_compact_gate_indices(
                            gate[j],
                            gate_threshold=compact_gate_threshold,
                            top_k=compact_top_k,
                            min_keep=compact_min_keep,
                        )
                    else:
                        idxs = list(range(layer.in_dim))
                    formula = simplify(SoftMinFormula(
                        [prev_refs[i] for i in idxs],
                        [gate[j, i] for i in idxs],
                        mu=layer.mu,
                        u=layer.u,
                        eps=layer.eps,
                    ))
                    symbolic_defs.append((name, formula))
                    current_refs.append(SymbolRef(name))
            prev_refs = current_refs

        for layer_idx, layer in enumerate(self.max_layers, start=1):
            current_refs = []
            if extraction_mode == "crisp":
                groups = layer.extract_connectivity(selector_threshold=selector_threshold)
                for j, idxs in enumerate(groups):
                    name = f"c{class_idx + 1}_u{layer_idx}_{j + 1}"
                    formula = simplify(Or(*(prev_refs[i] for i in idxs)))
                    symbolic_defs.append((name, formula))
                    current_refs.append(SymbolRef(name))
            else:
                with torch.no_grad():
                    gate = layer.gate_values().detach().cpu().numpy()
                for j in range(layer.out_dim):
                    name = f"c{class_idx + 1}_u{layer_idx}_{j + 1}"
                    if extraction_mode == "compact":
                        idxs = _select_compact_gate_indices(
                            gate[j],
                            gate_threshold=compact_gate_threshold,
                            top_k=compact_top_k,
                            min_keep=compact_min_keep,
                        )
                    else:
                        idxs = list(range(layer.in_dim))
                    formula = simplify(SoftMaxFormula(
                        [prev_refs[i] for i in idxs],
                        [gate[j, i] for i in idxs],
                        mu=layer.mu,
                        u=layer.u,
                        eps=layer.eps,
                    ))
                    symbolic_defs.append((name, formula))
                    current_refs.append(SymbolRef(name))
            prev_refs = current_refs

        bias = float(self.score_bias.detach().cpu().item())

        y_formula = simplify(
            ClipAffine(
                [(1.0, prev_refs[0])],
                bias=bias,
                u=self.u,
            )
        )

        symbolic_defs.append((f"y{class_idx + 1}", y_formula))
        return symbolic_defs


class MultiClassKESPaths(nn.Module):
    def __init__(self, input_dim=2, num_classes=2, num_hyperplanes=6, min_layer_dims=(4,), max_layer_dims=(2, 1),
                 u=U_VALUE, mu=4.0, temperature=1.0, dropout_p=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.u = u
        self.paths = nn.ModuleList([
            ClassKESPath(input_dim, num_hyperplanes, min_layer_dims, max_layer_dims, u, mu, temperature, dropout_p)
            for _ in range(num_classes)
        ])

    def forward(self, x):
        scores = [path(x) for path in self.paths]
        return torch.stack(scores, dim=1)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)

    def predict(self, x):
        return self.forward(x).argmax(dim=1)

    def extract_symbolic_system(
            self,
            feature_names=None,
            eps=1e-8,
            selector_threshold=0.5,
            extraction_mode="exact",
            compact_gate_threshold=0.10,
            compact_top_k=2,
            compact_min_keep=1,
            verbose=False,
    ):
        symbolic_defs = []
        iterator = range(len(self.paths))
        if verbose and tqdm is not None:
            iterator = tqdm(iterator, desc="Classes", leave=True)
        for c in iterator:
            symbolic_defs.extend(self.paths[c].extract_symbolic_path(
                class_idx=c,
                feature_names=feature_names,
                eps=eps,
                selector_threshold=selector_threshold,
                extraction_mode=extraction_mode,
                compact_gate_threshold=compact_gate_threshold,
                compact_top_k=compact_top_k,
                compact_min_keep=compact_min_keep,
                verbose=verbose,
            ))
        return symbolic_defs


def load_breast_cancer_dataset_pca_same_then_to_interval(n_components=3, val_size=0.2, test_size=0.2,
                                                         random_state=42, u_value=U_VALUE):
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_relative_size, random_state=random_state, stratify=y_train_full
    )

    scaler_raw = MinMaxScaler()
    X_train_scaled = scaler_raw.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler_raw.transform(X_val).astype(np.float32)
    X_test_scaled = scaler_raw.transform(X_test).astype(np.float32)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_val_pca = pca.transform(X_val_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    scaler_pca_interval = MinMaxScaler(feature_range=(0.0, float(u_value)))
    X_train_0u = scaler_pca_interval.fit_transform(X_train_pca).astype(np.float32)
    X_val_0u = scaler_pca_interval.transform(X_val_pca).astype(np.float32)
    X_test_0u = scaler_pca_interval.transform(X_test_pca).astype(np.float32)

    return {
        "X_train": X_train_0u, "X_val": X_val_0u, "X_test": X_test_0u,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": [f"x{i + 1}" for i in range(n_components)],
        "target_names": data.target_names,
        "scaler_raw": scaler_raw, "pca": pca, "scaler_pca_interval": scaler_pca_interval,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def print_pca_components_importance(pca, feature_names, top_k=3):
    print("\nPCA component composition")
    for i, comp in enumerate(pca.components_, start=1):
        idxs = np.argsort(np.abs(comp))[::-1][:top_k]
        terms = [f"({comp[j]:.3f} * {feature_names[j]})" for j in idxs]
        print(f"x{i} ≈ " + " + ".join(terms))
    print("-" * 60)


def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=1000, lr=0.01, batch_size=32,
                weight_decay=1e-4, print_every=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            pred_loss = criterion(logits, yb)

            bin_loss = gate_binarization_loss(model)
            sparse_loss = gate_sparsity_loss(model)

            loss = (
                    pred_loss
                    + 0.05 * bin_loss
                    + 0.005 * sparse_loss
            )
            loss.backward()
            optimizer.step()

        if epoch % print_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(X_train)
                train_pred = train_logits.argmax(dim=1)
                train_acc = (train_pred == y_train).float().mean().item()
                train_loss = criterion(train_logits, y_train).item()
                msg = f"Epoch {epoch:4d} | TrainLoss = {train_loss:.6f} | TrainAcc = {train_acc:.4f}"
                if X_val is not None and y_val is not None:
                    val_logits = model(X_val)
                    val_pred = val_logits.argmax(dim=1)
                    val_acc = (val_pred == y_val).float().mean().item()
                    val_loss = criterion(val_logits, y_val).item()
                    msg += f" | ValLoss = {val_loss:.6f} | ValAcc = {val_acc:.4f}"
                print(msg)


def evaluate_model(model, X, y):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        pred_labels = logits.argmax(dim=1)
        acc = (pred_labels == y).float().mean().item()
        ce = criterion(logits, y).item()
    return ce, acc, logits, probs, pred_labels


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
    }


def print_output_biases(model):
    print("\n===== OUTPUT BIASES =====")
    for c, path in enumerate(model.paths):
        b = float(path.score_bias.detach().cpu().item())
        print(f"class {c + 1}: score_bias = {b:.6f}")


def softmax_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    z = values - np.max(values)
    e = np.exp(z)
    return e / e.sum()


def row_env(row, feature_names) -> Dict[str, float]:
    return {feature_names[i]: float(row[i]) for i in range(len(feature_names))}


def formula_values(finals: Dict[str, Formula], row, feature_names, u=U_VALUE) -> np.ndarray:
    env = row_env(row, feature_names)
    names = sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s)
    return np.array([eval_formula(finals[name], env, u=u) for name in names], dtype=float)


def post_reconstruct(bits: List[int], u=Fraction(1)) -> float:
    if not bits:
        return 0.0
    return float(u) * sum(bits) / len(bits)


def moisil_post_value(value: float, n: int = 21, u: Fraction = Fraction(1)) -> float:
    bits = [1 if value >= float(Fraction(k, n - 1) * u) else 0 for k in range(1, n)]
    return post_reconstruct(bits, u=u)


def moisil_values_from_formula(finals: Dict[str, Formula], row, feature_names, n=21, u=Fraction(1)) -> np.ndarray:
    vals = formula_values(finals, row, feature_names, u=float(u))
    return np.array([moisil_post_value(v, n=n, u=u) for v in vals], dtype=float)


def evaluate_symbolic_formula_classifier(finals, X, y, feature_names, u=U_VALUE):
    preds, scores = [], []
    for row in X:
        vals = formula_values(finals, row, feature_names, u=u)
        probs = softmax_np(vals)
        preds.append(int(np.argmax(vals)))
        scores.append(float(probs[1]) if len(probs) > 1 else float(vals[0]))
    return compute_metrics(y, np.array(preds), np.array(scores))


def evaluate_moisil_formula_classifier(finals, X, y, feature_names, n=21, u=Fraction(1)):
    preds, scores = [], []
    for row in X:
        vals = moisil_values_from_formula(finals, row, feature_names, n=n, u=u)
        probs = softmax_np(vals)
        preds.append(int(np.argmax(vals)))
        scores.append(float(probs[1]) if len(probs) > 1 else float(vals[0]))
    return compute_metrics(y, np.array(preds), np.array(scores))


def formula_positive_scores(finals, X, feature_names, n: Optional[int] = None, u: Fraction = Fraction(1)) -> np.ndarray:
    out = []
    for row in X:
        if n is None:
            vals = formula_values(finals, row, feature_names, u=float(u))
        else:
            vals = moisil_values_from_formula(finals, row, feature_names, n=n, u=u)
        probs = softmax_np(vals)
        out.append(float(probs[1]) if len(probs) > 1 else float(vals[0]))
    return np.array(out)


def best_threshold_by_f1(y_true, scores):
    candidates = np.unique(scores)
    candidates = np.concatenate([[scores.min() - 1e-9], candidates, [scores.max() + 1e-9]])
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def evaluate_moisil_formula_calibrated(finals, X_val, y_val, X_test, y_test, feature_names, n=21, u=Fraction(1)):
    val_scores = formula_positive_scores(finals, X_val, feature_names, n=n, u=u)
    best_t, best_val_f1 = best_threshold_by_f1(y_val, val_scores)
    test_scores = formula_positive_scores(finals, X_test, feature_names, n=n, u=u)
    test_pred = (test_scores >= best_t).astype(int)
    metrics = compute_metrics(y_test, test_pred, test_scores)
    print(f"Best Moisil threshold on validation: {best_t:.6f}")
    print(f"Best validation F1: {best_val_f1:.6f}")
    print("Moisil score unique values:", np.unique(test_scores)[:20], "... total", len(np.unique(test_scores)))
    return metrics


def print_metric_table(rows: Dict[str, Dict[str, float]]) -> None:
    print("\nTABLE — PREDICTIVE PERFORMANCE")
    print("{:<28} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1-score", "AUC"
    ))
    for name, m in rows.items():
        print("{:<28} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            name, m["Accuracy"], m["Precision"], m["Recall"], m["F1-score"], m["AUC"]
        ))


def print_symbolic_diagnostics(model, finals, data, X_test_t, max_rows=10):
    print("\n===== SYMBOLIC DIAGNOSTICS: neural vs extracted formula =====")
    with torch.no_grad():
        logits = model(X_test_t[:max_rows])
        probs = torch.softmax(logits, dim=1)
    for i in range(max_rows):
        vals = formula_values(finals, data["X_test"][i], data["feature_names"])
        mvals = moisil_values_from_formula(finals, data["X_test"][i], data["feature_names"], n=21, u=Fraction(1))
        print(
            i,
            "true=", int(data["y_test"][i]),
            "neural_logits=", np.round(logits[i].detach().cpu().numpy(), 4),
            "neural_prob=", np.round(probs[i].detach().cpu().numpy(), 4),
            "symbolic=", np.round(vals, 4),
            "moisil=", np.round(mvals, 4),
        )


def moisil_discretize_scores(scores, n=21, u=1.0):
    thresholds = np.array([k * u / (n - 1) for k in range(1, n)])
    bits = scores[..., None] >= thresholds
    return bits.mean(axis=-1) * u


def softmax_numpy(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def moisil_bits_for_value(value: float, n=21, u=1.0):
    thresholds = np.array([k * u / (n - 1) for k in range(1, n)])
    bits = (value >= thresholds).astype(int)
    return bits, thresholds


def confidence_label(post_value: float):
    if post_value < 0.25:
        return "very low"
    if post_value < 0.50:
        return "low"
    if post_value < 0.75:
        return "medium"
    if post_value < 0.90:
        return "high"
    return "very high"


def print_moisil_threshold_explanations(model, X, y,
                                        feature_names, n=21, u=1.0, max_rows=8):
    print("L-M threshold explanation")

    with torch.no_grad():
        outputs = model(torch.tensor(X[:max_rows], dtype=torch.float32)).cpu().numpy()

    for i in range(max_rows):
        vals = outputs[i]
        moisil_vals = moisil_discretize_scores(vals[None, :], n=n, u=u)[0]
        pred = int(np.argmax(moisil_vals))

        print(f"\nSample {i}")
        print(f"true class: {int(y[i])}")
        print(f"network output: {np.round(vals, 4)}")
        print(f"Ł-M/Post output: {np.round(moisil_vals, 4)}")
        print(f"predicted class: {pred}")

        for c, value in enumerate(vals):
            bits, thresholds = moisil_bits_for_value(value, n=n, u=u)
            bit_string = "".join(str(int(b)) for b in bits)
            active_thresholds = thresholds[bits == 1]

            if len(active_thresholds) == 0:
                reached = "no positive threshold"
            else:
                reached = f"up to {active_thresholds.max():.2f}"

            print(
                f"class {c}: value={value:.4f}, "
                f"Post bits={bit_string}, "
                f"confidence={confidence_label(moisil_vals[c])}, "
                f"reaches {reached}"
            )


def write_moisil_threshold_explanations(filepath, model,
                                        X, y, n=21, u=1.0,
                                        max_rows=8):
    with torch.no_grad():
        outputs = model(torch.tensor(X[:max_rows], dtype=torch.float32)).cpu().numpy()

    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\nL-M/Post threshold explanations\n")
        f.write("=" * 80 + "\n")

        for i in range(max_rows):
            vals = outputs[i]
            moisil_vals = moisil_discretize_scores(vals[None, :], n=n, u=u)[0]
            pred = int(np.argmax(moisil_vals))

            f.write(f"\nSample {i}\n")
            f.write(f"true class: {int(y[i])}\n")
            f.write(f"network output: {np.round(vals, 4)}\n")
            f.write(f"Ł-M/Post output: {np.round(moisil_vals, 4)}\n")
            f.write(f"predicted class: {pred}\n")

            for c, value in enumerate(vals):
                bits, thresholds = moisil_bits_for_value(value, n=n, u=u)
                bit_string = "".join(str(int(b)) for b in bits)
                active_thresholds = thresholds[bits == 1]

                if len(active_thresholds) == 0:
                    reached = "no positive threshold"
                else:
                    reached = f"up to {active_thresholds.max():.2f}"

                f.write(
                    f"  class {c}: value={value:.4f}, "
                    f"Post bits={bit_string}, "
                    f"confidence={confidence_label(moisil_vals[c])}, "
                    f"reaches {reached}\n"
                )


def print_gate_statistics(model):
    print("\n===== GATE STATISTICS =====")

    for c, path in enumerate(model.paths):
        print(f"\nClass {c + 1}")

        for layer_name, layers in [
            ("min", path.min_layers),
            ("max", path.max_layers),
        ]:
            for li, layer in enumerate(layers, start=1):
                g = layer.gate_values().detach().cpu().numpy().ravel()
                print(
                    f"{layer_name}{li}: "
                    f"min={g.min():.4f}, "
                    f"q25={np.quantile(g, 0.25):.4f}, "
                    f"mean={g.mean():.4f}, "
                    f"q75={np.quantile(g, 0.75):.4f}, "
                    f"max={g.max():.4f}"
                )


def symbolic_outputs_for_X(finals, X, feature_names, u=U_VALUE):
    out = []

    for row in X:
        vals = formula_values(
            finals,
            row,
            feature_names,
            u=u,
        )
        out.append(vals)

    return np.array(out, dtype=float)


def symbolic_fidelity_diagnostics(model, finals, X, feature_names):
    with torch.no_grad():
        neural_outputs = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    symbolic_outputs = symbolic_outputs_for_X(finals, X, feature_names, u=U_VALUE)
    neural_pred = np.argmax(neural_outputs, axis=1)
    symbolic_pred = np.argmax(symbolic_outputs, axis=1)
    diff = symbolic_outputs - neural_outputs
    disagreements = int(np.sum(symbolic_pred != neural_pred))

    return {
        "fidelity": float(np.mean(symbolic_pred == neural_pred)),
        "disagreements": disagreements,
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "unique_outputs": int(len(np.unique(np.round(symbolic_outputs, 6), axis=0))),
    }


def print_extraction_quality_table(rows):
    print("\nTABLE — SYMBOLIC EXTRACTION QUALITY VS KES")
    print("{:<28} {:<8} {:<10} {:<10} {:<12} {:<12} {:<10} {:<8} {:<8}".format(
        "Formula", "Split", "Fidelity", "Disagree", "MSE(scores)", "MaxAbsErr", "Unique", "DAG", "Expand"
    ))
    for r in rows:
        print("{:<28} {:<8} {:<10.4f} {:<10d} {:<12.8f} {:<12.8f} {:<10d} {:<8d} {:<8d}".format(
            r["Formula"], r["Split"], r["Fidelity"], r["Disagree"], r["MSE"],
            r["MaxAbsErr"], r["Unique"], r["DAG"], r["Expand"]
        ))


def write_extraction_quality_table(f, rows):
    f.write("\nSymbolic extraction quality vs KES\n")
    for r in rows:
        f.write(
            f"{r['Formula']} [{r['Split']}]: "
            f"fidelity={r['Fidelity']:.6f}, "
            f"disagreements={r['Disagree']}, "
            f"mse={r['MSE']:.10f}, "
            f"max_abs_error={r['MaxAbsErr']:.10f}, "
            f"unique_outputs={r['Unique']}, "
            f"dag_len={r['DAG']}, expanded_len={r['Expand']}\n"
        )


def make_quality_row(label, split, model, finals, symbolic_defs, X, feature_names):
    d = symbolic_fidelity_diagnostics(model, finals, X, feature_names)
    return {
        "Formula": label,
        "Split": split,
        "Fidelity": d["fidelity"],
        "Disagree": d["disagreements"],
        "MSE": d["mse"],
        "MaxAbsErr": d["max_abs_error"],
        "Unique": d["unique_outputs"],
        "DAG": compact_formula_length(symbolic_defs),
        "Expand": expanded_formula_length(finals),
    }


def evaluate_symbolic_fidelity(model, X, feature_names,
                               selector_threshold=0.5, eps=1e-8, extraction_mode="exact",
                               compact_gate_threshold=0.10, compact_top_k=2, compact_min_keep=1):
    with torch.no_grad():
        neural_outputs = model(
            torch.tensor(X, dtype=torch.float32)
        ).cpu().numpy()

    neural_pred = np.argmax(neural_outputs, axis=1)

    symbolic_defs = model.extract_symbolic_system(
        feature_names=feature_names,
        selector_threshold=selector_threshold,
        eps=eps,
        extraction_mode=extraction_mode,
        compact_gate_threshold=compact_gate_threshold,
        compact_top_k=compact_top_k,
        compact_min_keep=compact_min_keep,
        verbose=False,
    )

    finals = get_fully_expanded_formulas(symbolic_defs)
    symbolic_outputs = symbolic_outputs_for_X(
        finals,
        X,
        feature_names,
        u=U_VALUE,
    )

    symbolic_pred = np.argmax(symbolic_outputs, axis=1)

    fidelity = np.mean(symbolic_pred == neural_pred)
    mse = np.mean((symbolic_outputs - neural_outputs) ** 2)

    unique_rows = np.unique(np.round(symbolic_outputs, 6), axis=0)

    return {
        "threshold": selector_threshold,
        "fidelity": fidelity,
        "mse": mse,
        "unique_outputs": len(unique_rows),
        "symbolic_defs": symbolic_defs,
        "finals": finals,
        "compact_length": compact_formula_length(symbolic_defs),
        "expanded_length": expanded_formula_length(finals),
    }


def find_best_compact_extraction(model, X_val, feature_names, gate_thresholds=None, top_k_values=(1, 2, 3),
                                min_keep=1, selection_policy="shortest"):
    if gate_thresholds is None:
        gate_thresholds = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    print("\n===== COMPACT EXTRACTION SEARCH =====")
    results = []
    for top_k in top_k_values:
        for gate_t in gate_thresholds:
            res = evaluate_symbolic_fidelity(
                model=model,
                X=X_val,
                feature_names=feature_names,
                extraction_mode="compact",
                compact_gate_threshold=float(gate_t),
                compact_top_k=int(top_k),
                compact_min_keep=int(min_keep),
            )
            res["gate_threshold"] = float(gate_t)
            res["top_k"] = int(top_k)
            res["min_keep"] = int(min_keep)
            results.append(res)
            print(
                f"top_k={top_k} | gate_threshold={gate_t:.2f} | "
                f"fidelity={res['fidelity']:.4f} | mse={res['mse']:.6f} | "
                f"compact_len={res['compact_length']} | expanded_len={res['expanded_length']}"
            )

    if selection_policy == "shortest":
        key_fn = lambda r: (-r["fidelity"], r["compact_length"], r["mse"])
    elif selection_policy == "closest":
        key_fn = lambda r: (-r["fidelity"], r["mse"], r["compact_length"])
    else:
        raise ValueError("selection_policy must be 'shortest' or 'closest'")

    best = sorted(results, key=key_fn)[0]
    best["selection_policy"] = selection_policy

    print("\nBest compact extraction:")
    print(
        f"selection_policy={selection_policy}, "
        f"top_k={best['top_k']}, gate_threshold={best['gate_threshold']:.2f}, "
        f"fidelity={best['fidelity']:.4f}, mse={best['mse']:.6f}, "
        f"compact_len={best['compact_length']}, expanded_len={best['expanded_length']}"
    )
    return best


def find_best_selector_threshold(model, X_val, feature_names, candidates=None):
    if candidates is None:
        candidates = np.linspace(0.55, 0.98, 18)

    print("\nSelector threshold search")
    results = []

    for t in candidates:
        res = evaluate_symbolic_fidelity(
            model=model,
            X=X_val,
            feature_names=feature_names,
            selector_threshold=float(t),
            extraction_mode="crisp",
        )

        results.append(res)

        print(
            f"threshold={t:.4f} | "
            f"fidelity={res['fidelity']:.4f} | "
            f"mse={res['mse']:.6f} | "
            f"unique symbolic outputs={res['unique_outputs']}"
        )

    best = sorted(
        results,
        key=lambda r: (-r["fidelity"], r["mse"], -r["unique_outputs"]),
    )[0]

    print("\nBest selector threshold:")
    print(
        f"threshold={best['threshold']:.4f}, "
        f"fidelity={best['fidelity']:.4f}, "
        f"mse={best['mse']:.6f}, "
        f"unique_outputs={best['unique_outputs']}"
    )

    return best


def gate_binarization_loss(model):
    loss = 0.0
    count = 0

    for path in model.paths:
        for layer in list(path.min_layers) + list(path.max_layers):
            g = layer.gate_values()
            loss = loss + torch.mean(g * (1.0 - g))
            count += 1

    if count == 0:
        return torch.tensor(0.0)

    return loss / count


def gate_sparsity_loss(model):
    loss = 0.0
    count = 0

    for path in model.paths:
        for layer in list(path.min_layers) + list(path.max_layers):
            g = layer.gate_values()
            loss = loss + torch.mean(g)
            count += 1

    if count == 0:
        return torch.tensor(0.0)

    return loss / count


def summarize_symbolic_nodes(symbolic_defs, X, feature_names, u=U_VALUE):
    node_values = {name: [] for name, _ in symbolic_defs}

    for row in X:
        env = {
            feature_names[j]: float(row[j])
            for j in range(len(feature_names))
        }

        for name, formula in symbolic_defs:
            value = eval_formula(formula, env, u=u)
            env[name] = value
            node_values[name].append(value)

    print("\nsymbolic node summary")
    for name, values in node_values.items():
        arr = np.array(values, dtype=float)
        print(
            f"{name:12s} "
            f"min={arr.min():.6f} "
            f"max={arr.max():.6f} "
            f"mean={arr.mean():.6f} "
            f"unique={len(np.unique(np.round(arr, 6)))}"
        )



if __name__ == "__main__":
    import contextlib
    import io

    set_reproducibility(42)

    outdir = "kes_breast_cancer_moisil_no_dnf_report"
    os.makedirs(outdir, exist_ok=True)

    n_components = 3
    n_moisil = 21
    train_epochs = int(os.environ.get("KES_TRAIN_EPOCHS", "1000"))
    quiet_training = os.environ.get("KES_QUIET_TRAINING", "1") != "0"
    compact_selection_policy = "shortest"
    compact_gate_thresholds = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    compact_top_k_values = (1,)
    compact_min_keep = 1

    data = load_breast_cancer_dataset_pca_same_then_to_interval(
        n_components=n_components,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        u_value=U_VALUE,
    )

    X_train_t = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_t = torch.tensor(data["y_train"], dtype=torch.long)
    X_val_t = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val_t = torch.tensor(data["y_val"], dtype=torch.long)
    X_test_t = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test_t = torch.tensor(data["y_test"], dtype=torch.long)

    model = MultiClassKESPaths(
        input_dim=n_components,
        num_classes=2,
        num_hyperplanes=6,
        min_layer_dims=(4,),
        max_layer_dims=(2, 1),
        u=U_VALUE,
        mu=6.0,
        temperature=1.0,
    )

    print("\nKES compact symbolic extraction", flush=True)
    if quiet_training:
        with contextlib.redirect_stdout(io.StringIO()):
            train_model(
                model,
                X_train_t,
                y_train_t,
                X_val=X_val_t,
                y_val=y_val_t,
                epochs=train_epochs,
                lr=0.01,
                batch_size=32,
                weight_decay=1e-4,
                print_every=train_epochs + 1,
            )
    else:
        train_model(
            model,
            X_train_t,
            y_train_t,
            X_val=X_val_t,
            y_val=y_val_t,
            epochs=train_epochs,
            lr=0.01,
            batch_size=32,
            weight_decay=1e-4,
            print_every=200,
        )

    _, _, kes_logits, kes_probs, kes_pred = evaluate_model(model, X_test_t, y_test_t)
    kes_metrics = compute_metrics(data["y_test"], kes_pred.cpu().numpy(), kes_probs[:, 1].cpu().numpy())

    exact_reference = evaluate_symbolic_fidelity(
        model=model,
        X=data["X_val"],
        feature_names=data["feature_names"],
        extraction_mode="exact",
    )

    compact_candidates = []
    for top_k in compact_top_k_values:
        for gate_t in compact_gate_thresholds:
            res = evaluate_symbolic_fidelity(
                model=model,
                X=data["X_val"],
                feature_names=data["feature_names"],
                extraction_mode="compact",
                compact_gate_threshold=float(gate_t),
                compact_top_k=int(top_k),
                compact_min_keep=int(compact_min_keep),
            )
            res["top_k"] = int(top_k)
            res["gate_threshold"] = float(gate_t)
            res["min_keep"] = int(compact_min_keep)
            compact_candidates.append(res)

    if compact_selection_policy == "shortest":
        compact_key = lambda r: (-r["fidelity"], r["expanded_length"], r["compact_length"], r["mse"])
    elif compact_selection_policy == "closest":
        compact_key = lambda r: (-r["fidelity"], r["mse"], r["expanded_length"], r["compact_length"])
    else:
        raise ValueError("compact_selection_policy must be 'shortest' or 'closest'")

    best_compact = sorted(compact_candidates, key=compact_key)[0]
    compact_gate_threshold = best_compact["gate_threshold"]
    compact_top_k = best_compact["top_k"]

    symbolic_defs = model.extract_symbolic_system(
        feature_names=data["feature_names"],
        selector_threshold=0.5,
        extraction_mode="compact",
        compact_gate_threshold=compact_gate_threshold,
        compact_top_k=compact_top_k,
        compact_min_keep=compact_min_keep,
        verbose=False,
    )
    finals = get_fully_expanded_formulas(symbolic_defs)

    dnf_approx_decimals = parse_optional_int_env("KES_DNF_APPROX_DECIMALS", 0)
    dnf_approx_coeff_eps = float(os.environ.get("KES_DNF_APPROX_COEFF_EPS", "0.0"))
    dnf_finals = quantize_finals_for_dnf(
        finals,
        decimals=dnf_approx_decimals,
        coeff_abs_threshold=dnf_approx_coeff_eps,
    )

    exact_defs = exact_reference["symbolic_defs"]
    exact_finals = exact_reference["finals"]

    exact_test_quality = make_quality_row(
        "exact", "test", model, exact_finals, exact_defs, data["X_test"], data["feature_names"]
    )
    compact_val_quality = make_quality_row(
        "compact", "val", model, finals, symbolic_defs, data["X_val"], data["feature_names"]
    )
    compact_test_quality = make_quality_row(
        "compact", "test", model, finals, symbolic_defs, data["X_test"], data["feature_names"]
    )

    exact_metrics = evaluate_symbolic_formula_classifier(
        exact_finals,
        data["X_test"],
        data["y_test"],
        data["feature_names"],
        u=U_VALUE,
    )
    compact_metrics = evaluate_symbolic_formula_classifier(
        finals,
        data["X_test"],
        data["y_test"],
        data["feature_names"],
        u=U_VALUE,
    )

    with torch.no_grad():
        neural_scores = model(X_test_t).cpu().numpy()
    exact_scores = symbolic_outputs_for_X(exact_finals, data["X_test"], data["feature_names"], u=U_VALUE)
    compact_scores = symbolic_outputs_for_X(finals, data["X_test"], data["feature_names"], u=U_VALUE)
    dnf_input_scores = symbolic_outputs_for_X(dnf_finals, data["X_test"], data["feature_names"], u=U_VALUE)

    neural_prob1 = softmax_numpy(neural_scores)[:, 1]
    exact_prob1 = softmax_numpy(exact_scores)[:, 1]
    compact_prob1 = softmax_numpy(compact_scores)[:, 1]

    neural_pred_np = np.argmax(neural_scores, axis=1)
    exact_pred_np = np.argmax(exact_scores, axis=1)
    compact_pred_np = np.argmax(compact_scores, axis=1)
    compact_disagree = int(np.sum(compact_pred_np != neural_pred_np))
    exact_disagree = int(np.sum(exact_pred_np != neural_pred_np))

    compact_score_abs = np.abs(compact_scores - neural_scores)
    compact_prob_abs = np.abs(compact_prob1 - neural_prob1)
    exact_score_abs = np.abs(exact_scores - neural_scores)
    exact_prob_abs = np.abs(exact_prob1 - neural_prob1)

    compact_margin = compact_scores[:, 1] - compact_scores[:, 0]
    neural_margin = neural_scores[:, 1] - neural_scores[:, 0]
    changed_margin_sign = int(np.sum(np.signbit(compact_margin) != np.signbit(neural_margin)))

    compact_acc_delta = float(compact_metrics["Accuracy"] - kes_metrics["Accuracy"])
    compact_auc_delta = float(compact_metrics["AUC"] - kes_metrics["AUC"])
    worst_err = np.max(compact_score_abs, axis=1)
    worst_idxs = np.argsort(worst_err)[::-1][:5]

    print("\nCompact extraction selected")
    print(
        f"policy={compact_selection_policy}, "
        f"top_k={compact_top_k}, "
        f"gate_threshold={compact_gate_threshold:.2f}"
    )

    print("\nFINAL COMPACT FORMULAS")
    print("y1(x1,x2,x3) =")
    print(f"  {finals['y1']}")
    print("y2(x1,x2,x3) =")
    print(f"  {finals['y2']}")
    negative = str(data["target_names"][0]) if "target_names" in data else "class 0"
    positive = str(data["target_names"][1]) if "target_names" in data else "class 1"
    print(f"\nDecision rule: predict {positive} iff y2 >= y1; otherwise predict {negative}.")

    print("\nDNF APPROXIMATION INPUT FORMULAS")
    if dnf_approx_decimals is None and dnf_approx_coeff_eps <= 0:
        print("No extra approximation before DNF; DNF will be equivalent to the compact formula.")
    else:
        dec_text = "exact" if dnf_approx_decimals is None else str(dnf_approx_decimals)
        print(f"Extra approximation before DNF: coefficient/bias rounding decimals={dec_text}, coeff_eps={dnf_approx_coeff_eps:g}")
    print("y1_DNF_input(x1,x2,x3) =")
    print(f"{dnf_finals['y1']}")
    print("y2_DNF_input(x1,x2,x3) =")
    print(f"  {dnf_finals['y2']}")

    print("\nFINAL APPROXIMATE DNF FORMULAS")
    dnf_scores = None
    dnf_metrics = None
    dnf_moisil_scores = None
    dnf_moisil_metrics = None
    try:
        dnf_obj_y1, dnf_vars_y1, dnf_piece_list_y1 = formula_to_dnf(dnf_finals["y1"])
        dnf_obj_y2, dnf_vars_y2, dnf_piece_list_y2 = formula_to_dnf(dnf_finals["y2"])
        dnf_ok_y1 = dnf_equivalence_on_grid(dnf_finals["y1"], dnf_obj_y1, dnf_vars_y1, step=Fraction(1, 4))
        dnf_ok_y2 = dnf_equivalence_on_grid(dnf_finals["y2"], dnf_obj_y2, dnf_vars_y2, step=Fraction(1, 4))
        dnf_pieces_y1 = len(dnf_piece_list_y1)
        dnf_pieces_y2 = len(dnf_piece_list_y2)
        dnf_y1 = dnf_obj_y1.pretty(dnf_vars_y1)
        dnf_y2 = dnf_obj_y2.pretty(dnf_vars_y2)
        dnf_map = {"y1": dnf_obj_y1, "y2": dnf_obj_y2}
        dnf_scores = dnf_outputs_for_X(dnf_map, data["X_test"], data["feature_names"])
        dnf_metrics = metrics_from_score_matrix(data["y_test"], dnf_scores)
        dnf_moisil_scores = moisil_discretize_scores(dnf_scores, n=n_moisil, u=1.0)
        dnf_moisil_metrics = metrics_from_score_matrix(data["y_test"], dnf_moisil_scores)

        print("y1_DNF(x1,x2,x3) =")
        print(f"  {dnf_y1}")
        print("y2_DNF(x1,x2,x3) =")
        print(f"  {dnf_y2}")
        print(f"DNF grid check: y1={dnf_ok_y1}, y2={dnf_ok_y2}; pieces: y1={dnf_pieces_y1}, y2={dnf_pieces_y2}")
    except Exception as e:
        dnf_y1 = dnf_y2 = "[DNF transform failed]"
        dnf_ok_y1 = dnf_ok_y2 = False
        dnf_pieces_y1 = dnf_pieces_y2 = 0
        print(f"DNF transform failed: {e}")

    kes_moisil_scores = moisil_discretize_scores(neural_scores, n=n_moisil, u=1.0)
    exact_moisil_scores = moisil_discretize_scores(exact_scores, n=n_moisil, u=1.0)
    compact_moisil_scores = moisil_discretize_scores(compact_scores, n=n_moisil, u=1.0)
    dnf_input_moisil_scores = moisil_discretize_scores(dnf_input_scores, n=n_moisil, u=1.0)

    named_scores = [
        ("KES neural", neural_scores),
        ("Exact formula", exact_scores),
        ("Compact formula", compact_scores),
        ("DNF input approx", dnf_input_scores),
        ("Moisil/Post(KES)", kes_moisil_scores),
        ("Moisil/Post(exact)", exact_moisil_scores),
        ("Moisil/Post(compact)", compact_moisil_scores),
        ("Moisil/Post(DNF input)", dnf_input_moisil_scores),
    ]
    if dnf_scores is not None:
        named_scores.insert(4, ("Approximate DNF", dnf_scores))
        named_scores.append(("Moisil/Post(DNF)", dnf_moisil_scores))

    metric_rows = [metric_audit_row(label, data["y_test"], scores, neural_scores) for label, scores in named_scores]
    audit_rows = [(label, score_matrix_audit(neural_scores, scores)) for label, scores in named_scores if
                  label != "KES neural"]

    print_metric_audit_table(metric_rows)
    print_pairwise_score_difference_table(named_scores)

    if dnf_scores is not None:
        dnf_vs_input = score_matrix_audit(dnf_input_scores, dnf_scores)
        dnf_moisil_vs_input_moisil = score_matrix_audit(dnf_input_moisil_scores, dnf_moisil_scores)
        print("\nDNF CONSISTENCY CHECK")
        print(f"DNF vs DNF-input score max error     : {dnf_vs_input['ScoreMaxErr']:.10f}")
        print(f"DNF vs DNF-input disagreements       : {dnf_vs_input['Disagree']}")
        print(f"Moisil(DNF) vs Moisil(DNF-input) err : {dnf_moisil_vs_input_moisil['ScoreMaxErr']:.10f}")
        print(f"Moisil(DNF) vs Moisil(DNF-input) diff: {dnf_moisil_vs_input_moisil['Disagree']}")

    print("\nCOMPACT FORMULA AUDIT — this is the main table")
    print(f"test samples                 : {len(data['X_test'])}")
    print(f"disagreements vs KES argmax  : {compact_disagree}")
    print(f"margin sign changes vs KES   : {changed_margin_sign}")
    print(f"fidelity vs KES              : {compact_test_quality['Fidelity']:.6f}")
    print(f"score MSE vs KES             : {compact_test_quality['MSE']:.10f}")
    print(f"score max abs error vs KES   : {float(np.max(compact_score_abs)):.8f}")
    print(f"probability max abs error    : {float(np.max(compact_prob_abs)):.8f}")
    print(f"unique compact score pairs   : {compact_test_quality['Unique']}")
    print(f"expanded formula length      : {compact_test_quality['Expand']}")

    print("\nTRUE-LABEL METRICS — shown only as a sanity check")
    print(f"KES accuracy / AUC           : {kes_metrics['Accuracy']:.6f} / {kes_metrics['AUC']:.8f}")
    print(f"Compact accuracy / AUC       : {compact_metrics['Accuracy']:.6f} / {compact_metrics['AUC']:.8f}")
    print(f"Compact minus KES            : accuracy {compact_acc_delta:+.6f}, AUC {compact_auc_delta:+.8f}")

    print("\nLARGEST SCORE DEVIATIONS — compact vs KES")
    print("{:>4} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}".format(
        "idx", "true", "KES_y1", "KES_y2", "cmp_y1", "cmp_y2", "max_err", "KESpred", "cmpPred"
    ))
    print("-" * 86)
    for idx in worst_idxs:
        print("{:>4d} {:>5d} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>8d} {:>8d}".format(
            int(idx), int(data["y_test"][idx]),
            float(neural_scores[idx, 0]), float(neural_scores[idx, 1]),
            float(compact_scores[idx, 0]), float(compact_scores[idx, 1]),
            float(worst_err[idx]), int(neural_pred_np[idx]), int(compact_pred_np[idx])
        ))

    report_path = os.path.join(outdir, "compact_formula_dnf_approx_tradeoff_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("KES compact symbolic extraction — minimal audit report\n")
        f.write(f"policy={compact_selection_policy}\n")
        f.write(f"top_k={compact_top_k}\n")
        f.write(f"gate_threshold={compact_gate_threshold:.6f}\n\n")
        f.write("Final compact formulas — expanded, no intermediate symbols\n")
        f.write(f"y1(x1,x2,x3) = {finals['y1']}\n")
        f.write(f"y2(x1,x2,x3) = {finals['y2']}\n")
        f.write(f"Decision rule: predict {positive} iff y2 >= y1; otherwise predict {negative}.\n\n")
        f.write("Final compact formulas in DNF\n")
        f.write(f"y1_DNF(x1,x2,x3) = {dnf_y1}\n")
        f.write(f"y2_DNF(x1,x2,x3) = {dnf_y2}\n")
        f.write(f"dnf_grid_check_y1={dnf_ok_y1}\n")
        f.write(f"dnf_grid_check_y2={dnf_ok_y2}\n")
        f.write(f"dnf_piece_count_y1={dnf_pieces_y1}\n")
        f.write(f"dnf_piece_count_y2={dnf_pieces_y2}\n\n")
        f.write("Metrics for all score sources — test split\n")
        for label, m in metric_rows:
            f.write(
                f"{label}: accuracy={m['Accuracy']:.8f}, precision={m['Precision']:.8f}, "
                f"recall={m['Recall']:.8f}, f1={m['F1-score']:.8f}, auc={m['AUC']:.10f}\n"
            )
        f.write("\nScore-level audit vs KES — test split\n")
        for label, a in audit_rows:
            f.write(
                f"{label}: disagree={int(a['Disagree'])}, fidelity={a['Fidelity']:.8f}, "
                f"score_mse={a['ScoreMSE']:.10f}, score_max_abs_error={a['ScoreMaxErr']:.10f}, "
                f"unique={int(a['Unique'])}\n"
            )
        if dnf_scores is not None:
            f.write("\nDNF consistency check\n")
            f.write(f"dnf_vs_dnf_input_score_max_error={dnf_vs_input['ScoreMaxErr']:.10f}\n")
            f.write(f"dnf_vs_dnf_input_disagreements={int(dnf_vs_input['Disagree'])}\n")
            f.write(
                f"moisil_dnf_vs_moisil_dnf_input_score_max_error={dnf_moisil_vs_input_moisil['ScoreMaxErr']:.10f}\n")
            f.write(f"moisil_dnf_vs_moisil_dnf_input_disagreements={int(dnf_moisil_vs_input_moisil['Disagree'])}\n")
        f.write("\n")

        f.write("Compact formula audit\n")
        f.write(f"test_samples={len(data['X_test'])}\n")
        f.write(f"disagreements_vs_kes_argmax={compact_disagree}\n")
        f.write(f"margin_sign_changes_vs_kes={changed_margin_sign}\n")
        f.write(f"fidelity_vs_kes={compact_test_quality['Fidelity']:.8f}\n")
        f.write(f"score_mse_vs_kes={compact_test_quality['MSE']:.10f}\n")
        f.write(f"score_max_abs_error_vs_kes={float(np.max(compact_score_abs)):.10f}\n")
        f.write(f"probability_max_abs_error={float(np.max(compact_prob_abs)):.10f}\n")
        f.write(f"unique_compact_score_pairs={compact_test_quality['Unique']}\n")
        f.write(f"expanded_formula_length={compact_test_quality['Expand']}\n\n")
        f.write("True-label sanity check\n")
        f.write(f"kes_accuracy={kes_metrics['Accuracy']:.8f}\n")
        f.write(f"kes_auc={kes_metrics['AUC']:.10f}\n")
        f.write(f"compact_accuracy={compact_metrics['Accuracy']:.8f}\n")
        f.write(f"compact_auc={compact_metrics['AUC']:.10f}\n")
        f.write(f"compact_minus_kes_accuracy={compact_acc_delta:+.8f}\n")
        f.write(f"compact_minus_kes_auc={compact_auc_delta:+.10f}\n\n")
        f.write("Largest score deviations — compact vs KES\n")
        f.write("idx,true,KES_y1,KES_y2,compact_y1,compact_y2,max_err,KES_pred,compact_pred\n")
        for idx in worst_idxs:
            f.write(
                f"{int(idx)},{int(data['y_test'][idx])},"
                f"{float(neural_scores[idx, 0]):.10f},{float(neural_scores[idx, 1]):.10f},"
                f"{float(compact_scores[idx, 0]):.10f},{float(compact_scores[idx, 1]):.10f},"
                f"{float(worst_err[idx]):.10f},{int(neural_pred_np[idx])},{int(compact_pred_np[idx])}\n"
            )

    print(f"\nDNF/Moisil metrics report saved to: {report_path}")