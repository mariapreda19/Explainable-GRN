from __future__ import annotations
import os
import time
import itertools
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Tuple, Union, Optional
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

U_VALUE = 2.0


class Formula:
    def to_string(self):
        raise NotImplementedError

    def __str__(self):
        return self.to_string()


class Bottom(Formula):
    def to_string(self):
        return "⊥"


class Top(Formula):
    def to_string(self):
        return "⊤"


class SymbolRef(Formula):
    def __init__(self, name: str):
        self.name = name

    def to_string(self):
        return self.name


class Diamond(Formula):
    def __init__(self, coeff: float, child: Formula):
        self.coeff = float(coeff)
        self.child = child

    def to_string(self):
        c = f"{self.coeff:.6f}".rstrip("0").rstrip(".")
        if c == "-0":
            c = "0"
        return f"◇_{c}({self.child})"


class Not(Formula):
    def __init__(self, child: Formula):
        self.child = child

    def to_string(self):
        return f"¬({self.child})"


class Oplus(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def to_string(self):
        return f"({self.left} ⊕ {self.right})"


class Odot(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right

    def to_string(self):
        return f"({self.left} ⊙ {self.right})"


class And(Formula):
    def __init__(self, *children):
        self.children = list(children)

    def to_string(self):
        if len(self.children) == 0:
            return "⊤"
        if len(self.children) == 1:
            return str(self.children[0])
        return "(" + " ∧ ".join(str(c) for c in self.children) + ")"


class Or(Formula):
    def __init__(self, *children):
        self.children = list(children)

    def to_string(self):
        if len(self.children) == 0:
            return "⊥"
        if len(self.children) == 1:
            return str(self.children[0])
        return "(" + " ∨ ".join(str(c) for c in self.children) + ")"


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

    return formula


def linear_combination(L, eps=1e-12):
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


def crelu_u(x, u=U_VALUE):
    return torch.clamp(x, 0.0, u)


class CReLU_U(nn.Module):
    def __init__(self, u=U_VALUE):
        super().__init__()
        self.u = u

    def forward(self, x):
        return crelu_u(x, self.u)


class SmoothMinLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            u: float = U_VALUE,
            mu: float = 4.0,
            temperature: float = 1.0,
            dropout_p: float = 0.0,
            eps: float = 1e-8,
    ):
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


class SmoothMaxLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            u: float = U_VALUE,
            mu: float = 4.0,
            temperature: float = 1.0,
            dropout_p: float = 0.0,
            eps: float = 1e-8,
    ):
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

        num = torch.logsumexp(self.mu * x_exp + logw, dim=-1)
        den = torch.logsumexp(logw, dim=-1).squeeze(0)

        y = (num - den.unsqueeze(0)) / self.mu
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


class ClassKESPath(nn.Module):
    def __init__(
            self,
            input_dim=2,
            num_hyperplanes=6,
            min_layer_dims=(4,),
            max_layer_dims=(2, 1),
            u: float = U_VALUE,
            mu=4.0,
            temperature=1.0,
            dropout_p=0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_hyperplanes = num_hyperplanes
        self.min_layer_dims = list(min_layer_dims)
        self.max_layer_dims = list(max_layer_dims)
        self.u = u

        if len(self.max_layer_dims) == 0:
            raise ValueError("Trebuie sa existe cel putin un strat max.")
        if self.max_layer_dims[-1] != 1:
            raise ValueError("Ultimul strat max trebuie sa aiba exact 1 neuron.")

        self.linear = nn.Linear(input_dim, num_hyperplanes)
        self.linear_act = CReLU_U(u=u)

        self.min_layers = nn.ModuleList()
        prev_dim = num_hyperplanes
        for out_dim in self.min_layer_dims:
            self.min_layers.append(
                SmoothMinLayer(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    u=u,
                    mu=mu,
                    temperature=temperature,
                    dropout_p=dropout_p,
                )
            )
            prev_dim = out_dim

        self.max_layers = nn.ModuleList()
        for out_dim in self.max_layer_dims:
            self.max_layers.append(
                SmoothMaxLayer(
                    in_dim=prev_dim,
                    out_dim=out_dim,
                    u=u,
                    mu=mu,
                    temperature=temperature,
                    dropout_p=dropout_p,
                )
            )
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
            class_idx: int,
            feature_names=None,
            eps=1e-8,
            selector_threshold=0.5,
            verbose=False,
    ):
        symbolic_defs = []

        if feature_names is None:
            input_refs = [SymbolRef(f"x{i + 1}") for i in range(self.input_dim)]
        else:
            input_refs = [SymbolRef(str(name)) for name in feature_names]

        W = self.linear.weight.detach().cpu().numpy()
        b = self.linear.bias.detach().cpu().numpy()

        prev_refs = []

        iterator = range(self.num_hyperplanes)
        if verbose and tqdm is not None:
            iterator = tqdm(iterator, desc=f"Class {class_idx + 1} hyperplanes", leave=True)

        for j in iterator:
            name = f"c{class_idx + 1}_p{j + 1}"
            coeffs = [(float(W[j, i]), input_refs[i]) for i in range(self.input_dim)]
            coeffs.append((float(b[j]), 0))

            formula = simplify(linear_combination(coeffs, eps=eps))
            symbolic_defs.append((name, formula))
            prev_refs.append(SymbolRef(name))

        for layer_idx, layer in enumerate(self.min_layers, start=1):
            groups = layer.extract_connectivity(selector_threshold=selector_threshold)
            current_refs = []

            for j, idxs in enumerate(groups):
                name = f"c{class_idx + 1}_m{layer_idx}_{j + 1}"
                children = [prev_refs[i] for i in idxs]
                formula = simplify(And(*children))
                symbolic_defs.append((name, formula))
                current_refs.append(SymbolRef(name))
            prev_refs = current_refs

        for layer_idx, layer in enumerate(self.max_layers, start=1):
            groups = layer.extract_connectivity(selector_threshold=selector_threshold)
            current_refs = []

            for j, idxs in enumerate(groups):
                name = f"c{class_idx + 1}_u{layer_idx}_{j + 1}"
                children = [prev_refs[i] for i in idxs]
                formula = simplify(Or(*children))
                symbolic_defs.append((name, formula))
                current_refs.append(SymbolRef(name))
            prev_refs = current_refs

        symbolic_defs.append((f"y{class_idx + 1}", prev_refs[0]))

        return symbolic_defs

    def get_active_connections(self, selector_threshold=0.5):
        info = {"min": [], "max": []}
        for layer in self.min_layers:
            info["min"].append(layer.extract_connectivity(selector_threshold=selector_threshold))
        for layer in self.max_layers:
            info["max"].append(layer.extract_connectivity(selector_threshold=selector_threshold))
        return info


class MultiClassKESPaths(nn.Module):
    def __init__(
            self,
            input_dim=2,
            num_classes=2,
            num_hyperplanes=6,
            min_layer_dims=(4,),
            max_layer_dims=(2, 1),
            u: float = U_VALUE,
            mu=4.0,
            temperature=1.0,
            dropout_p=0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.u = u

        self.paths = nn.ModuleList([
            ClassKESPath(
                input_dim=input_dim,
                num_hyperplanes=num_hyperplanes,
                min_layer_dims=min_layer_dims,
                max_layer_dims=max_layer_dims,
                u=u,
                mu=mu,
                temperature=temperature,
                dropout_p=dropout_p,
            )
            for _ in range(num_classes)
        ])

    def forward(self, x):
        scores = [path(x) for path in self.paths]
        logits = torch.stack(scores, dim=1)
        return logits

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def extract_symbolic_system(
            self,
            feature_names=None,
            eps=1e-8,
            selector_threshold=0.5,
            verbose=False,
    ):
        symbolic_defs = []

        t_global = time.time()

        iterator = range(len(self.paths))
        if verbose and tqdm is not None:
            iterator = tqdm(iterator, desc="Classes", leave=True)

        for c in iterator:
            path = self.paths[c]
            part = path.extract_symbolic_path(
                class_idx=c,
                feature_names=feature_names,
                eps=eps,
                selector_threshold=selector_threshold,
                verbose=verbose,
            )
            symbolic_defs.extend(part)

        return symbolic_defs

    def print_active_connections(self, selector_threshold=0.5):
        for c, path in enumerate(self.paths, start=1):
            info = path.get_active_connections(selector_threshold=selector_threshold)
            print(f"\nCLASS {c}")

            for li, groups in enumerate(info["min"], start=1):
                print(f"  [MIN layer {li}]")
                for j, idxs in enumerate(groups, start=1):
                    print(f"    neuron {j}: {idxs}")

            for li, groups in enumerate(info["max"], start=1):
                print(f"  [MAX layer {li}]")
                for j, idxs in enumerate(groups, start=1):
                    print(f"    neuron {j}: {idxs}")


def print_gate_values(model):
    for c, path in enumerate(model.paths, start=1):
        print(f"\nCLASS {c} GATES")

        for li, layer in enumerate(path.min_layers, start=1):
            g = layer.gate_values().detach().cpu().numpy()
            print(f"\nMIN layer {li} gates:")
            print(np.round(g, 4))

        for li, layer in enumerate(path.max_layers, start=1):
            g = layer.gate_values().detach().cpu().numpy()
            print(f"\nMAX layer {li} gates:")
            print(np.round(g, 4))


def F(x: Union[int, float, str, Fraction]) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, float):
        return Fraction(str(x))
    return Fraction(x)


U = F(U_VALUE)
ZERO = F(0)


def frac_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    s = f"{float(q):.10f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


@dataclass
class Affine:
    coef: Dict[str, Fraction] = field(default_factory=dict)
    const: Fraction = Fraction(0, 1)

    def __add__(self, other: "Affine") -> "Affine":
        keys = set(self.coef) | set(other.coef)
        out: Dict[str, Fraction] = {}
        for k in keys:
            v = self.coef.get(k, Fraction(0)) + other.coef.get(k, Fraction(0))
            if v != 0:
                out[k] = v
        return Affine(out, self.const + other.const)

    def __sub__(self, other: "Affine") -> "Affine":
        keys = set(self.coef) | set(other.coef)
        out: Dict[str, Fraction] = {}
        for k in keys:
            v = self.coef.get(k, Fraction(0)) - other.coef.get(k, Fraction(0))
            if v != 0:
                out[k] = v
        return Affine(out, self.const - other.const)

    def __mul__(self, scalar: Union[int, float, str, Fraction]) -> "Affine":
        s = F(scalar)
        out: Dict[str, Fraction] = {}
        for k, v in self.coef.items():
            w = v * s
            if w != 0:
                out[k] = w
        return Affine(out, self.const * s)

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
        if c is None:
            return True
        return self.const == c

    def cube_minmax(self, variables: List[str]) -> Tuple[Fraction, Fraction]:
        mn = self.const
        mx = self.const
        for var in variables:
            a = self.coef.get(var, Fraction(0))
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
                parts.append(f"{frac_str(a)}{var}")

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
class Inequality:
    expr: Affine

    @staticmethod
    def le(lhs: Affine, rhs: Affine) -> "Inequality":
        return Inequality(lhs - rhs)

    @staticmethod
    def ge(lhs: Affine, rhs: Affine) -> "Inequality":
        return Inequality(rhs - lhs)

    def key(self):
        return self.expr.key()

    def pretty(self) -> str:
        return f"{self.expr.pretty()} <= 0"


@dataclass
class Region:
    constraints: List[Inequality] = field(default_factory=list)

    def add(self, ineq: Inequality) -> "Region":
        return Region(self.constraints + [ineq])

    def extend(self, other: "Region") -> "Region":
        return Region(self.constraints + other.constraints)

    def dedup(self) -> "Region":
        seen = set()
        out: List[Inequality] = []
        for c in self.constraints:
            k = c.key()
            if k not in seen:
                seen.add(k)
                out.append(c)
        return Region(out)

    def pretty(self) -> str:
        r = self.dedup()
        if not r.constraints:
            return "True"
        return " AND ".join(c.pretty() for c in r.constraints)


@dataclass
class Piece:
    region: Region
    affine: Affine

    def pretty(self) -> str:
        return f"If [{self.region.pretty()}] then {self.affine.pretty()}"


class Expr:
    pass


@dataclass
class Var(Expr):
    name: str


@dataclass
class Const(Expr):
    value: Fraction


@dataclass
class Neg(Expr):
    sub: Expr


@dataclass
class OPlusExpr(Expr):
    left: Expr
    right: Expr


@dataclass
class ODotExpr(Expr):
    left: Expr
    right: Expr


@dataclass
class SMul(Expr):
    scalar: Fraction
    sub: Expr


TOKEN_RE = re.compile(
    r"""\s*(
        smul
      | oplus
      | odot
      | [A-Za-z_][A-Za-z0-9_]*
      | \d+\.\d+|\d+/\d+|\d+
      | \(
      | \)
      | ,
      | \*
    )""",
    re.VERBOSE,
)


class Parser:
    def __init__(self, text: str) -> None:
        self.tokens = [t for t in TOKEN_RE.findall(text)]
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def eat(self, tok: Optional[str] = None) -> str:
        cur = self.peek()
        if cur is None:
            raise ValueError("Unexpected end of input")
        if tok is not None and cur != tok:
            raise ValueError(f"Expected {tok!r}, got {cur!r}")
        self.pos += 1
        return cur

    def parse(self) -> Expr:
        e = self.parse_oplus()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token after parse: {self.peek()!r}")
        return e

    def parse_oplus(self) -> Expr:
        left = self.parse_odot()
        while self.peek() == "oplus":
            self.eat("oplus")
            right = self.parse_odot()
            left = OPlusExpr(left, right)
        return left

    def parse_odot(self) -> Expr:
        left = self.parse_postfix()
        while self.peek() == "odot":
            self.eat("odot")
            right = self.parse_postfix()
            left = ODotExpr(left, right)
        return left

    def parse_postfix(self) -> Expr:
        node = self.parse_atom()
        while self.peek() == "*":
            self.eat("*")
            node = Neg(node)
        return node

    def parse_atom(self) -> Expr:
        tok = self.peek()
        if tok == "(":
            self.eat("(")
            e = self.parse_oplus()
            self.eat(")")
            return e

        if tok == "smul":
            self.eat("smul")
            self.eat("(")
            scalar = F(self.eat())
            self.eat(",")
            sub = self.parse_oplus()
            self.eat(")")
            return SMul(scalar, sub)

        if tok is not None and re.fullmatch(r"\d+\.\d+|\d+/\d+|\d+", tok):
            self.eat()
            return Const(F(tok))

        if tok is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            self.eat()
            return Var(tok)

        raise ValueError(f"Unexpected token: {tok!r}")


def parse_formula(text: str) -> Expr:
    return Parser(text).parse()


def merge_vars(expr: Expr) -> List[str]:
    out = set()

    def rec(e: Expr) -> None:
        if isinstance(e, Var):
            out.add(e.name)
        elif isinstance(e, Const):
            return
        elif isinstance(e, Neg):
            rec(e.sub)
        elif isinstance(e, (OPlusExpr, ODotExpr)):
            rec(e.left)
            rec(e.right)
        elif isinstance(e, SMul):
            rec(e.sub)
        else:
            raise TypeError(type(e))

    rec(expr)
    return sorted(out)


def dedup_pieces(pieces: List[Piece]) -> List[Piece]:
    seen = set()
    out: List[Piece] = []
    for p in pieces:
        key = (tuple(c.key() for c in p.region.dedup().constraints), p.affine.key())
        if key not in seen:
            seen.add(key)
            out.append(Piece(p.region.dedup(), p.affine))
    return out


def piecewise_affine(expr: Expr) -> List[Piece]:
    def rec(e: Expr) -> List[Piece]:
        if isinstance(e, Var):
            return [Piece(Region(), Affine({e.name: F(1)}, F(0)))]

        if isinstance(e, Const):
            return [Piece(Region(), Affine({}, e.value))]

        if isinstance(e, Neg):
            return [Piece(p.region, Affine({}, U) - p.affine) for p in rec(e.sub)]

        if isinstance(e, SMul):
            return [Piece(p.region, e.scalar * p.affine) for p in rec(e.sub)]

        if isinstance(e, OPlusExpr):
            out: List[Piece] = []
            top = Affine({}, U)
            for p, q in itertools.product(rec(e.left), rec(e.right)):
                base = p.region.extend(q.region)
                s = p.affine + q.affine
                out.append(Piece(base.add(Inequality.le(s, top)), s))
                out.append(Piece(base.add(Inequality.ge(s, top)), top))
            return out

        if isinstance(e, ODotExpr):
            out: List[Piece] = []
            top = Affine({}, U)
            zero = Affine({}, ZERO)
            for p, q in itertools.product(rec(e.left), rec(e.right)):
                base = p.region.extend(q.region)
                s = p.affine + q.affine
                out.append(Piece(base.add(Inequality.le(s, top)), zero))
                out.append(Piece(base.add(Inequality.ge(s, top)), s - top))
            return out

        raise TypeError(type(e))

    return dedup_pieces(rec(expr))


def cube_constraints(variables: List[str]) -> List[Inequality]:
    out: List[Inequality] = []
    for v in variables:
        out.append(Inequality.le(Affine({v: F(1)}, F(0)), Affine({}, U)))
        out.append(Inequality.le(Affine({v: F(-1)}, F(0)), Affine({}, ZERO)))
    return out


def region_with_cube(region: Region, variables: List[str]) -> Region:
    return Region(region.constraints + cube_constraints(variables)).dedup()


def solve_linear_system_fraction(A: List[List[Fraction]], b: List[Fraction]):
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


def satisfies_region(point: Dict[str, Fraction], region: Region) -> bool:
    return all(c.expr.eval(point) <= 0 for c in region.constraints)


def poly_vertices(region: Region, variables: List[str]) -> List[Dict[str, Fraction]]:
    full = region_with_cube(region, variables)
    cons = full.constraints
    n = len(variables)

    if n == 0:
        return [{}] if satisfies_region({}, full) else []
    vertices: List[Dict[str, Fraction]] = []
    seen = set()
    for idxs in itertools.combinations(range(len(cons)), n):
        A: List[List[Fraction]] = []
        b: List[Fraction] = []
        for i in idxs:
            expr = cons[i].expr
            A.append([expr.coef.get(v, Fraction(0)) for v in variables])
            b.append(-expr.const)
        sol = solve_linear_system_fraction(A, b)
        if sol is None:
            continue

        point = {v: sol[k] for k, v in enumerate(variables)}
        if not satisfies_region(point, full):
            continue

        key = tuple(point[v] for v in variables)
        if key not in seen:
            seen.add(key)
            vertices.append(point)

    if not vertices:
        for bits in itertools.product([Fraction(0), U], repeat=n):
            point = {v: bits[i] for i, v in enumerate(variables)}
            if satisfies_region(point, full):
                key = tuple(point[v] for v in variables)
                if key not in seen:
                    seen.add(key)
                    vertices.append(point)

    return vertices


def region_is_empty(region: Region, variables: List[str]) -> bool:
    return len(poly_vertices(region, variables)) == 0


def affine_minmax_on_region(aff: Affine, region: Region, variables: List[str]) -> Tuple[Fraction, Fraction]:
    verts = poly_vertices(region, variables)
    if not verts:
        raise ValueError("Empty region encountered")
    vals = [aff.eval(v) for v in verts]
    return min(vals), max(vals)


@dataclass
class ClipAffine:
    affine: Affine

    def key(self):
        return self.affine.key()

    def pretty(self, variables: List[str], drop_lower_clip: bool = False) -> str:
        mn, mx = self.affine.cube_minmax(variables)
        a = self.affine.pretty()
        u_text = frac_str(U)

        actual_need_lower = (mn < 0) and (not drop_lower_clip)

        actual_need_upper = (mx > U)

        if not actual_need_lower and not actual_need_upper:
            return a
        if actual_need_lower and not actual_need_upper:
            return f"({a} ∨ 0)"
        if not actual_need_lower and actual_need_upper:
            return f"({a} ∧ {u_text})"
        return f"(({a} ∨ 0) ∧ {u_text})"


@dataclass
class MeetTerm:
    factors: List[ClipAffine]

    def pretty(self, variables: List[str], drop_lower_clip: bool = False) -> str:
        if not self.factors:
            return frac_str(U)

        texts = []
        has_nonnegative_factor = False

        for f in self.factors:
            mn, _ = f.affine.cube_minmax(variables)
            if mn >= 0:
                has_nonnegative_factor = True
                break

        for f in self.factors:
            local_drop = drop_lower_clip or has_nonnegative_factor
            texts.append(f.pretty(variables, drop_lower_clip=local_drop))

        return " ∧ ".join(texts)


@dataclass
class DNF:
    clauses: List[MeetTerm]

    def pretty(self, variables: List[str]) -> str:
        if not self.clauses:
            return "0"

        drop_lower_clip = len(self.clauses) > 1

        clause_texts = []
        for c in self.clauses:
            txt = c.pretty(variables, drop_lower_clip=drop_lower_clip)
            if txt != "0":
                clause_texts.append(txt)

        if not clause_texts:
            return "0"

        return " ∨ ".join(clause_texts)


def eval_expr(expr: Expr, env: Dict[str, Fraction]) -> Fraction:
    if isinstance(expr, Var):
        return env[expr.name]
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Neg):
        return U - eval_expr(expr.sub, env)
    if isinstance(expr, OPlusExpr):
        return min(U, eval_expr(expr.left, env) + eval_expr(expr.right, env))
    if isinstance(expr, ODotExpr):
        return max(ZERO, eval_expr(expr.left, env) + eval_expr(expr.right, env) - U)
    if isinstance(expr, SMul):
        return expr.scalar * eval_expr(expr.sub, env)
    raise TypeError(type(expr))


def clip0u(q: Fraction) -> Fraction:
    if q < ZERO:
        return ZERO
    if q > U:
        return U
    return q


def eval_clip_affine(c: ClipAffine, env: Dict[str, Fraction]) -> Fraction:
    return clip0u(c.affine.eval(env))


def eval_meet(term: MeetTerm, env):
    if not term.factors:
        return U
    return min(eval_clip_affine(f, env) for f in term.factors)


def eval_dnf(dnf: DNF, env: Dict[str, Fraction]) -> Fraction:
    if not dnf.clauses:
        return F(0)
    return max(eval_meet(c, env) for c in dnf.clauses)


def unique_affines(items: List[Affine]) -> List[Affine]:
    seen = set()
    out: List[Affine] = []
    for a in items:
        k = a.key()
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def simplify_meet(term: MeetTerm, variables: List[str]) -> MeetTerm:
    uniq: Dict[Tuple, ClipAffine] = {}
    for f in term.factors:
        uniq[f.key()] = f

    factors = list(uniq.values())
    keep: List[ClipAffine] = []

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

            if mn_diff >= 0:
                redundant = True
                break

        if not redundant:
            keep.append(f)

    return MeetTerm(keep)


def simplify_dnf(dnf: DNF, variables: List[str]) -> DNF:
    seen = set()
    out: List[MeetTerm] = []

    for c in dnf.clauses:
        sc = simplify_meet(c, variables)
        key = tuple(sorted(f.key() for f in sc.factors))
        if key not in seen:
            seen.add(key)
            out.append(sc)

    return DNF(out)


def clause_from_piece(target: Piece, components: List[Affine], variables: List[str]) -> Optional[MeetTerm]:
    L = target.affine
    if L.is_const(ZERO):
        return None
    if L.is_const(U):
        return None

    factors: List[ClipAffine] = []

    for A in components:
        diff = A - L
        mn, _ = affine_minmax_on_region(diff, target.region, variables)
        if mn >= 0:
            if A.is_const(U):
                continue
            factors.append(ClipAffine(A))

    if not any(f.affine.key() == L.key() for f in factors):
        factors.append(ClipAffine(L))

    return simplify_meet(MeetTerm(factors), variables)


def disjunctive_normal_form(expr: Expr) -> DNF:
    variables = merge_vars(expr)
    raw_pieces = piecewise_affine(expr)
    pieces = [p for p in raw_pieces if not region_is_empty(p.region, variables)]

    if not pieces:
        return DNF([])

    components = unique_affines([p.affine for p in pieces])
    nonconst = [a for a in components if not a.is_const()]
    if not nonconst:
        if any(a.is_const(U) for a in components):
            return DNF([MeetTerm([])])
        return DNF([])

    clauses: List[MeetTerm] = []
    for p in pieces:
        clause = clause_from_piece(p, components, variables)
        if clause is not None and len(clause.factors) > 0:
            is_zero = any(f.affine.is_const(ZERO) for f in clause.factors)
            if not is_zero:
                clauses.append(clause)

    return simplify_dnf(DNF(clauses), variables)


@dataclass
class TransformResult:
    expr: Expr
    variables: List[str]
    pieces: List[Piece]
    dnf: DNF
    final_unicode: str


def transform_formula(text: str) -> TransformResult:
    expr = parse_formula(text)
    variables = merge_vars(expr)
    pieces = [p for p in piecewise_affine(expr) if not region_is_empty(p.region, variables)]
    dnf = disjunctive_normal_form(expr)
    return TransformResult(
        expr=expr,
        variables=variables,
        pieces=pieces,
        dnf=dnf,
        final_unicode=dnf.pretty(variables),
    )


def rational_grid(step: Fraction) -> List[Fraction]:
    n = int(U / step)
    return [step * k for k in range(n + 1)]


def verify_on_grid(expr: Expr, dnf: DNF, variables: List[str], step: Fraction = Fraction(1, 2)) -> bool:
    if len(variables) > 2:
        return True

    values = rational_grid(step)
    for tup in itertools.product(values, repeat=len(variables)):
        env = {variables[i]: tup[i] for i in range(len(variables))}
        lhs = eval_expr(expr, env)
        rhs = eval_dnf(dnf, env)
        if lhs != rhs:
            print(
                "Mismatch at",
                {k: frac_str(v) for k, v in env.items()},
                "expr=",
                frac_str(lhs),
                "dnf=",
                frac_str(rhs),
            )
            return False
    return True


def build_definition_map(symbolic_defs):
    return {name: formula for name, formula in symbolic_defs}


def dedup_formula_children(children):
    seen = set()
    out = []
    for c in children:
        s = str(c)
        if s not in seen:
            seen.add(s)
            out.append(c)
    return out


def expand_formula_all(formula, def_map):
    if isinstance(formula, SymbolRef):
        name = formula.name
        if name in def_map:
            return expand_formula_all(def_map[name], def_map)
        return formula

    if isinstance(formula, And):
        children = [expand_formula_all(c, def_map) for c in formula.children]
        flat = []
        for c in children:
            if isinstance(c, And):
                flat.extend(c.children)
            else:
                flat.append(c)
        flat = dedup_formula_children(flat)
        return simplify(And(*flat))

    if isinstance(formula, Or):
        children = [expand_formula_all(c, def_map) for c in formula.children]
        flat = []
        for c in children:
            if isinstance(c, Or):
                flat.extend(c.children)
            else:
                flat.append(c)
        flat = dedup_formula_children(flat)
        return simplify(Or(*flat))

    if isinstance(formula, Not):
        return simplify(Not(expand_formula_all(formula.child, def_map)))

    if isinstance(formula, Oplus):
        return simplify(Oplus(
            expand_formula_all(formula.left, def_map),
            expand_formula_all(formula.right, def_map)
        ))

    if isinstance(formula, Odot):
        return simplify(Odot(
            expand_formula_all(formula.left, def_map),
            expand_formula_all(formula.right, def_map)
        ))

    if isinstance(formula, Diamond):
        return simplify(Diamond(formula.coeff, expand_formula_all(formula.child, def_map)))

    return formula


def get_fully_expanded_formulas(symbolic_defs):
    def_map = build_definition_map(symbolic_defs)
    finals = {}
    for name, formula in symbolic_defs:
        if name.startswith("y"):
            finals[name] = simplify(expand_formula_all(formula, def_map))
    return finals


def formula_to_inequality_text(formula, pred_map):
    if isinstance(formula, SymbolRef):
        return pred_map.get(formula.name, formula.name)

    if isinstance(formula, Bottom):
        return "⊥"

    if isinstance(formula, Top):
        return "⊤"

    if isinstance(formula, Not):
        return f"¬({formula_to_inequality_text(formula.child, pred_map)})"

    if isinstance(formula, And):
        if len(formula.children) == 0:
            return "⊤"
        if len(formula.children) == 1:
            return formula_to_inequality_text(formula.children[0], pred_map)
        return "(" + " ∧ ".join(formula_to_inequality_text(c, pred_map) for c in formula.children) + ")"

    if isinstance(formula, Or):
        if len(formula.children) == 0:
            return "⊥"
        if len(formula.children) == 1:
            return formula_to_inequality_text(formula.children[0], pred_map)
        return "(" + " ∨ ".join(formula_to_inequality_text(c, pred_map) for c in formula.children) + ")"

    if isinstance(formula, Diamond):
        child = formula_to_inequality_text(formula.child, pred_map)
        coeff = f"{formula.coeff:.6f}".rstrip("0").rstrip(".")
        return f"◇_{coeff}({child})"

    if isinstance(formula, Oplus):
        return f"({formula_to_inequality_text(formula.left, pred_map)} ⊕ {formula_to_inequality_text(formula.right, pred_map)})"

    if isinstance(formula, Odot):
        return f"({formula_to_inequality_text(formula.left, pred_map)} ⊙ {formula_to_inequality_text(formula.right, pred_map)})"

    return str(formula)


def affine_expr_from_weights(w, b, feature_names=None, decimals=4):
    terms = []

    for i, coeff in enumerate(w):
        coeff = float(coeff)
        c = round(coeff, decimals)
        if abs(c) < 10 ** (-decimals):
            continue

        var_name = feature_names[i] if feature_names is not None else f"x{i + 1}"

        if len(terms) == 0:
            terms.append(f"{c}*{var_name}")
        else:
            if c >= 0:
                terms.append(f"+ {c}*{var_name}")
            else:
                terms.append(f"- {abs(c)}*{var_name}")

    b = round(float(b), decimals)
    if abs(b) >= 10 ** (-decimals):
        if len(terms) == 0:
            terms.append(f"{b}")
        else:
            if b >= 0:
                terms.append(f"+ {b}")
            else:
                terms.append(f"- {abs(b)}")

    if len(terms) == 0:
        return "0"

    return " ".join(terms)


def get_predicate_inequality_map(model, feature_names=None, decimals=4, relation=">= 0"):
    pred_map = {}
    for class_idx, path in enumerate(model.paths, start=1):
        W = path.linear.weight.detach().cpu().numpy()
        b = path.linear.bias.detach().cpu().numpy()

        for j in range(W.shape[0]):
            name = f"c{class_idx}_p{j + 1}"
            expr = affine_expr_from_weights(W[j], b[j], feature_names=feature_names, decimals=decimals)
            pred_map[name] = f"({expr} {relation})"
    return pred_map


def formula_to_parser_text(formula: Formula) -> str:
    if isinstance(formula, Bottom):
        return "0"

    if isinstance(formula, Top):
        return frac_str(U)

    if isinstance(formula, SymbolRef):
        return formula.name

    if isinstance(formula, Not):
        child = formula_to_parser_text(formula.child)
        return f"({child})*"

    if isinstance(formula, Diamond):
        if isinstance(formula.child, Top):
            return frac_str(F(formula.coeff))
        child = formula_to_parser_text(formula.child)
        return f"smul({formula.coeff},{child})"

    if isinstance(formula, Oplus):
        left = formula_to_parser_text(formula.left)
        right = formula_to_parser_text(formula.right)
        return f"({left} oplus {right})"

    if isinstance(formula, Odot):
        left = formula_to_parser_text(formula.left)
        right = formula_to_parser_text(formula.right)
        return f"({left} odot {right})"

    if isinstance(formula, And):
        if len(formula.children) == 0:
            return frac_str(U)
        txt = formula_to_parser_text(formula.children[0])
        for c in formula.children[1:]:
            txt = f"({txt} odot {formula_to_parser_text(c)})"
        return txt

    if isinstance(formula, Or):
        if len(formula.children) == 0:
            return "0"
        txt = formula_to_parser_text(formula.children[0])
        for c in formula.children[1:]:
            txt = f"({txt} oplus {formula_to_parser_text(c)})"
        return txt

    raise TypeError(type(formula))


def load_breast_cancer_dataset_pca_same_then_to_interval(
        n_components=3,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        u_value=U_VALUE,
):
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    target_names = data.target_names

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_relative_size,
        random_state=random_state, stratify=y_train_full
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

    pca_feature_names = [f"x{i + 1}" for i in range(n_components)]

    return {
        "X_train": X_train_0u,
        "X_val": X_val_0u,
        "X_test": X_test_0u,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": pca_feature_names,
        "target_names": target_names,
        "scaler_raw": scaler_raw,
        "pca": pca,
        "scaler_pca_interval": scaler_pca_interval,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def train_model(
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=2000,
        lr=5e-2,
        batch_size=64,
        weight_decay=1e-5,
        print_every=100,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            train_pred = train_logits.argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()
            train_loss = criterion(train_logits, y_train).item()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_logits = model(X_val)
                val_pred = val_logits.argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean().item()
                val_loss = criterion(val_logits, y_val).item()

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            else:
                val_loss = None
                val_acc = None

        if epoch % print_every == 0 or epoch == 1:
            msg = (
                f"Epoch {epoch:4d} | "
                f"TrainLoss = {train_loss:.6f} | TrainAcc = {train_acc:.4f}"
            )
            if val_loss is not None:
                msg += f" | ValLoss = {val_loss:.6f} | ValAcc = {val_acc:.4f}"
            print(msg)

    return history


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


def print_final_interpretable_formulas(symbolic_defs):
    finals = get_fully_expanded_formulas(symbolic_defs)

    print("final extracted formulas")
    for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
        print(f"{name} := {finals[name]}")


def print_final_interpretable_inequalities(model, symbolic_defs, feature_names=None, decimals=4):
    finals = get_fully_expanded_formulas(symbolic_defs)
    pred_map = get_predicate_inequality_map(model, feature_names=feature_names, decimals=decimals, relation=">= 0")

    print("final formulas as inequalities")
    for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
        txt = formula_to_inequality_text(finals[name], pred_map)
        print(f"{name} := {txt}")


def print_final_dnf_results(symbolic_defs, verify_step=Fraction(1, 2)):
    finals = get_fully_expanded_formulas(symbolic_defs)

    print("final dnf formulas")
    for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
        parser_text = formula_to_parser_text(finals[name])

        try:
            result = transform_formula(parser_text)
            ok = verify_on_grid(result.expr, result.dnf, result.variables, step=verify_step)

            print(f"{name} := {result.final_unicode}")
            if len(result.variables) <= 2:
                print(f"  grid verif: {ok}")
            else:
                print(f"  grid verif: skipped exact check for {len(result.variables)} vars")
        except Exception as e:
            print(f"{name} := [DNF transform failed]")
            print(f"  parser_text = {parser_text}")
            print(f"  error = {e}")


def export_symbolic_report_with_dnf(
        model,
        symbolic_defs,
        filepath,
        feature_names=None,
        decimals=4,
        verify_step=Fraction(1, 2),
        benchmark_results=None,
        clean_func=None
):
    finals = get_fully_expanded_formulas(symbolic_defs)
    pred_map = get_predicate_inequality_map(model, feature_names=feature_names, decimals=decimals, relation=">= 0")

    with open(filepath, "w", encoding="utf-8") as f:
        if benchmark_results:
            f.write("=== MODEL COMPARISON BENCHMARK ===\n")
            template = "{:<25} | {:<12} | {:<12}\n"
            f.write(template.format("Model", "Accuracy", "ROC-AUC"))
            f.write("-" * 55 + "\n")
            for res in benchmark_results:
                f.write(template.format(res['Model'], f"{res['Accuracy']:.4f}", f"{res['ROC-AUC']:.4f}"))
            f.write("\n\n")

        f.write("=== FINAL INTERPRETABLE FORMULAS ===\n")
        for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
            f.write(f"{name} := {finals[name]}\n")

        f.write("\n=== FINAL INTERPRETABLE INEQUALITIES ===\n")
        for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
            txt = formula_to_inequality_text(finals[name], pred_map)
            f.write(f"{name} := {txt}\n")

        f.write("\n=== FINAL DNF (LOGICAL REGIONS) ===\n")
        for name in sorted(finals.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
            parser_text = formula_to_parser_text(finals[name])
            try:
                result = transform_formula(parser_text)
                dnf_text = result.final_unicode

                if clean_func:
                    dnf_text = clean_func(dnf_text)

                f.write(f"{name} := {dnf_text}\n")
            except Exception as e:
                f.write(f"{name} := [DNF transform failed]: {e}\n")


def print_pca_components_importance(pca, original_feature_names):
    print("\nPCA component composition")
    for i, component in enumerate(pca.components_):
        top_indices = np.argsort(np.abs(component))[::-1][:3]
        parts = []
        for idx in top_indices:
            weight = component[idx]
            name = original_feature_names[idx]
            parts.append(f"({weight:.3f} * {name})")
        print(f"x{i + 1} ≈ {' + '.join(parts)}")
    print("-" * 60)


def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class ThresholdAtom:
    affine: Affine
    threshold: Fraction

    def key(self):
        return (self.affine.key(), self.threshold)

    def as_constraint(self) -> Inequality:
        # affine >= threshold  <=>  threshold - affine <= 0
        return Inequality.ge(self.affine, Affine({}, self.threshold))

    def margin(self) -> Affine:
        # atomul e adevarat cand margin >= 0
        return self.affine - Affine({}, self.threshold)

    def pretty(self) -> str:
        return f"[{self.affine.pretty()} >= {frac_str(self.threshold)}]"


@dataclass
class BoolClause:
    atoms: List[ThresholdAtom]

    def pretty(self) -> str:
        if not self.atoms:
            return "True"
        return " ∧ ".join(a.pretty() for a in self.atoms)


@dataclass
class BoolDNF:
    clauses: List[BoolClause]

    def pretty(self) -> str:
        if not self.clauses:
            return "False"

        parts = []
        for c in self.clauses:
            txt = c.pretty()
            if " ∧ " in txt:
                txt = f"({txt})"
            parts.append(txt)

        return " ∨ ".join(parts)


def threshold_expand_dnf(dnf: DNF, threshold: Fraction) -> BoolDNF:
    clauses = []

    for clause in dnf.clauses:
        atoms = []

        for factor in clause.factors:
            atoms.append(
                ThresholdAtom(
                    affine=factor.affine,
                    threshold=threshold
                )
            )

        clauses.append(BoolClause(atoms))

    return BoolDNF(clauses)


def clause_region(clause: BoolClause) -> Region:
    return Region([a.as_constraint() for a in clause.atoms]).dedup()


def atom_implied_by_clause(atom: ThresholdAtom, clause: BoolClause, variables: List[str]) -> bool:
    region = clause_region(clause)

    if region_is_empty(region, variables):
        return True

    mn, _ = affine_minmax_on_region(atom.margin(), region, variables)

    return mn >= 0


def clause_implies_clause(source: BoolClause, target: BoolClause, variables: List[str]) -> bool:
    for atom in target.atoms:
        if not atom_implied_by_clause(atom, source, variables):
            return False

    return True


def simplify_bool_clause(clause: BoolClause, variables: List[str]) -> Optional[BoolClause]:
    unique = {}
    for atom in clause.atoms:
        unique[atom.key()] = atom

    atoms = list(unique.values())

    if region_is_empty(clause_region(BoolClause(atoms)), variables):
        return None

    kept = []

    for i, atom in enumerate(atoms):
        other_atoms = [a for j, a in enumerate(atoms) if j != i]
        other_clause = BoolClause(other_atoms)

        if atom_implied_by_clause(atom, other_clause, variables):
            continue

        kept.append(atom)

    simplified = BoolClause(kept)

    if region_is_empty(clause_region(simplified), variables):
        return None

    return simplified


def simplify_bool_dnf(bool_dnf: BoolDNF, variables: List[str]) -> BoolDNF:
    clauses = []

    for clause in bool_dnf.clauses:
        sc = simplify_bool_clause(clause, variables)
        if sc is not None:
            clauses.append(sc)

    unique = {}
    for c in clauses:
        key = tuple(sorted(a.key() for a in c.atoms))
        unique[key] = c

    clauses = list(unique.values())

    kept = []

    for i, ci in enumerate(clauses):
        redundant = False

        for j, cj in enumerate(clauses):
            if i == j:
                continue

            if clause_implies_clause(ci, cj, variables):
                redundant = True
                break

        if not redundant:
            kept.append(ci)

    return BoolDNF(kept)


def post_explainability_from_final_dnf(result: TransformResult, N: int = 6) -> Dict[int, BoolDNF]:
    out = {}
    for k in range(1, N):
        threshold = Fraction(k, N - 1) * U
        raw = threshold_expand_dnf(result.dnf, threshold)
        simplified = simplify_bool_dnf(raw, result.variables)
        out[k] = simplified

    return out


def print_post_explainability_for_symbolic_defs(symbolic_defs, N: int = 6):
    finals = get_fully_expanded_formulas(symbolic_defs)

    print("\n LM/post explainability from dnf formula")

    for name in sorted(finals.keys()):
        parser_text = formula_to_parser_text(finals[name])

        try:
            result = transform_formula(parser_text)
            Fs = post_explainability_from_final_dnf(result, N=N)

            print(f"\n{name}:")
            print(f"DNF := {result.final_unicode}")

            for k, formula in Fs.items():
                threshold = Fraction(k, N - 1) * U
                print(f"F_{k} [{name} >= {frac_str(threshold)}] := {formula.pretty()}")

        except Exception as e:
            print(f"{name}: LM/Post explainability failed: {e}")


if __name__ == "__main__":
    set_reproducibility(42)

    outdir = "kes_breast_cancer_dnf_final_report"
    os.makedirs(outdir, exist_ok=True)

    n_components = 3
    selector_threshold = 0.5

    data = load_breast_cancer_dataset_pca_same_then_to_interval(
        n_components=n_components,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        u_value=U_VALUE,
    )

    raw_breast_data = load_breast_cancer()
    print_pca_components_importance(data['pca'], raw_breast_data.feature_names)

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
        temperature=1.0
    )

    print("\nStarting Training KES Model...")
    train_model(
        model, X_train_t, y_train_t, X_val=X_val_t, y_val=y_val_t,
        epochs=1000, lr=0.01, batch_size=32, weight_decay=1e-4, print_every=200
    )

    _, test_acc, _, test_probs, test_pred = evaluate_model(model, X_test_t, y_test_t)
    test_auc = roc_auc_score(data["y_test"], test_probs[:, 1].cpu().numpy())

    print("\n" + "=" * 55)
    print(f"{'MODEL COMPARISON BENCHMARK':^55}")
    print("=" * 55)

    models_to_compare = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "RuleFit (Proxy-GBM)": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)
    }

    benchmark_results = []
    benchmark_results.append({
        "Model": "Our model",
        "Accuracy": test_acc,
        "ROC-AUC": test_auc
    })

    X_train_np, y_train_np = data["X_train"], data["y_train"]
    X_test_np, y_test_np = data["X_test"], data["y_test"]

    for name, m in models_to_compare.items():
        m.fit(X_train_np, y_train_np)
        y_p = m.predict(X_test_np)
        y_prob = m.predict_proba(X_test_np)[:, 1]

        benchmark_results.append({
            "Model": name,
            "Accuracy": (y_p == y_test_np).mean(),
            "ROC-AUC": roc_auc_score(y_test_np, y_prob)
        })

    template = "{:<25} | {:<12} | {:<12}"
    print(template.format("Model", "Accuracy", "ROC-AUC"))
    print("-" * 55)
    for res in benchmark_results:
        print(template.format(res['Model'], f"{res['Accuracy']:.4f}", f"{res['ROC-AUC']:.4f}"))
    print("=" * 55 + "\n")

    print("Extracting Symbolic Logic...")
    symbolic_defs = model.extract_symbolic_system(
        feature_names=data["feature_names"],
        selector_threshold=selector_threshold,
        verbose=True
    )

    print("\n--- EXPLAINABILITY RESULTS ---")
    print_final_interpretable_formulas(symbolic_defs)
    print_final_interpretable_inequalities(model, symbolic_defs, feature_names=data["feature_names"])

    print("\n--- DNF FORMULAS (Logical Regions) ---")


    def clean_text(txt: str) -> str:
        t = txt.replace("smul(1,", "").replace("smul(1.0,", "")

        t = re.sub(r'\(?[^∨()]*∧\s*0[^∨()]*\)?', '', t)
        t = re.sub(r'∨\s*∨', '∨', t)
        t = t.strip(' ∨')

        if not t.strip(): t = "0"

        def round_match(m):
            return str(round(float(m.group(0)), 3))

        t = re.sub(r'\d+\.\d{4,}', round_match, t)

        return t.strip()


    finals = get_fully_expanded_formulas(symbolic_defs)
    for name in sorted(finals.keys()):
        parser_text = formula_to_parser_text(finals[name])
        try:
            res = transform_formula(parser_text)
            print(f"{name} := {clean_text(res.final_unicode)}")
        except:
            pass

    print_post_explainability_for_symbolic_defs(
        symbolic_defs,
        N=6
    )

    export_symbolic_report_with_dnf(
        model,
        symbolic_defs,
        filepath=f"{outdir}/symbolic_report.txt",
        feature_names=data["feature_names"],
        decimals=3,
        verify_step=Fraction(1, 2),
        benchmark_results=benchmark_results,
        clean_func=clean_text
    )
    print(f"\nReport successfully saved in: {outdir}/breast_cancer_dnf_report.txt")

