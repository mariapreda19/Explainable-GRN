from fractions import Fraction
from pyeda.inter import exprvar
from pyeda.boolalg.minimization import espresso_exprs

def frac(x):
    x = Fraction(x)
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"

class ThresholdPredicate:
    def __init__(self, alpha, beta):
        self.alpha = Fraction(alpha)
        self.beta = Fraction(beta)

    def threshold(self, y):
        return self.alpha + y * self.beta

    def dominates(self, other):
        d0 = self.threshold(Fraction(0)) - other.threshold(Fraction(0))
        d1 = self.threshold(Fraction(1)) - other.threshold(Fraction(1))
        return d0 >= 0 and d1 >= 0

    def impossible(self):
        vals = [self.threshold(Fraction(0)),self.threshold(Fraction(1))]
        return min(vals) > 1

    def __str__(self):
        if self.beta == 0:
            return f"[x >= {frac(self.alpha)}]"

        sign = "+" if self.beta > 0 else "-"
        return f"[x >= {frac(self.alpha)} {sign} {frac(abs(self.beta))}y]"

def espresso_minimize_warp(predicates):
    varbs = []
    for i in range(len(predicates)):
        varbs.append(exprvar(f"P{i}"))

    expr = varbs[0]
    for v in varbs[1:]:
        expr |= v
    result, = espresso_exprs(expr)
    return result

def semantic_reduce(predicates):
    preds = []
    for p in predicates:
        if not p.impossible():
            preds.append(p)

    kept = []
    for i, p1 in enumerate(preds):
        redundant = False

        for j, p2 in enumerate(preds):
            if i == j:
                continue
            if p1.dominates(p2):
                redundant = True
                break

        if not redundant:
            kept.append(p1)

    return kept

def predicates_to_or_string(predicates):
    if not predicates:
        return "0"

    return " ∨ ".join(str(p) for p in predicates)

def generate_Fk(k, N):
    t = Fraction(k, N-1)
    p1 = ThresholdPredicate(alpha=t/Fraction(4,5), beta = 0)
    p2 = ThresholdPredicate(alpha=(t + Fraction(2,5))/Fraction(3,2), beta = Fraction(2,5))
    return [p1, p2]

N = 6

for k in range(1, N):
    print(f"\nk = {k}\n")

    preds = generate_Fk(k, N)

    print("Original:\n")
    print(" ", predicates_to_or_string(preds))

    expr = espresso_minimize_warp(preds)

    print("Espresso boolean form:\n")
    print(expr)

    reduced = semantic_reduce(preds)
    print("minimization:")
    print(" ", predicates_to_or_string(reduced))

