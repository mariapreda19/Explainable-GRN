from fractions import Fraction
from itertools import combinations
from math import inf


def L(n):
    L = []
    for k in range(n):
        L.append(Fraction(k, n-1))
    return L

def iota_n(x, n):
    return Fraction(round((n - 1) * x), n - 1)

def neg(x):
    return Fraction(1)-x

def OPlus(x,y):
    return min(Fraction(1), x+y)

def ODot(x,y):
    return max(Fraction(0), x+y-1)

def sigma(k, x, n):
    return int(x>=Fraction(k, n-1))

def post_vector(x,n):
    result = []
    for k in range(1,n):
        result.append(sigma(k, x, n))
    return tuple(result)

def reconstruct_from_post(vec, n):
    return Fraction(sum(vec), n-1)

def print_no_reduction_fraction(x, n):
    if isinstance(x, Fraction):
        k = int(x*(n-1))
        if k == 0:
            return "0"
        if k == n-1:
            return "1"
        return f"{k}/{n-1}"

    return str(x)


def parse_sergio_rows(rows):
    raw_edges = []
    for row in rows:
        target = int(row[0])
        no_regulators = int(row[1])

        regulators = row[2:2 + no_regulators]
        k_values = row[2+no_regulators:2+2*no_regulators]
        for src, k in zip(regulators, k_values):
            raw_edges.append((f"g{int(src)}", f"g{target}", Fraction(str(k))))

    return raw_edges

def build_edges(raw_edges, n):
    k_max = -1
    for a, b, w in raw_edges:
        if abs(w) > k_max:
            k_max = abs(w)

    edges = []
    for src, tgt, raw_w in raw_edges:
        if raw_w >= 0:
            mode = "activation"
        else:
            mode = "inhibition"

        weight = iota_n(abs(raw_w)/k_max, n)
        edges.append({"src": src, "tgt": tgt, "mode": mode, "weight": weight, "raw_weight": raw_w})

    return edges

def activator_input(gene, state, EDGES):
    value = Fraction(0)
    for e in EDGES:
        if e["tgt"] == gene and e["mode"] == "activation":
            value = OPlus(value, ODot(e["weight"], state[e["src"]]))
    return value

def inhibitor_input(gene, state, EDGES):
    value = Fraction(0)
    for e in EDGES:
        if e["tgt"] == gene and e["mode"] == "inhibition":
            value = OPlus(value, ODot(e["weight"], state[e["src"]]))
    return value

def update_gene(gene, state, EDGES):
    has_input = any(e["tgt"] == gene for e in EDGES)
    if not has_input:
        return state[gene]
    A = activator_input(gene, state, EDGES)
    I = inhibitor_input(gene, state, EDGES)
    return ODot(A, neg(I))

def update_state(state, GENES, EDGES):

    next_state = {}

    for gene in GENES:
        next_state[gene] = update_gene(gene, state, EDGES)

    return next_state

def run_n_steps(initial_state, steps, GENES, EDGES):
    states = [initial_state]
    transitions = []

    current = initial_state
    for step in range(steps):
        next = update_state(current, GENES, EDGES)
        if current == next:
            break
        transitions.append((current, next))
        states.append(next)
        current = next

    return states, transitions


def print_state(N, GENES, label, state):
    values = ", ".join(f"{g}={print_no_reduction_fraction(state[g], N)}" for g in GENES)
    print(f"{label}: ({values})")


def print_edges(N, EDGES):
    print(f"\nParsed normalized edges in L_{N}")
    for e in EDGES:
        print(
            f"{e['src']} -> {e['tgt']} | "
            f"mode={e['mode']} | "
            f"raw={e['raw_weight']} | "
            f"L_{N} weight={print_no_reduction_fraction(e['weight'], N)}")


def print_update_equations():
    print("\nUpdate equations")
    print("g0' = g0")
    print("g1' = 1 ⊙ g0")
    print("g2' = (4/6 ⊙ g0) ⊙ ¬(3/6 ⊙ g1)")
    print("g3' = (4/6 ⊙ g1) ⊙ ¬(5/6 ⊙ g2)")


def print_trajectory(N, GENES, states):
    print("\nTrajectory for fixed number of steps")

    for t, state in enumerate(states):
        print_state(N, GENES,f"x^{t}", state)


def print_lm_expansion(N, GENES, state):
    for g in GENES:
        vec = post_vector(state[g], N)
        rec = reconstruct_from_post(vec, N)
        print(f"{g}: {vec} -> {print_no_reduction_fraction(rec, N)}")


def print_lm_expansion_along_trajectory(N, GENES, states, STEPS):
    print("\nLM/Post expansion along trajectory")
    index = 0
    for t, state in enumerate(states):
        print(f"\nx^{t}:")
        print_lm_expansion(N, GENES, state)
        index += 1

    if index != STEPS:
        print("Fixed point")


def literal_truth(N, state, gene, k, positive=True):
    value = bool(sigma(k, state[gene], N))
    return value if positive else not value


def implication_holds(N, transitions, premise, conclusions):
    support = 0

    for st, nxt in transitions:
        context = {
            "t": st,
            "next": nxt,
        }

        premise_true = all(
            literal_truth(N,context[time], gene, k, positive)
            for gene, k, positive, time in premise
        )

        if premise_true:
            support += 1

            conclusions_true = all(
                literal_truth(N,context[time], gene, k, positive)
                for gene, k, positive, time in conclusions
            )

            if not conclusions_true:
                return False, support

    return support > 0, support


def literal_to_str(lit):
    gene, k, positive, time = lit
    prime = "'" if time == "next" else ""

    if positive:
        return f"σ_{k}({gene}{prime})"

    return f"¬σ_{k}({gene}{prime})"


def rule_to_str(premise, conclusions):
    left = " ∧ ".join(literal_to_str(lit) for lit in premise)
    right = " ∧ ".join(literal_to_str(lit) for lit in conclusions)
    return f"{left} -> {right}"

def expression_level(x, n):
    if x == Fraction(0):
        return "not expressed"

    if x == Fraction(1):
        return "saturated"

    level = int(x * (n - 1))
    max_level = n - 1

    low_limit = max_level / 3
    medium_limit = 2 * max_level / 3

    if level <= low_limit:
        return "low"

    if level <= medium_limit:
        return "medium"

    return "high"


def describe_literal(N, lit):
    gene, k, positive, time = lit
    prime = "'" if time == "next" else ""
    threshold = Fraction(k, N - 1)
    threshold_text = print_no_reduction_fraction(threshold, N)
    level = expression_level(threshold, N)

    if positive:
        if level == "saturated":
            return f"{gene}{prime} is saturated"
        return f"{gene}{prime} is at least {level} ({threshold_text})"

    if level == "saturated":
        return f"{gene}{prime} is not saturated"

    return f"{gene}{prime} does not reach {level} ({threshold_text})"


def describe_regulatory_context(premise, conclusion_gene, EDGES):
    activators = []
    inhibitors = []

    for lit in premise:
        gene, k, positive, time = lit

        for e in EDGES:
            if e["src"] == gene and e["tgt"] == conclusion_gene:
                if e["mode"] == "activation":
                    activators.append(gene)
                if e["mode"] == "inhibition":
                    inhibitors.append(gene)

    parts = []

    if len(activators) > 0:
        parts.append("activatory regulators involved: " + ", ".join(activators))

    if len(inhibitors) > 0:
        parts.append("inhibitory regulators involved: " + ", ".join(inhibitors))

    if len(parts) == 0:
        return "no direct regulatory edge from the premise genes to the conclusion gene"

    return "; ".join(parts)


def interpret_rule(N, EDGES, premise, conclusions):
    premise_texts = []

    for lit in premise:
        premise_texts.append(describe_literal(N, lit))

    conclusion_texts = []

    for lit in conclusions:
        conclusion_texts.append(describe_literal(N, lit))

    conclusion_gene = conclusions[0][0]
    context_text = describe_regulatory_context(premise, conclusion_gene, EDGES)

    return (
        "if "
        + " and ".join(premise_texts)
        + ", then "
        + " and ".join(conclusion_texts)
        + ". Regulatory context: "
        + context_text
        + "."
    )

def extract_single_conclusion_rules(N, GENES, transitions, max_premise_size=2, min_support=1):
    premise_literals = []
    for g in GENES:
        for k in range(1, N):
            premise_literals.append((g, k, True, "t"))

    conclusion_literals = []
    for g in GENES:
        for k in range(1, N):
            positive_literal = (g, k, True, "next")
            negative_literal = (g, k, False, "next")
            conclusion_literals.append(positive_literal)
            conclusion_literals.append(negative_literal)

    rules = []

    for size in range(1, max_premise_size + 1):
        for premise in combinations(premise_literals, size):
            for conclusion in conclusion_literals:
                ok, support = implication_holds(N, transitions, list(premise), [conclusion])
                if ok and support >= min_support:
                    rules.append((list(premise), [conclusion], support))

    return rules


def filter_nontrivial_rules(rules):
    filtered = []

    for premise, conclusions, support in rules:
        premise_genes = {lit[0] for lit in premise}
        conclusion_genes = {lit[0] for lit in conclusions}

        if not conclusion_genes.issubset(premise_genes):
            filtered.append((premise, conclusions, support))

    return filtered


def check_second_transition_rule(N, GENES, states):
    print("\nRule extracted from the second transition x^1 -> x^2")
    st = states[1]
    nxt = states[2]
    print_state(N, GENES,"x^1", st)
    print_state(N, GENES, "x^2", nxt)
    premise = [("g1", 6, True, "t"), ("g2", 4, True, "t"),]
    conclusions = [("g3", 1, True, "next"), ("g3", 2, False, "next"),]
    ok, support = implication_holds(N,[(st, nxt)], premise, conclusions)

    print(rule_to_str(premise, conclusions), ok, f"support={support}")
    print("\nInterpretation:")
    print("if g1 is saturated and the inhibitor g2 is high,")
    print("then g3' reaches only level 1/6 and does not reach level 2/6.")

