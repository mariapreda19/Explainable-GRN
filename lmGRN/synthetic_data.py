from fractions import Fraction
from utils import parse_sergio_rows, build_edges, print_no_reduction_fraction, print_edges
from utils import print_update_equations, run_n_steps, print_trajectory, L, print_lm_expansion_along_trajectory
from utils import check_second_transition_rule, extract_single_conclusion_rules, filter_nontrivial_rules
from utils import rule_to_str, interpret_rule

N = 7
STEPS = 6

GENES = ["g0", "g1", "g2", "g3"]

SERGIO_ROWS = [
    [1, 1, 0, 2.4, 2],
    [2, 2, 0, 1, 1.8, -1.2, 2, 2],
    [3, 2, 1, 2, 1.5, -2.0, 2, 2],
]

INITIAL_STATE = {
    "g0": Fraction(1),
    "g1": Fraction(1, 6),
    "g2": Fraction(0),
    "g3": Fraction(0),
}

RAW_EDGES = parse_sergio_rows(SERGIO_ROWS)
EDGES = build_edges(RAW_EDGES, N)

print(f"\nL_{N}")
print([print_no_reduction_fraction(x, N) for x in L(N)])

print_edges(N, EDGES)
print_update_equations()

states, transitions = run_n_steps(INITIAL_STATE, STEPS, GENES, EDGES)

print_trajectory(N, GENES, states)
print_lm_expansion_along_trajectory(N, GENES, states, STEPS)

check_second_transition_rule(N, GENES, states)

rules = extract_single_conclusion_rules(N, GENES, transitions, max_premise_size=2, min_support=1)
rules = filter_nontrivial_rules(rules)
rules = sorted(rules, key=lambda r: (-r[2], len(r[0]), rule_to_str(r[0], r[1])))

print("\nInterpreted extracted rules")
for premise, conclusions, support in rules[:10]:
    print(rule_to_str(premise, conclusions), f"[support={support}]")
    print(interpret_rule(N, EDGES, premise, conclusions))
    print()







