from fractions import Fraction
from utils import (
    parse_sergio_rows,
    build_edges,
    print_no_reduction_fraction,
    print_edges,
    run_n_steps,
    print_trajectory,
    L,
    print_lm_expansion_along_trajectory,
    extract_single_conclusion_rules,
    filter_nontrivial_rules,
    rule_to_str,
    interpret_rule
)

def read_sergio_file(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            row = []
            for val in line.split(","):
                row.append(float(val))
            rows.append(row)
    return rows


def max_gene_from_sergio_rows(rows):
    max_gene = -1
    for row in rows:
        target = int(row[0])
        no_regulators = int(row[1])
        regulators = row[2:2 + no_regulators]
        max_gene = max(max_gene, target)
        for src in regulators:
            max_gene = max(max_gene, int(src))
    return max_gene


def genes_from_sergio_rows(rows):
    max_gene = max_gene_from_sergio_rows(rows)
    genes = []
    for ind in range(max_gene + 1):
        genes.append(f"g{ind}")
    return genes



N = 7
STEPS = 6
INPUT_FILE = "../dataset/Interaction_cID_8.txt"

SERGIO_ROWS = read_sergio_file(INPUT_FILE)

MAX_GENE = max_gene_from_sergio_rows(SERGIO_ROWS)
GENES = genes_from_sergio_rows(SERGIO_ROWS)

print("Max gene:", MAX_GENE)
print("Number of genes:", len(GENES))

RAW_EDGES = parse_sergio_rows(SERGIO_ROWS)
EDGES = build_edges(RAW_EDGES, N)

INITIAL_STATE = {g: Fraction(1) for g in GENES}

print(f"\nL_{N}")
print([print_no_reduction_fraction(x, N) for x in L(N)])

print_edges(N, EDGES)

states, transitions = run_n_steps(INITIAL_STATE, STEPS, GENES, EDGES)

print_trajectory(N, GENES, states)
print_lm_expansion_along_trajectory(N, GENES, states, STEPS)

rules = extract_single_conclusion_rules(N, GENES, transitions, max_premise_size=2, min_support=1)
rules = filter_nontrivial_rules(rules)
rules = sorted(rules,key=lambda r: (-r[2], len(r[0]), rule_to_str(r[0], r[1])))

print("\nInterpreted extracted rules")

for premise, conclusions, support in rules[:10]:
    print(rule_to_str(premise, conclusions), f"[support={support}]")
    print(interpret_rule(N, EDGES, premise, conclusions))
    print()