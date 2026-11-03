"""
derive_D3_jury.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Prints the three Jury-condition expressions for the D3 tangential block,
from the Tr and c2 that derive_D3_block.py computed. Run that script
first; this one only reads its saved output.
"""
import sympy as sp
import pickle

phi, lam = sp.symbols('phi lambda', positive=True)
try:
    with open('D3block_TrC2.pkl', 'rb') as f:
        Tr, c2 = pickle.load(f)
except FileNotFoundError:
    raise SystemExit(
        "D3block_TrC2.pkl not found in the current directory.\n"
        "Run 'python3 derive_D3_block.py' first (from this same directory) "
        "to generate it, then re-run this script."
    )

J1 = sp.together(1 - c2)
J2 = sp.together(1 + c2 - Tr)
J3 = sp.together(1 + c2 + Tr)

for name, expr in [('1-c2 (Jury cond 1)', J1), ('1+c2-Tr (Jury cond 2)', J2), ('1+c2+Tr (Jury cond 3)', J3)]:
    num, den = sp.fraction(expr)
    num = sp.expand(num)
    den = sp.expand(den)
    print(f"\n=== {name} ===")
    print("Denominator:", sp.factor(den))
    print("Numerator (expanded):", num)
