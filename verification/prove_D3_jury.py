"""
prove_D3_jury.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Expands the paper's stated N1 and N2 (Jury conditions 1 and 2 for the
D3 tangential-stability proposition) as polynomials in s = lambda^2-3,
so each coefficient's sign can be checked by eye. Prints only, does not
assert anything, and does not cover N3; see verify_3comp_D3_jury.py for
an automated check of all three.
"""
import sympy as sp

phi, s = sp.symbols('phi s', positive=True)
lam2 = 3 + s   # lambda^2 = 3+s, s>0 captures existence region lambda^2>3

# Condition 1 numerator
N1 = lambda L2: 72*L2**3*phi**2 + 24*L2**3*phi + 810*L2**2*phi**3 + 162*L2**2*phi**2 - 1458*L2*phi**3 + 1458*phi**3
N1s = sp.expand(N1(lam2))
poly1 = sp.Poly(N1s, s)
print("Condition 1 numerator, as polynomial in s=lambda^2-3 (coeffs should be checked for sign):")
for monom, coeff in poly1.terms():
    print(f"  s^{monom[0]}: {sp.factor(coeff)}")

print()
# Condition 2 numerator
def N2f(L2):
    return (15552*L2**5*phi**5 + 20736*L2**5*phi**4 + 8640*L2**5*phi**3 + 1152*L2**5*phi**2
            - 34992*L2**4*phi**5 - 58320*L2**4*phi**4 - 25920*L2**4*phi**3 - 3456*L2**4*phi**2
            - 34992*L2**3*phi**5 + 5832*L2**3*phi**4 + 7776*L2**3*phi**3
            + 52488*L2**2*phi**5 - 17496*L2**2*phi**4 - 23328*L2**2*phi**3
            - 118098*L2*phi**5 - 104976*L2*phi**4
            - 118098*phi**5)
N2s = sp.expand(N2f(lam2))
poly2 = sp.Poly(N2s, s)
print("Condition 2 numerator, as polynomial in s:")
for monom, coeff in poly2.terms():
    print(f"  s^{monom[0]}: {sp.factor(coeff)}")
