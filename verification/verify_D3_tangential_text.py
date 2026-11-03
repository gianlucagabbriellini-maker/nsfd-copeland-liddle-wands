"""
verify_D3_tangential_text.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Checks that N1, N2, N3, D1 and D2 as printed in the paper are
algebraically identical to the Tr and c2 that derive_D3_block.py
computed independently. Needs D3block_TrC2.pkl in the current
directory; run derive_D3_block.py first if it isn't there.
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

s = lam**2 - 3

D1 = lam**2*(1+3*phi)*(4*lam**2+6*lam**2*phi+9*phi)*(4*lam**2+12*lam**2*phi+9*phi)
D2 = lam**2*(1+3*phi)*(4*lam**2+6*lam**2*phi+9*phi)**2*(4*lam**2+12*lam**2*phi+9*phi)**2

N1 = (24*phi*(3*phi+1)*s**3 + 54*phi*(15*phi**2+15*phi+4)*s**2
      + 162*phi*(21*phi**2+18*phi+4)*s + 162*phi*(3*phi+1)*(9*phi+4))

N2 = (576*phi**2*(3*phi+1)**2*(3*phi+2)*s**5
      + 432*phi**2*(3*phi+1)*(153*phi**2+144*phi+32)*s**4
      + 1944*phi**2*(2*phi+1)*(9*phi+4)*(27*phi+8)*s**3
      + 1944*phi**2*(9*phi+4)*(117*phi**2+90*phi+16)*s**2
      + 1458*phi**2*(9*phi+4)**2*(15*phi+4)*s)

N3 = (64*(3*phi+1)**2*(3*phi+2)**2*(9*phi+4)*s**5
      + 48*(3*phi+1)*(3*phi+2)**2*(693*phi**2+468*phi+80)*s**4
      + 72*(97443*phi**5+216270*phi**4+186138*phi**3+77904*phi**2+15936*phi+1280)*s**3
      + 108*(255879*phi**5+533223*phi**4+432756*phi**3+171648*phi**2+33408*phi+2560)*s**2
      + 162*(9*phi+4)*(37341*phi**4+56106*phi**3+30672*phi**2+7296*phi+640)*s
      + 3888*(3*phi+1)**2*(9*phi+4)**2*(15*phi+4))

claim1 = N1/D1
claim2 = N2/D2
claim3 = N3/D2

diff1 = sp.simplify(claim1 - (1-c2))
diff2 = sp.simplify(claim2 - (1+c2-Tr))
diff3 = sp.simplify(claim3 - (1+c2+Tr))
print("diff (1-c2) - N1/D1 :", diff1)
print("diff (1+c2-Tr) - N2/D2 :", diff2)
print("diff (1+c2+Tr) - N3/D2 :", diff3)

assert diff1 == 0, "Paper's N1/D1 does not match the independently-derived (1-c2)"
assert diff2 == 0, "Paper's N2/D2 does not match the independently-derived (1+c2-Tr)"
assert diff3 == 0, "Paper's N3/D2 does not match the independently-derived (1+c2+Tr)"
print("\nAll three differences are exactly zero: the paper's N1, N2, N3 "
      "for D3 are algebraically identical to this independent derivation.")
