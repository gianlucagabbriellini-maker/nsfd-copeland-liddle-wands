"""
derive_D3_block.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Builds the Jacobian of the (x,y,v) block at D3 in the three-component
model from scratch, and extracts the two eigenvalues that determine
stability there (the third is the exact zero radial eigenvalue).

Does not reuse the two-component Theorem D formulas: Kn, Delta_n and Pn
differ between the two models, so those formulas do not apply here. The
method mirrors Theorem D's: write the Jury quantities 1-det J,
1+det J-Tr J, 1+det J+Tr J as rational functions of (phi, lambda) over a
common denominator, with numerators whose positivity is checkable by
inspection over lambda^2>3, phi in (0,1/3).
"""
import sympy as sp

phi, lam = sp.symbols('phi lambda', positive=True)
x, y, v = sp.symbols('x y v', real=True)

# 3-comp pre-projection formulas restricted to u=0 (S then = x^2-y^2+1)
Dn = 1 + phi*(sp.sqrt(6)/2*lam*x + sp.Rational(3,2)*y**2)
Kn = 1 + phi*(sp.Rational(3,2)*x**2 + sp.Rational(3,2))   # u=0
Pn = 1 + phi*(3 + sp.Rational(3,2)*y**2)
ytil = Kn*y/Dn
xtil = Kn*(x*Dn + phi*sp.sqrt(6)/2*lam*y**2)/(Pn*Dn)
S = x**2 - y**2 + 1
gp = sp.sqrt((1+3*phi*S)/(1+3*phi))   # matter update (gamma_m=1)
vtil = v*gp

F = sp.Matrix([xtil, ytil, vtil])
Jtil = F.jacobian([x,y,v])

xD3 = sp.sqrt(6)/(2*lam)
vD3 = sp.sqrt(1-3/lam**2)
point = (xD3, xD3, vD3)
zstar = sp.Matrix(point)
Jt = Jtil.subs(dict(zip((x,y,v), point)))
P = sp.eye(3) - zstar*zstar.T
J = sp.simplify(P*Jt)

# As at C4, D3 sits exactly on the unit sphere, so the projection turns J
# into (I - z*z^T) times the raw Jacobian. z* is a left null vector of
# the result (z*^T J = 0, not J z* = 0); confirm that here too.
print("Confirm z*^T J = 0 (left null vector) in this block too:")
print(sp.simplify(zstar.T*J))

Tr = sp.together(sp.simplify(J.trace()))
Det = sp.together(sp.simplify(J.det()))
print("\nTr J =", Tr)
print("Det J =", Det, "  (expect 0: trivial radial eigenvalue lives in this 3-space)")

# c2 is the second elementary symmetric polynomial of J's eigenvalues:
# the sum of the three 2x2 principal minors, without solving for the
# eigenvalues directly.
minors = []
idxs = [(0,1),(0,2),(1,2)]
for i,j in idxs:
    sub = J[[i,j],[i,j]]
    minors.append(sub.det())
c2 = sp.simplify(sum(minors))
print("\nc2 (sum of principal 2x2 minors) =", sp.together(c2))

# Sanity check: with one eigenvalue known to be 0, the other two (mu1,
# mu2) must satisfy mu1+mu2=Tr and mu1*mu2=c2. Solve for them and verify
# directly that they are eigenvalues of J, at two sample (phi, lambda).
mu = sp.symbols('mu')
mu1, mu2 = sp.solve(mu**2 - Tr*mu + c2, mu)
checked_at = [(sp.Rational(4), sp.Rational(1, 5)), (sp.Rational(10), sp.Rational(1, 10))]
for lam_val, phi_val in checked_at:
    subs = {lam: lam_val, phi: phi_val}
    J_num = J.subs(subs)          # substitute first, det() last: much cheaper
    mu1_num, mu2_num = mu1.subs(subs), mu2.subs(subs)
    charpoly_at_mu1 = sp.simplify((J_num - mu1_num*sp.eye(3)).det())
    charpoly_at_mu2 = sp.simplify((J_num - mu2_num*sp.eye(3)).det())
    assert charpoly_at_mu1 == 0 and charpoly_at_mu2 == 0, \
        f"mu1, mu2 solved from Tr, c2 are not actually eigenvalues of J at lambda={lam_val}, phi={phi_val}"
print(f"\nSanity check passed: the mu1, mu2 solved from Tr and c2 really are "
      f"eigenvalues of J (checked at (lambda,phi) = {[(float(l), float(p)) for l, p in checked_at]}), "
      f"confirming c2 is genuinely mu1*mu2 and not a bookkeeping slip.")

import pickle
with open('D3block_TrC2.pkl','wb') as f:
    pickle.dump((Tr, c2), f)
print("\nSaved Tr, c2 to D3block_TrC2.pkl")
