"""
verify_propositions.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Symbolic cross-check of most closed-form propositions and theorems in
Sections 3-4, building the map directly from the discretized equations
rather than importing nsfd_core, so it is an independent check.
Asserts the key checkpoints (fixed-point preservation, the block-diagonal
derivatives, a few numeric spot-checks on the sign arguments) instead of
only printing them.
"""
import sympy as sp

print("="*70)
print("PART 1 -- Two-component model (Section 3.2-3.4)")
print("="*70)

xn, yn, phi, lam, gam = sp.symbols('x_n y_n phi lambda gamma', real=True, positive=True)
xn1, yn1 = sp.symbols('x_n+1 y_n+1')

# Eq. (8)-(9): exploded, sequentially-substitutable discretization
rhs_x = (-3*xn1 + sp.sqrt(6)/2*lam*yn*yn1 + 3*xn**3 + sp.Rational(3,2)*gam*xn
          - sp.Rational(3,2)*gam*xn**2*xn1 - sp.Rational(3,2)*gam*yn**2*xn1)
eq_x = sp.Eq((xn1 - xn)/phi, rhs_x)
rhs_y = (-sp.sqrt(6)/2*lam*xn*yn1 + 3*xn**2*yn + sp.Rational(3,2)*gam*yn
          - sp.Rational(3,2)*gam*xn**2*yn1 - sp.Rational(3,2)*gam*yn**2*yn1)
eq_y = sp.Eq((yn1 - yn)/phi, rhs_y)
sol = sp.solve([eq_x, eq_y], [xn1, yn1])
x1, y1 = sp.simplify(sol[xn1]), sp.simplify(sol[yn1])

print("\n--- Fixed-point preservation (Rule III) ---")
for name, (xs, ys) in {
    'O': (0, 0),
    'B_+': (1, 0),
    'B_-': (-1, 0),
    'C': (lam/sp.sqrt(6), sp.sqrt(1 - lam**2/6)),
    'D': (sp.sqrt(sp.Rational(3,2))*gam/lam, sp.sqrt(3*gam*(2-gam)/2)/lam),
}.items():
    dx = sp.simplify(x1.subs({xn: xs, yn: ys}) - xs)
    dy = sp.simplify(y1.subs({xn: xs, yn: ys}) - ys)
    status = 'OK' if dx == 0 and dy == 0 else 'FAIL'
    print(f"  {name}: x_next - x = {dx}, y_next - y = {dy}  [{status}]")
    assert status == 'OK', f"{name} is not an exact fixed point of the two-component map!"

print("\n--- Proposition 1: well-posedness / y-positivity threshold ---")
Deltan = 1 + phi*(sp.sqrt(6)/2*lam*xn + sp.Rational(3,2)*gam*(xn**2+yn**2))
# vertex case (lambda < sqrt(6)*gamma): Delta at vertex x*=-lambda/(sqrt6 gamma)
xstar = -lam/(sp.sqrt(6)*gam)
Delta_vertex = sp.simplify(Deltan.subs({xn: xstar, yn: 0}))
coeff_vertex = sp.simplify((Delta_vertex - 1)/phi)
print(f"  Delta at interior vertex (lambda<sqrt6*gamma): coefficient of phi = {coeff_vertex}")
print(f"    = 1.5*gamma - lambda^2/(4*gamma), positive there since lambda^2<6*gamma^2  [analytic, see paper]")
# endpoint case (lambda >= sqrt(6)*gamma): Delta at x=-1
Delta_endpoint = sp.simplify(Deltan.subs({xn: -1, yn: 0}))
print(f"  Delta at endpoint x=-1: {Delta_endpoint}")
lamA_solved = sp.solve(sp.Eq(1 + sp.Rational(1,3)*(sp.Rational(3,2)*gam - sp.sqrt(6)/2*lam), 0), lam)
print(f"  lambda_A(gamma) solving Delta(-1,0,phi->1/3)=0: {lamA_solved}")
lamA_formula = sp.sqrt(6)*(2+gam)/2
lamA_matches = sp.simplify(lamA_solved[0] - lamA_formula) == 0
print(f"  matches sqrt(6)*(2+gamma)/2? {lamA_matches}")
assert lamA_matches, "lambda_A(gamma) does not match the paper's formula!"

print("\n--- Theorem D (elementary stability at D) ---")
J = sp.Matrix([[sp.diff(x1, xn), sp.diff(x1, yn)],
               [sp.diff(y1, xn), sp.diff(y1, yn)]])
xD = sp.sqrt(sp.Rational(3,2))*gam/lam
yD = sp.sqrt(3*gam*(2-gam)/2)/lam
JD = sp.simplify(J.subs({xn: xD, yn: yD}))
TrD, DetD = sp.trace(JD), JD.det()
cond2D = sp.factor(sp.simplify(1 + DetD - TrD))
cond1D = sp.factor(sp.simplify(1 - DetD))
print(f"  Condition 2 (stable iff lambda^2>3*gamma): {cond2D}")
print(f"  Condition 1 (should hold unconditionally): {cond1D}")
# Spot-check with real numbers where D actually exists and should be
# stable (lambda^2=9 > 3*gamma=3): both conditions should come out
# positive, not just look plausible in symbolic form.
spot_D = {lam: sp.Rational(3), gam: sp.Rational(1), phi: sp.Rational(1, 5)}
c1D_val, c2D_val = cond1D.subs(spot_D), cond2D.subs(spot_D)
print(f"  Spot-check at lambda=3, gamma=1, phi=0.2: Condition1={c1D_val}, Condition2={c2D_val}")
assert c1D_val > 0 and c2D_val > 0, "Theorem D's Jury conditions are not both positive where D is stable!"

print("\n--- Theorem C (elementary stability at C) ---")
xC, yC = lam/sp.sqrt(6), sp.sqrt(1 - lam**2/6)
JC = sp.simplify(J.subs({xn: xC, yn: yC}))
TrC, DetC = sp.trace(JC), JC.det()
cond2C = sp.factor(sp.simplify(1 + DetC - TrC))
cond1C = sp.simplify(1 - DetC)
num1C, den1C = sp.fraction(sp.together(cond1C))
print(f"  Condition 2 (stable iff lambda^2<3*gamma): {cond2C}")
# Same kind of spot-check as for Theorem D, but where C exists and
# should be stable instead (lambda^2=1 < 3*gamma=3).
spot_C = {lam: sp.Rational(1), gam: sp.Rational(1), phi: sp.Rational(1, 5)}
c1C_val, c2C_val = cond1C.subs(spot_C), cond2C.subs(spot_C)
print(f"  Spot-check at lambda=1, gamma=1, phi=0.2: Condition1={c1C_val}, Condition2={c2C_val}")
assert c1C_val > 0 and c2C_val > 0, "Theorem C's Jury conditions are not both positive where C is stable!"
poly_phi = sp.Poly(sp.expand(num1C), phi)
coeffs = poly_phi.all_coeffs()
print(f"  Condition 1 numerator, as polynomial in phi, coefficients: {coeffs}")

print("\n  --- Closed-form completion of Condition 1 (B(phi) > 0) ---")
B0 = coeffs[-1] if len(coeffs) > 1 else 0
Bphi_coeff = coeffs[-2] if len(coeffs) > 1 else coeffs[-1]
u = sp.symbols('u', positive=True)  # u = lambda^2
Bphi_coeff_u = Bphi_coeff  # already contains lambda symbolically; view as poly in lambda^2
print(f"  B(0)/phi-coefficient extracted from numerator = phi*B(phi):")
Bfull = sp.simplify(sp.expand(num1C) / phi)
Bpoly = sp.Poly(Bfull, phi)
B_coeffs = Bpoly.all_coeffs()  # [coeff of phi^1, coeff of phi^0] since Bfull is linear in phi
print(f"  B(phi) = {sp.expand(Bfull)}")
B_at_0 = Bfull.subs(phi, 0)
B_phi_coeff = sp.simplify(sp.diff(Bfull, phi))
print(f"  B(0)   = {sp.factor(B_at_0)}")
print(f"  dB/dphi = {sp.expand(B_phi_coeff)}  (constant, since B linear in phi)")
# as function of u=lambda^2: check concavity and endpoints
B_phi_coeff_u = B_phi_coeff.subs(lam**2, u) if lam**2 in B_phi_coeff.args else B_phi_coeff
c_u = sp.expand(B_phi_coeff.subs(lam, sp.sqrt(u)))
print(f"  coefficient of phi as function of u=lambda^2: {c_u}")
c_u_at_0 = sp.simplify(c_u.subs(u, 0))
c_u_at_3gam = sp.factor(sp.simplify(c_u.subs(u, 3*gam)))
c_u_concavity = sp.diff(c_u, u, 2)
print(f"    at u=0:      {c_u_at_0}")
print(f"    at u=3*gamma: {c_u_at_3gam}")
print(f"    2nd derivative in u (concavity check): {c_u_concavity}")
# The paper's argument needs this coefficient to be concave in u (so its
# minimum over [0,3*gamma] sits at one of the two endpoints) and positive
# at both of those endpoints; check both claims numerically at a
# representative gamma rather than leaving them as something to notice
# by eye in the printout above.
gam_spot = sp.Rational(1)
assert c_u_concavity.subs(gam, gam_spot) <= 0, "Coefficient is not concave in u -- paper's endpoint argument would not apply!"
assert c_u_at_0.subs(gam, gam_spot) > 0, "Coefficient is not positive at u=0!"
assert c_u_at_3gam.subs(gam, gam_spot) > 0, "Coefficient is not positive at u=3*gamma!"

print("\n" + "="*70)
print("PART 2 -- Three-component model (Section 4.2)")
print("="*70)

un = sp.symbols('u_n', real=True, positive=True)
rhs_x3 = (-3*xn1 + sp.sqrt(6)/2*lam*yn*yn1 + sp.Rational(3,2)*xn**3
          + sp.Rational(3,2)*xn + sp.Rational(1,2)*xn*un**2 - sp.Rational(3,2)*yn**2*xn1)
eq_x3 = sp.Eq((xn1 - xn)/phi, rhs_x3)
rhs_y3 = (-sp.sqrt(6)/2*lam*xn*yn1 + sp.Rational(3,2)*xn**2*yn + sp.Rational(3,2)*yn
          + sp.Rational(1,2)*yn*un**2 - sp.Rational(3,2)*yn**2*yn1)
eq_y3 = sp.Eq((yn1 - yn)/phi, rhs_y3)
sol3 = sp.solve([eq_x3, eq_y3], [xn1, yn1])
x1_3, y1_3 = sp.simplify(sol3[xn1]), sp.simplify(sol3[yn1])

print("\n--- Fixed point C4 and block-diagonal structure ---")
dxC = sp.simplify(x1_3.subs({xn: xC, yn: yC, un: 0}) - xC)
dyC = sp.simplify(y1_3.subs({xn: xC, yn: yC, un: 0}) - yC)
print(f"  C4: x_next-x = {dxC}, y_next-y = {dyC}")
dx1du = sp.simplify(sp.diff(x1_3, un).subs(un, 0))
dy1du = sp.simplify(sp.diff(y1_3, un).subs(un, 0))
print(f"  d(x_next)/du at u=0 = {dx1du}  (should be 0, for block-diagonal Jacobian)")
print(f"  d(y_next)/du at u=0 = {dy1du}  (should be 0)")
assert dx1du == 0 and dy1du == 0, "(x,y) block is not decoupled from u at C4 -- Jacobian would not be block-diagonal!"

print("\n--- Omega_r, Omega_m unconditional positivity ---")
S_sym = sp.symbols('S', positive=True)
Omega_n = sp.symbols('Omega_n', positive=True)
Omega_next = Omega_n*(1 + 3*phi*S_sym)/(1 + 4*phi)
print(f"  Omega_{{n+1}} = {Omega_next}")
print(f"  numerator manifestly positive for Omega_n,phi,S>=0: {sp.simplify(Omega_n*(1+3*phi*S_sym))}")
print(f"  denominator manifestly positive: 1+4*phi > 0 always")

print("\nAll checks complete. Compare against the paper's equations (10)-(25).")
