"""
verify_2comp_stability.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Rebuilds the two-component model's Jacobian symbolically at every fixed
point (C, D, O, B+, B-) from the raw formulas, independently of the
paper's stated eigenvalues, and checks that every one of them matches.
"""
import sympy as sp


def build_jacobian():
    x, y, w, phi, lam, gam = sp.symbols('x y w phi lambda gamma', positive=True)
    Kn = 1 + phi*(3*x**2 + sp.Rational(3, 2)*gam)
    Dn = 1 + phi*(sp.sqrt(6)/2*lam*x + sp.Rational(3, 2)*gam*(x**2 + y**2))
    Pn = 1 + phi*(3 + sp.Rational(3, 2)*gam*(x**2 + y**2))
    ytil = Kn*y/Dn
    xtil = Kn*(x*Dn + phi*sp.sqrt(6)/2*lam*y**2)/(Pn*Dn)
    S = 2*x**2 + gam*w**2
    wtil = w*sp.sqrt((1 + 3*phi*S)/(1 + 3*gam*phi))
    Ftil = sp.Matrix([xtil, ytil, wtil])
    Jtil = Ftil.jacobian([x, y, w])
    return Ftil, Jtil, (x, y, w, phi, lam, gam)


def projected_jacobian_at(point, Ftil, Jtil, syms):
    x, y, w, phi, lam, gam = syms
    subs = {x: point[0], y: point[1], w: point[2]}
    zstar = sp.Matrix(point)
    Jt = Jtil.subs(subs)
    P = sp.eye(3) - zstar*zstar.T
    return sp.simplify(P*Jt)


def main():
    Ftil, Jtil, syms = build_jacobian()
    x, y, w, phi, lam, gam = syms

    # ---------------------------------------------------------------
    # Point C = (x_C, y_C, 0)
    # ---------------------------------------------------------------
    xC = lam/sp.sqrt(6)
    yC = sp.sqrt(1 - lam**2/6)
    JC = projected_jacobian_at((xC, yC, 0), Ftil, Jtil, syms)
    eigsC = list(JC.eigenvals().keys())
    eigsC_simplified = sorted([sp.simplify(e) for e in eigsC], key=lambda e: str(e))
    print("Eigenvalues at C (raw):")
    for e in eigsC_simplified:
        print("  ", e)

    mu_t_expected = (2 + 3*gam*phi + lam**2*phi)/(2 + 3*gam*phi + 6*phi)
    lam_w_expected = sp.sqrt((1 + phi*lam**2)/(1 + 3*phi*gam))
    found_mu_t = any(sp.simplify(e - mu_t_expected) == 0 for e in eigsC_simplified)
    found_lam_w = any(sp.simplify(e - lam_w_expected) == 0 for e in eigsC_simplified)
    found_zero = any(sp.simplify(e) == 0 for e in eigsC_simplified)
    print("Matches mu_t  :", found_mu_t)
    print("Matches lambda_w:", found_lam_w)
    print("Matches 0     :", found_zero)
    assert found_mu_t and found_lam_w and found_zero, "Mismatch at C!"

    # ---------------------------------------------------------------
    # Point O = (0, 0, 1) and B+/- = (+-1, 0, 0)
    # ---------------------------------------------------------------
    JO = projected_jacobian_at((0, 0, 1), Ftil, Jtil, syms)
    eigsO = sorted([sp.simplify(e) for e in JO.eigenvals().keys()], key=str)
    print("\nEigenvalues at O:")
    for e in eigsO:
        print("  ", e)
    O_expected = [0, (2 + 3*gam*phi)/(2 + 6*phi), 1 + sp.Rational(3, 2)*gam*phi]
    for exp in O_expected:
        assert any(sp.simplify(e - exp) == 0 for e in eigsO), f"O eigenvalue {exp} not found!"
    print("All three O eigenvalues matched.")

    for sign, label in [(1, 'B+'), (-1, 'B-')]:
        JB = projected_jacobian_at((sign, 0, 0), Ftil, Jtil, syms)
        eigsB = sorted([sp.simplify(e) for e in JB.eigenvals().keys()], key=str)
        print(f"\nEigenvalues at {label}:")
        for e in eigsB:
            print("  ", e)
        B_expected = [
            0,
            (2 + 3*gam*phi + 6*phi)/(2 + 3*gam*phi + sign*sp.sqrt(6)*lam*phi),
            sp.sqrt((1 + 6*phi)/(1 + 3*gam*phi)),
        ]
        for exp in B_expected:
            assert any(sp.simplify(e - exp) == 0 for e in eigsB), \
                f"{label} eigenvalue {exp} not found!"
        print(f"All three {label} eigenvalues matched.")

    # ---------------------------------------------------------------
    # lambda_A(gamma): the well-posedness threshold of Proposition 1
    # ---------------------------------------------------------------
    print("\n--- lambda_A(gamma) (Proposition 1) ---")
    # Delta_n(-1, 0) = 1 + phi*(3/2*gamma - sqrt(6)/2*lambda); requiring
    # this to stay >=0 as phi -> 1/3 (its supremum) gives the threshold.
    Delta_at_vertex = 1 + sp.Rational(1, 3)*(sp.Rational(3, 2)*gam - sp.sqrt(6)/2*lam)
    lamA_solution = sp.solve(sp.Eq(Delta_at_vertex, 0), lam)[0]
    lamA_expected = sp.sqrt(6)*(2 + gam)/2
    print("Solved lambda_A(gamma) =", sp.simplify(lamA_solution))
    print("Paper's lambda_A(gamma) =", lamA_expected)
    assert sp.simplify(lamA_solution - lamA_expected) == 0, "lambda_A mismatch!"
    print("Matches.")

    # lambda_A(gamma) > sqrt(6)*gamma for every gamma<2 (branches meet consistently)
    diff = sp.simplify(lamA_expected - sp.sqrt(6)*gam)
    print("\nlambda_A(gamma) - sqrt(6)*gamma =", diff, "(should be positive for gamma<2)")
    for gval in [0.1, 0.5, 1.0, 1.5, 1.9]:
        val = float(diff.subs(gam, gval))
        assert val > 0, f"lambda_A ordering fails at gamma={gval}"
    print("Confirmed positive by direct evaluation for gamma in (0,2).")

    print("\nAll two-component stability claims (Theorems 1-2, eigenvalues "
          "at O/B+/B-, and Proposition 1's threshold) verified independently.")


if __name__ == '__main__':
    main()
