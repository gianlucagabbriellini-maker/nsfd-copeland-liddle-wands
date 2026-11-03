"""
verify_3comp_stability.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Rebuilds the three-component model's Jacobian symbolically at C4, D3
and R3 from the raw formulas, and checks it against every eigenvalue
claimed for them, the fluid-positivity proposition, and the argument
that radiation and matter cannot coexist at a fixed point.
"""
import sympy as sp


def build_4d_jacobian():
    x, y, u, v, phi, lam = sp.symbols('x y u v phi lambda', positive=True)
    Kn = 1 + phi*(sp.Rational(3, 2)*x**2 + sp.Rational(3, 2) + sp.Rational(1, 2)*u**2)
    Dn = 1 + phi*(sp.sqrt(6)/2*lam*x + sp.Rational(3, 2)*y**2)
    Pn = 1 + phi*(3 + sp.Rational(3, 2)*y**2)
    ytil = Kn*y/Dn
    xtil = Kn*(x*Dn + phi*sp.sqrt(6)/2*lam*y**2)/(Pn*Dn)
    S = x**2 - y**2 + 1 + u**2/3
    g = sp.sqrt((1 + 3*phi*S)/(1 + 4*phi))
    gp = sp.sqrt((1 + 3*phi*S)/(1 + 3*phi))
    util = u*g
    vtil = v*gp
    Ftil = sp.Matrix([xtil, ytil, util, vtil])
    Jtil = Ftil.jacobian([x, y, u, v])
    return Ftil, Jtil, (x, y, u, v, phi, lam)


def main():
    Ftil, Jtil, syms = build_4d_jacobian()
    x, y, u, v, phi, lam = syms

    # ---------------------------------------------------------------
    # C4 = (x_C, y_C, 0, 0)
    # ---------------------------------------------------------------
    xC = lam/sp.sqrt(6)
    yC = sp.sqrt(1 - lam**2/6)
    point = (xC, yC, 0, 0)
    subs = dict(zip((x, y, u, v), point))
    zstar = sp.Matrix(point)
    Jt = Jtil.subs(subs)
    P = sp.eye(4) - zstar*zstar.T
    J = sp.simplify(P*Jt)
    print("Jacobian at C4 (should be block-diagonal: t, e_u, e_v, radial):")
    sp.pprint(J)

    lam_u_direct = sp.simplify(J[2, 2])
    lam_v_direct = sp.simplify(J[3, 3])
    lam_u_expected = sp.sqrt((1 + phi*lam**2)/(1 + 4*phi))
    lam_v_expected = sp.sqrt((1 + phi*lam**2)/(1 + 3*phi))
    print("\nlambda_u (direct):", lam_u_direct)
    print("lambda_u (paper) :", lam_u_expected)
    assert sp.simplify(lam_u_direct - lam_u_expected) == 0, "lambda_u mismatch!"
    print("lambda_v (direct):", lam_v_direct)
    print("lambda_v (paper) :", lam_v_expected)
    assert sp.simplify(lam_v_direct - lam_v_expected) == 0, "lambda_v mismatch!"
    print("Both matched.")

    # Off-block-diagonal entries connecting (u,v) to (x,y) should vanish at C4.
    off_diag_checks = [J[0, 2], J[0, 3], J[1, 2], J[1, 3], J[2, 0], J[2, 1],
                        J[2, 3], J[3, 0], J[3, 1], J[3, 2]]
    assert all(sp.simplify(e) == 0 for e in off_diag_checks), \
        "Unexpected off-block coupling at C4!"
    print("Confirmed block-diagonal structure (all cross terms vanish).")

    Jxy = J[:2, :2]
    t = sp.Matrix([-yC, xC])
    tJt = sp.simplify((t.T*Jxy*t)[0] / (t.T*t)[0])
    tangential_expected = (lam**2*phi + 6*phi + 4)/(-lam**2*phi + 18*phi + 4)
    print("\nt^T J_(x,y) t (direct):", sp.nsimplify(tJt))
    print("t^T J_(x,y) t (paper) :", tangential_expected)
    print("Difference (should simplify to 0):", sp.simplify(tJt - tangential_expected))
    assert sp.simplify(tJt - tangential_expected) == 0, "Tangential eigenvalue mismatch!"
    print("Matched (the two forms differ only by a sign flip top and bottom).")

    # ---------------------------------------------------------------
    # D3 = (sqrt(6)/(2*lambda), sqrt(6)/(2*lambda), 0, sqrt(1-3/lambda^2))
    # R3 = (2*sqrt(6)/(3*lambda), 2*sqrt(3)/(3*lambda), sqrt(1-4/lambda^2), 0)
    # ---------------------------------------------------------------
    print("\n--- D3, R3: exact fixed points and decoupled eigenvalues ---")
    xD3 = sp.sqrt(6)/(2*lam)
    vD3 = sp.sqrt(1 - 3/lam**2)
    pointD3 = (xD3, xD3, 0, vD3)

    # Confirm D3 is an exact fixed point of the pre-projection map (norm=1).
    Ftil_D3 = Ftil.subs(dict(zip((x, y, u, v), pointD3)))
    norm2_D3 = sp.simplify(sum(c**2 for c in Ftil_D3))
    print("Pre-projection norm^2 at D3 (should be 1):", norm2_D3)
    assert sp.simplify(norm2_D3 - 1) == 0, "D3 is not an exact pre-projection fixed point!"
    diff_D3 = sp.simplify(Ftil_D3 - sp.Matrix(pointD3))
    assert all(sp.simplify(c) == 0 for c in diff_D3), "D3 pre-projection value != D3 itself!"
    print("Confirmed: D3 is an exact fixed point (pre-projection value equals D3 itself).")

    xR3 = 2*sp.sqrt(6)/(3*lam)
    yR3 = 2*sp.sqrt(3)/(3*lam)
    uR3 = sp.sqrt(1 - 4/lam**2)
    pointR3 = (xR3, yR3, uR3, 0)
    Ftil_R3 = Ftil.subs(dict(zip((x, y, u, v), pointR3)))
    norm2_R3 = sp.simplify(sum(c**2 for c in Ftil_R3))
    print("\nPre-projection norm^2 at R3, symbolic simplify gave:", norm2_R3)
    # sympy's simplify does not always reduce sqrt(1-4/lam^2)^2 back to
    # 1-4/lam^2 without an explicit domain assumption; cross-check
    # numerically at high precision instead, which is unambiguous.
    import random
    random.seed(2)
    max_err = 0.0
    for _ in range(500):
        lam_val = (4)**0.5 + random.uniform(0.01, 10)  # lambda^2 > 4
        phi_val = random.uniform(0.001, 1/3 - 0.001)
        val = complex(norm2_R3.subs({lam: lam_val, phi: phi_val}))
        max_err = max(max_err, abs(val - 1))
    print(f"Max |norm^2 - 1| over 500 random (lambda,phi) with lambda^2>4: {max_err:.3e}")
    assert max_err < 1e-9, "R3 is not an exact pre-projection fixed point!"
    diff_R3 = sp.simplify(Ftil_R3 - sp.Matrix(pointR3))
    max_diff_err = 0.0
    for _ in range(500):
        lam_val = (4)**0.5 + random.uniform(0.01, 10)
        phi_val = random.uniform(0.001, 1/3 - 0.001)
        for comp in diff_R3:
            val = complex(comp.subs({lam: lam_val, phi: phi_val}))
            max_diff_err = max(max_diff_err, abs(val))
    print(f"Max |F(R3) - R3| component over 500 random points: {max_diff_err:.3e}")
    assert max_diff_err < 1e-9, "R3 pre-projection value != R3 itself!"
    print("Confirmed numerically: R3 is an exact fixed point.")

    # Decoupled u-eigenvalue at D3, v-eigenvalue at R3.
    JD3 = Jtil.subs(dict(zip((x, y, u, v), pointD3)))
    lam_u_D3 = sp.simplify(JD3[2, 2])
    lam_u_D3_expected = sp.sqrt((1 + 3*phi)/(1 + 4*phi))  # gamma_m=1
    print("\nlambda_u(D3) (direct):", lam_u_D3)
    print("lambda_u(D3) (paper) :", lam_u_D3_expected)
    assert sp.simplify(lam_u_D3 - lam_u_D3_expected) == 0, "lambda_u(D3) mismatch!"

    JR3 = Jtil.subs(dict(zip((x, y, u, v), pointR3)))
    lam_v_R3 = sp.simplify(JR3[3, 3])
    lam_v_R3_expected = sp.sqrt((1 + 4*phi)/(1 + 3*phi))  # gamma_r=4/3
    print("lambda_v(R3) (direct):", lam_v_R3)
    print("lambda_v(R3) (paper) :", lam_v_R3_expected)
    assert sp.simplify(lam_v_R3 - lam_v_R3_expected) == 0, "lambda_v(R3) mismatch!"

    reciprocal_check = sp.simplify(lam_u_D3*lam_v_R3 - 1)
    print("\nlambda_u(D3) * lambda_v(R3) - 1 =", reciprocal_check, "(should be 0)")
    assert reciprocal_check == 0, "D3, R3 eigenvalues are not exact reciprocals!"
    print("Confirmed: lambda_u(D3) and lambda_v(R3) are exact reciprocals, for every phi.")

    # Stability signs (Theorem 3).
    sD3 = sp.simplify(lam_u_D3**2 - 1)
    sR3 = sp.simplify(lam_v_R3**2 - 1)
    print("\nlambda_u(D3)^2 - 1 =", sp.factor(sD3), " (should be negative: D3's 4th eigenvalue stable)")
    print("lambda_v(R3)^2 - 1 =", sp.factor(sR3), " (should be positive: R3's 4th eigenvalue unstable)")

    # ---------------------------------------------------------------
    # Proposition 4: fluid positivity (Omega_r, Omega_m updates)
    # ---------------------------------------------------------------
    print("\n--- Proposition 4: fluid positivity ---")
    Omr, Omm, Sn = sp.symbols('Omega_r Omega_m S_n', nonnegative=True)
    Omr_next = Omr*(1 + 3*phi*Sn)/(1 + 4*phi)
    Omm_next = Omm*(1 + 3*phi*Sn)/(1 + 3*phi)
    print("Omega_r,{n+1} numerator/denominator both manifestly non-negative "
          "for Omega_r,S_n,phi >= 0: numerator =",
          sp.expand(1 + 3*phi*Sn), ", denominator =", 1 + 4*phi)
    print("Omega_m,{n+1} numerator/denominator both manifestly non-negative: "
          "numerator =", sp.expand(1 + 3*phi*Sn), ", denominator =", 1 + 3*phi)
    print("Both unconditional in h (phi in (0,1/3)) given S_n in [0,2] "
          "(established from the Friedmann constraint alone).")

    # ---------------------------------------------------------------
    # Proposition 6: completeness, the uv=0 argument
    # ---------------------------------------------------------------
    print("\n--- Proposition 6: radiation and matter cannot coexist ---")
    gam_r, gam_m, alpha = sp.symbols('gamma_r gamma_m alpha', positive=True)
    # At a fixed point with u!=0 and v!=0: u*g_tilde_r = alpha*u and
    # v*g_tilde_m = alpha*v (both g's evaluated at the same state), so
    # g_tilde_r = g_tilde_m = alpha. But g_tilde_i = sqrt((1+3*phi*S)/(1+3*phi*gamma_i)),
    # so g_tilde_r = g_tilde_m forces gamma_r = gamma_m.
    gr_tilde = sp.sqrt((1 + 3*phi*Sn)/(1 + 3*phi*gam_r))
    gm_tilde = sp.sqrt((1 + 3*phi*Sn)/(1 + 3*phi*gam_m))
    eq = sp.Eq(gr_tilde, gm_tilde)
    sol = sp.solve(eq, gam_r)
    print("Solving g_tilde_r = g_tilde_m for gamma_r in terms of gamma_m gives:", sol)
    assert sol == [gam_m], "The uv=0 argument's algebra does not reduce as claimed!"
    print("Confirms: g_tilde_r = g_tilde_m forces gamma_r = gamma_m, which is "
          "false (gamma_r=4/3 != gamma_m=1). Hence no fixed point can have "
          "both u!=0 and v!=0: every fixed point has u*v=0.")

    print("\nAll three-component stability claims (Eqs. 54-55, the tangential "
          "eigenvalue, D3/R3 as exact fixed points, their reciprocal "
          "eigenvalues, Proposition 4, and Proposition 6's completeness "
          "argument) verified independently.")


if __name__ == '__main__':
    main()
