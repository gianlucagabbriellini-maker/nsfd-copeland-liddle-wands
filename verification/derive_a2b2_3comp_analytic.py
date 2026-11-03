"""
derive_a2b2_3comp_analytic.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Analytic derivation of the center-manifold coefficients a_2, b_2, c_2 for
the three-component model at its transcritical threshold, lambda^2=3.
Here gamma_m=1 and gamma_r=4/3 are fixed numbers rather than symbolic
parameters, so x_C=y_C=1/sqrt(2) are exact, making this more direct than
the two-component case.

Reproduces:
  - c_2 = 0, from the strict stability of the u-direction,
  - the 2x2 linear system for a_2, b_2 and its solution a_2=-x_C, b_2=0,
  - the cubic coefficient f'''(0) = -9*phi/(3*phi+1).
"""
import sympy as sp


def main():
    phi, v = sp.symbols('phi v', positive=True)
    a2, b2, c2 = sp.symbols('a_2 b_2 c_2')

    lam2 = 3  # threshold lambda^2 = 3*gamma_m, gamma_m = 1
    xC = sp.sqrt(lam2)/sp.sqrt(6)
    yC = sp.sqrt(1 - sp.Rational(lam2, 6))
    print("x_C =", xC, " y_C =", yC, " (should both be 1/sqrt(2))")
    assert sp.simplify(xC - 1/sp.sqrt(2)) == 0
    assert sp.simplify(yC - 1/sp.sqrt(2)) == 0

    lam = sp.sqrt(lam2)

    # --- Step 1: c_2 = 0, from strict stability of the u (radiation) direction ---
    # lambda_u = sqrt((1+phi*lambda^2)/(1+4*phi)) at this threshold (Eq. 54).
    lam_u = sp.sqrt((1 + phi*lam2)/(1 + 4*phi))
    is_lam_u_lt_1 = sp.simplify((1 + phi*lam2) - (1 + 4*phi))  # numerator of lam_u^2-1
    print("\nlambda_u^2 - 1 numerator (should be negative for phi>0):",
          sp.factor(is_lam_u_lt_1))
    # c_2*(lambda_u - 1) = 0 forces c_2 = 0 since lambda_u < 1 strictly.
    print("=> c_2 = 0 (the only solution of c_2*(lambda_u-1)=0 with lambda_u<1)")

    # --- Step 2: the graph, with c_2=0 already substituted ---
    x = xC + a2*v**2
    y = yC + b2*v**2
    u = 0  # c_2 = 0

    # Pre-projection (x,y) block, three-component version (Eqs. 43-44).
    Kn = 1 + phi*(sp.Rational(3, 2)*x**2 + sp.Rational(3, 2) + sp.Rational(1, 2)*u**2)
    Dn = 1 + phi*(sp.sqrt(6)/2*lam*x + sp.Rational(3, 2)*y**2)
    Pn = 1 + phi*(3 + sp.Rational(3, 2)*y**2)
    ytil = Kn*y/Dn
    xtil = Kn*(x*Dn + phi*sp.sqrt(6)/2*lam*y**2)/(Pn*Dn)

    xtil_s = sp.series(xtil, v, 0, 3).removeO()
    ytil_s = sp.series(ytil, v, 0, 3).removeO()
    cx2 = sp.nsimplify(sp.simplify(xtil_s.coeff(v, 2)))
    cy2 = sp.nsimplify(sp.simplify(ytil_s.coeff(v, 2)))
    print("\nPre-projection O(v^2) coefficients (linear in a_2, b_2):")
    print("  coeff of x-tilde:", cx2)
    print("  coeff of y-tilde:", cy2)

    # u,v components' O(v^2)/O(1) behavior, needed for the projection norm.
    # S(x,y,u) = x^2 - y^2 + 1 + u^2/3, with u=0 here.
    S = x**2 - y**2 + 1
    g = sp.sqrt((1 + 3*phi*S)/(1 + 4*phi))     # multiplies u=0 -> util = 0
    gp = sp.sqrt((1 + 3*phi*S)/(1 + 3*phi))    # multiplies v
    gp_at_0 = gp.subs(v, 0)
    print("  g'(x_C,y_C,0) (should equal lambda_v):", sp.nsimplify(gp_at_0))

    # --- Step 3: projection correction D*Pi(C4) = I - C4 C4^T at O(v^2) ---
    # Ambient state is (x,y,u,v); at C4=(xC,yC,0,0) with u=util=0 identically
    # here (c_2=0), and v~O(v) itself (the center coordinate). The norm is
    # N^2 = xtil^2+ytil^2+util^2+vtil^2 = 1 + K v^2 + O(v^4), and the
    # v-independent part of vtil^2/v^2 at v=0 equals gp_at_0^2.
    K = 2*xC*cx2 + 2*yC*cy2 + gp_at_0**2
    proj_x2 = sp.simplify(cx2 - xC*K/2)
    proj_y2 = sp.simplify(cy2 - yC*K/2)

    E1 = sp.expand(proj_x2 - a2)
    E2 = sp.expand(proj_y2 - b2)
    M11 = sp.simplify(sp.diff(E1, a2)); M12 = sp.simplify(sp.diff(E1, b2))
    M21 = sp.simplify(sp.diff(E2, a2)); M22 = sp.simplify(sp.diff(E2, b2))
    r1 = sp.simplify(-(E1 - M11*a2 - M12*b2))
    r2 = sp.simplify(-(E2 - M21*a2 - M22*b2))

    print("\nExplicit 2x2 linear system for a_2, b_2:")
    print(f"  ({M11}) a_2 + ({M12}) b_2 = {r1}")
    print(f"  ({M21}) a_2 + ({M22}) b_2 = {r2}")

    det = sp.simplify(M11*M22 - M12*M21)
    print("\nDeterminant:", det, " (nonzero for phi>0 =>  unique solution)")

    sol = sp.solve([sp.Eq(M11*a2 + M12*b2, r1),
                     sp.Eq(M21*a2 + M22*b2, r2)], [a2, b2], dict=True)[0]
    A2 = sp.simplify(sol[a2])
    B2 = sp.simplify(sol[b2])
    print("\nSolution: a_2 =", A2, "  b_2 =", B2)

    match_a2 = sp.simplify(A2 + xC) == 0
    match_b2 = sp.simplify(B2) == 0
    print("Matches a_2 = -x_C:", match_a2)
    print("Matches b_2 = 0   :", match_b2)
    assert match_a2 and match_b2, "Mismatch with the paper's Eq. (64)!"

    # --- Step 4: the cubic coefficient f'''(0) ---
    # With a_2,b_2,c_2 now fixed, expand v-tilde/N to O(v^3) and read off
    # the reduced map f(v) = v_{n+1}.
    a2v, b2v = A2, B2
    xg = xC + a2v*v**2
    yg = yC + b2v*v**2
    Sg = xg**2 - yg**2 + 1
    gp_full = sp.sqrt((1 + 3*phi*Sg)/(1 + 3*phi))
    vtil = v*gp_full

    Kn_g = 1 + phi*(sp.Rational(3, 2)*xg**2 + sp.Rational(3, 2))
    Dn_g = 1 + phi*(sp.sqrt(6)/2*lam*xg + sp.Rational(3, 2)*yg**2)
    Pn_g = 1 + phi*(3 + sp.Rational(3, 2)*yg**2)
    ytil_g = Kn_g*yg/Dn_g
    xtil_g = Kn_g*(xg*Dn_g + phi*sp.sqrt(6)/2*lam*yg**2)/(Pn_g*Dn_g)

    N2 = sp.series(xtil_g**2 + ytil_g**2 + vtil**2, v, 0, 4).removeO()
    N = sp.sqrt(N2)
    fv = sp.series(vtil/N, v, 0, 4).removeO()
    fv = sp.simplify(fv)

    fprime0 = sp.simplify(sp.diff(fv, v, 1).subs(v, 0))
    fpp0 = sp.diff(fv, v, 2).subs(v, 0)
    fppp0 = sp.simplify(sp.diff(fv, v, 3).subs(v, 0))
    print("\nReduced map f(v)=v_{n+1}:")
    print("  f'(0)   =", fprime0)
    print("  f''(0)  =", fpp0)
    print("  f'''(0) =", fppp0)

    expected = -9*phi/(3*phi + 1)
    match_fppp = sp.simplify(fppp0 - expected) == 0
    print("Matches f'''(0) = -9*phi/(3*phi+1):", match_fppp)
    assert fprime0 == 1 and fpp0 == 0 and match_fppp, \
        "Mismatch with the paper's Eq. (65)!"

    print("\nVerified: matches Eqs. (64)-(65) of the paper exactly.")


if __name__ == '__main__':
    main()
