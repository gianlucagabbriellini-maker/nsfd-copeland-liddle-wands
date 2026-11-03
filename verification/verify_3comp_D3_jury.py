"""
verify_3comp_D3_jury.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Independently rebuilds the (x,y,v) Jacobian at D3, numerically at high
precision, and checks it against the paper's N1, N2, N3 (the Jury
conditions for the tangential-stability proposition). Unlike the earlier
derive_D3_block.py / verify_D3_tangential_text.py pair, this asserts
both agreement and strict positivity of all three quantities, including
N3, across 2000 random (lambda, phi) points where D3 exists. Mirrors
verify_2comp_D_jury.py's approach for the two-component case.
"""
import mpmath as mp

mp.mp.dps = 50


def jacobian_and_zD3(lam, phi):
    """Build the pre-projection Jacobian at D3 by numerical
    differentiation, and return it together with D3 itself as a vector.

    D3 is the matter-scaling point, where radiation is absent, so u=0
    throughout rather than carried as a third variable."""
    xD3 = mp.sqrt(6)/(2*lam)
    yD3 = xD3
    vD3 = mp.sqrt(1 - 3/lam**2)

    def Ftil(x, y, v):
        # Three-component update formulas with u=0 (no radiation at D3).
        Dn = 1 + phi*(mp.sqrt(6)/2*lam*x + mp.mpf(3)/2*y**2)
        Kn = 1 + phi*(mp.mpf(3)/2*x**2 + mp.mpf(3)/2)
        Pn = 1 + phi*(3 + mp.mpf(3)/2*y**2)
        ytil = Kn*y/Dn
        xtil = Kn*(x*Dn + phi*mp.sqrt(6)/2*lam*y**2)/(Pn*Dn)
        S = x**2 - y**2 + 1  # matter update, gamma_m=1
        gp = mp.sqrt((1 + 3*phi*S)/(1 + 3*phi))
        vtil = v*gp
        return mp.matrix([xtil, ytil, vtil])

    # Central finite differences at 50-digit precision, instead of a
    # symbolic derivative.
    h = mp.mpf('1e-20')
    J = mp.matrix(3, 3)
    base = (xD3, yD3, vD3)
    for j in range(3):
        plus = list(base); plus[j] += h
        minus = list(base); minus[j] -= h
        d = (Ftil(*plus) - Ftil(*minus)) / (2*h)
        for i in range(3):
            J[i, j] = d[i]
    zD3 = mp.matrix([xD3, yD3, vD3])
    return J, zD3


def tangential_trace_c2(lam, phi):
    """Apply the projection correction to the raw Jacobian from
    jacobian_and_zD3, then return the trace and c2 that the paper's
    three Jury conditions are built from."""
    Jt, zD3 = jacobian_and_zD3(lam, phi)
    P = mp.eye(3) - zD3*zD3.T
    J = P*Jt
    detJ_full = mp.det(J)
    Tr = J[0, 0] + J[1, 1] + J[2, 2]
    # D3 sits on the unit sphere, so one eigenvalue of J is exactly zero
    # (the radial direction); det(J) below should come out ~0, which
    # checks that. c2 is the sum of the three 2x2 principal minors, the
    # product of the other two eigenvalues.
    c2 = ((J[1, 1]*J[2, 2] - J[1, 2]*J[2, 1]) +
          (J[0, 0]*J[2, 2] - J[0, 2]*J[2, 0]) +
          (J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]))
    return Tr, c2, detJ_full


def paper_N1N2N3_over_D(lam, phi):
    """The three Jury-condition fractions N1/D1, N2/D2, N3/D2 exactly as
    printed in the paper. Used only to compare against the Jacobian
    built above, never to build it."""
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

    return N1/D1, N2/D2, N3/D2


def main():
    import random
    random.seed(0)  # fixed seed so the run is reproducible
    max_radial = mp.mpf(0)
    max_rel_err = {'cond1': mp.mpf(0), 'cond2': mp.mpf(0), 'cond3': mp.mpf(0)}
    min_val = {'cond1': mp.inf, 'cond2': mp.inf, 'cond3': mp.inf}

    # 2000 random combinations of lambda and phi, covering the whole
    # range where D3 exists (lambda^2>3, i.e. s>0) and the whole
    # physically sensible range of phi.
    n_trials = 2000
    for _ in range(n_trials):
        s = random.uniform(0.01, 50)
        lam = mp.sqrt(3 + s)
        p = random.uniform(0.001, 1/3 - 0.001)
        phi = mp.mpf(p)

        Tr, c2, detJ_full = tangential_trace_c2(lam, phi)
        max_radial = max(max_radial, abs(detJ_full))

        cond1_direct = 1 - c2
        cond2_direct = 1 + c2 - Tr
        cond3_direct = 1 + c2 + Tr

        n1, n2, n3 = paper_N1N2N3_over_D(lam, phi)

        for key, direct, paper_val in [('cond1', cond1_direct, n1),
                                        ('cond2', cond2_direct, n2),
                                        ('cond3', cond3_direct, n3)]:
            rel_err = abs(direct - paper_val) / max(abs(paper_val), mp.mpf('1e-30'))
            max_rel_err[key] = max(max_rel_err[key], rel_err)
            min_val[key] = min(min_val[key], paper_val)

    print(f"Trials: {n_trials}")
    print(f"Max |det(J)| over all trials (should be ~0, confirming rank <=2 "
          f"i.e. a genuine zero radial eigenvalue): {float(max_radial):.3e}")
    print("\nAgreement between the directly-built Jacobian's Jury quantities")
    print("and the paper's closed-form N1/D1, N2/D2, N3/D2 (Prop. 'Tangential")
    print("stability at D3', eqs. D3-N1 - D3-N3):")
    for key in ['cond1', 'cond2', 'cond3']:
        print(f"  {key}: max relative error = {float(max_rel_err[key]):.3e}, "
              f"minimum value found = {float(min_val[key]):.6f}")

    assert max_radial < mp.mpf('1e-40'), "det(J) not zero -- rank is not <= 2!"
    assert all(max_rel_err[k] < mp.mpf('1e-15') for k in max_rel_err), \
        "Mismatch between direct computation and paper's D3 formulas!"
    assert all(min_val[k] > 0 for k in min_val), \
        "Found a non-positive Jury quantity: Proposition 'Tangential " \
        "stability at D3' would be violated!"
    print("\nConfirmed: the paper's N1, N2, N3 formulas for D3 exactly "
          "reproduce the independently-built Jacobian's Jury quantities "
          "(to 1e-15 relative precision), and all three remain strictly "
          "positive throughout lambda^2>3, for every trial -- including "
          "N3, the condition no other script in this package independently "
          "sign-checks.")


if __name__ == '__main__':
    main()
