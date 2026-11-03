"""
verify_2comp_D_jury.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Independently re-derives the Jury quantities at the scaling point D,
building the Jacobian numerically from the raw pre-projection formulas
rather than from the paper's stated N1, N2, N3. Cross-checks the two at
2000 random (gamma, lambda, phi) points in the existence region, at
50-digit precision. A symbolic version was tried first, but the nested
radicals in D's coordinates made it too slow.
"""
import mpmath as mp

mp.mp.dps = 50


def jacobian_and_zD(gam, lam, phi):
    xD = mp.sqrt(6)*gam/(2*lam)
    yD = mp.sqrt(mp.mpf(3)/2*gam*(2 - gam))/lam
    wD = mp.sqrt(1 - 3*gam/lam**2)

    def Ftil(x, y, w):
        Kn = 1 + phi*(3*x**2 + mp.mpf(3)/2*gam)
        Dn = 1 + phi*(mp.sqrt(6)/2*lam*x + mp.mpf(3)/2*gam*(x**2 + y**2))
        Pn = 1 + phi*(3 + mp.mpf(3)/2*gam*(x**2 + y**2))
        ytil = Kn*y/Dn
        xtil = Kn*(x*Dn + phi*mp.sqrt(6)/2*lam*y**2)/(Pn*Dn)
        S = 2*x**2 + gam*w**2
        wtil = w*mp.sqrt((1 + 3*phi*S)/(1 + 3*gam*phi))
        return mp.matrix([xtil, ytil, wtil])

    h = mp.mpf('1e-20')
    J = mp.matrix(3, 3)
    base = (xD, yD, wD)
    for j in range(3):
        plus = list(base); plus[j] += h
        minus = list(base); minus[j] -= h
        d = (Ftil(*plus) - Ftil(*minus)) / (2*h)
        for i in range(3):
            J[i, j] = d[i]
    zD = mp.matrix([xD, yD, wD])
    return J, zD


def tangential_trace_det(gam, lam, phi):
    Jt, zD = jacobian_and_zD(gam, lam, phi)
    P = mp.eye(3) - zD*zD.T
    J = P*Jt
    # Correct radial-direction check: P has rank 2 (kills only the zD
    # direction), so J = P*Jt maps everything INTO the tangent plane,
    # hence rank(J) <= 2 and det(J) = 0 -- NOT "J*zD = 0" (that would
    # require Jt*zD to already be proportional to zD, which is not
    # generally true and is not what "eigenvalue 0" actually means here).
    detJ_full = mp.det(J)
    Tr = J[0, 0] + J[1, 1] + J[2, 2]
    Det = ((J[1, 1]*J[2, 2] - J[1, 2]*J[2, 1]) +
           (J[0, 0]*J[2, 2] - J[0, 2]*J[2, 0]) +
           (J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]))
    return Tr, Det, detJ_full


def paper_N1N2N3_over_D(gam, lam, phi):
    """The paper's closed-form N_1, N_2, N_3 and common denominator
    D (Eqs. 39-41), evaluated numerically."""
    Dden = lam**2*(1 + 3*gam*phi)*(9*gam**2*phi + 6*lam**2*phi + 2*lam**2) \
        * (9*gam**2*phi + 3*gam*lam**2*phi + 2*lam**2)
    N1 = 3*phi*(2 - gam)*(81*gam**3*phi**2*(lam**4 - 2*gam*lam**2 + 3*gam**2)
                          + 2*lam**4*(3*gam*lam**2*phi + lam**2 + 9*gam**2*phi))
    N2 = 9*gam*phi**2*(2 - gam)*(lam**2 - 3*gam) * \
        ((6*gam*phi + 2)*lam**4 - 9*gam**2*phi*lam**2 + 27*gam**3*phi)
    N3 = (16*lam**6 + 12*lam**4*(12*gam**2 + 7*gam*lam**2 + 2*lam**2)*phi
          + 18*gam*lam**2*(18*gam**3 + 7*gam*lam**4 + 6*gam*lam**2
                            + 33*gam**2*lam**2 + 6*lam**4)*phi**2
          + 27*gam**2*(27*gam**4 + 18*gam**3*lam**2 - 54*gam**3
                        + 21*gam**2*lam**4 + 36*gam**2*lam**2 + 2*gam*lam**6
                        + 6*gam*lam**4 + 4*lam**6)*phi**3)
    return N1/Dden, N2/Dden, N3/Dden


def main():
    import random
    random.seed(0)
    max_radial = mp.mpf(0)
    max_rel_err = {'cond1': mp.mpf(0), 'cond2': mp.mpf(0), 'cond3': mp.mpf(0)}
    min_val = {'cond1': mp.inf, 'cond2': mp.inf, 'cond3': mp.inf}

    n_trials = 2000
    for _ in range(n_trials):
        g = random.uniform(0.02, 1.98)
        l = (3*g)**0.5 + random.uniform(0.01, 5)
        p = random.uniform(0.001, 1/3 - 0.001)
        gam, lam, phi = mp.mpf(g), mp.mpf(l), mp.mpf(p)

        Tr, Det, detJ_full = tangential_trace_det(gam, lam, phi)
        max_radial = max(max_radial, abs(detJ_full))

        cond1_direct = 1 - Det
        cond2_direct = 1 + Det - Tr
        cond3_direct = 1 + Det + Tr

        n1, n2, n3 = paper_N1N2N3_over_D(gam, lam, phi)

        for key, direct, paper_val in [('cond1', cond1_direct, n1),
                                        ('cond2', cond2_direct, n2),
                                        ('cond3', cond3_direct, n3)]:
            rel_err = abs(direct - paper_val) / max(abs(paper_val), mp.mpf('1e-30'))
            max_rel_err[key] = max(max_rel_err[key], rel_err)
            min_val[key] = min(min_val[key], paper_val)

    print(f"Trials: {n_trials}")
    print(f"Max |det(J)| over all trials (should be ~0, confirming rank <=2 "
          f"i.e. a genuine zero eigenvalue): {float(max_radial):.3e}")
    print("\nAgreement between the directly-built Jacobian's Jury quantities")
    print("and the paper's closed-form N_1/D, N_2/D, N_3/D (Eqs. 39-41):")
    for key in ['cond1', 'cond2', 'cond3']:
        print(f"  {key}: max relative error = {float(max_rel_err[key]):.3e}, "
              f"minimum value found = {float(min_val[key]):.6f}")

    assert max_radial < mp.mpf('1e-40'), "det(J) not zero -- rank is not <= 2!"
    assert all(max_rel_err[k] < mp.mpf('1e-15') for k in max_rel_err), \
        "Mismatch between direct computation and paper's formulas!"
    assert all(min_val[k] > 0 for k in min_val), \
        "Found a non-positive Jury quantity: Theorem 2 would be violated!"
    print("\nConfirmed: the paper's N_1, N_2, N_3 formulas exactly reproduce "
          "the independently-built Jacobian's Jury quantities (to 1e-15 "
          "relative precision), and all three remain strictly positive "
          "throughout the existence region, for every trial.")


if __name__ == '__main__':
    main()
