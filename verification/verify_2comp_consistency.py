"""
verify_2comp_consistency.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Checks Rule I for the full coupled two-component map: one NSFD step
should match the continuous flow to O(h^2) locally. A symbolic check
was tried first, but the nested square roots made it too slow, so this
uses high-precision Richardson extrapolation instead, confirming the
one-step error shrinks by about 4x each time h is halved.
"""
import mpmath as mp

def main():
    print("--- High-precision numerical Richardson check of local truncation order ---\n")
    mp.mp.dps = 30
    x0, y0, w0, lam0, gam0 = mp.mpf('0.3'), mp.mpf('0.7'), mp.mpf('0.6481'), mp.mpf(2), mp.mpf(1)

    def rhs_num(t, state):
        xx, yy, ww = state
        S = 2*xx**2 + gam0*ww**2
        dx = -3*xx + mp.sqrt(6)/2*lam0*yy**2 + mp.mpf(3)/2*xx*S
        dy = -mp.sqrt(6)/2*lam0*xx*yy + mp.mpf(3)/2*yy*S
        dw = mp.mpf(3)/2*ww*(S - gam0)
        return [dx, dy, dw]

    def rk4_step(state, hh):
        k1 = rhs_num(0, state)
        s2 = [state[i] + hh/2*k1[i] for i in range(3)]
        k2 = rhs_num(0, s2)
        s3 = [state[i] + hh/2*k2[i] for i in range(3)]
        k3 = rhs_num(0, s3)
        s4 = [state[i] + hh*k3[i] for i in range(3)]
        k4 = rhs_num(0, s4)
        return [state[i] + hh/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(3)]

    def discrete_step_num(state, hh):
        xx, yy, ww = state
        phi_val = (1 - mp.e**(-3*hh))/3
        Kn = 1 + phi_val*(3*xx**2 + mp.mpf(3)/2*gam0)
        Dn = 1 + phi_val*(mp.sqrt(6)/2*lam0*xx + mp.mpf(3)/2*gam0*(xx**2 + yy**2))
        Pn = 1 + phi_val*(3 + mp.mpf(3)/2*gam0*(xx**2 + yy**2))
        ytil = Kn*yy/Dn
        xtil = Kn*(xx*Dn + phi_val*mp.sqrt(6)/2*lam0*yy**2)/(Pn*Dn)
        S = 2*xx**2 + gam0*ww**2
        wtil = ww*mp.sqrt((1 + 3*phi_val*S)/(1 + 3*gam0*phi_val))
        norm = mp.sqrt(xtil**2 + ytil**2 + wtil**2)
        return [xtil/norm, ytil/norm, wtil/norm]

    state0 = [x0, y0, w0]
    errors = []
    for hh in [mp.mpf('0.02'), mp.mpf('0.01'), mp.mpf('0.005')]:
        # very fine RK4 sub-stepping as a proxy for the exact flow over [0,hh]
        n_sub = 4000
        s = list(state0)
        for _ in range(n_sub):
            s = rk4_step(s, hh/n_sub)
        d = discrete_step_num(state0, hh)
        err = mp.sqrt(sum((s[i] - d[i])**2 for i in range(3)))
        errors.append(err)
        print(f"  h={float(hh):.4f}: one-step error = {float(err):.6e}")

    ratio1 = errors[0]/errors[1]
    ratio2 = errors[1]/errors[2]
    print(f"\n  Error ratio at h-halving: {float(ratio1):.3f}, {float(ratio2):.3f} "
          f"(should be close to 4, confirming O(h^2) local truncation)")
    assert 3.5 < ratio1 < 4.5 and 3.5 < ratio2 < 4.5, \
        "Local truncation order does not match O(h^2)!"
    print("Confirmed: one-step error scales as O(h^2), i.e. the scheme is "
          "first-order accurate (global error O(h)), for the complete "
          "coupled map.")


if __name__ == '__main__':
    main()
