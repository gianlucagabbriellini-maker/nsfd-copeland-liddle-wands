"""
verify_point5_checks.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Runs the four checks from the second review round, point 5: reference-
solution accuracy, step-refinement order, maximum-in-time error, and
behaviour near the Delta_n, P_n thresholds, for the same trajectory as
Figure 3. Now tracks w=sqrt(Omega_gamma) and projects every step; it
previously tracked only (x,y) without projecting, which was not the
scheme the well-posedness proposition actually covers.
"""
import numpy as np
from scipy.integrate import solve_ivp

GAM = 1.0

def rhs(t, state, lam, gam=GAM):
    x, y = state
    S = (2-gam)*x**2 - gam*y**2 + gam
    dx = -3*x + np.sqrt(6)/2*lam*y**2 + 1.5*x*S
    dy = -np.sqrt(6)/2*lam*x*y + 1.5*y*S
    return [dx, dy]

def phi_of_h(h):
    return (1-np.exp(-3*h))/3

def step(x, y, w, phi, lam, gam=GAM):
    Kn = 1 + phi*(3*x**2 + 1.5*gam)
    Deltan = 1 + phi*(np.sqrt(6)/2*lam*x + 1.5*gam*(x**2+y**2))
    Pn = 1 + phi*(3 + 1.5*gam*(x**2+y**2))
    y1 = Kn*y/Deltan
    x1 = Kn*(x*Deltan + phi*np.sqrt(6)/2*lam*y**2)/(Pn*Deltan)
    S = (2-gam)*x**2 - gam*y**2 + gam
    Omega1 = w**2*(1+3*phi*S)/(1+3*gam*phi)
    w1 = np.sqrt(max(Omega1, 0.0))
    norm = np.sqrt(x1**2+y1**2+w1**2)
    return x1/norm, y1/norm, w1/norm, Deltan, Pn

def nsfd_traj(x0, y0, h, lam, Ntot, gam=GAM):
    n = int(Ntot/h)
    phi = phi_of_h(h)
    x, y, w = x0, y0, np.sqrt(max(1-x0**2-y0**2, 0))
    traj = [(0.0, x, y)]
    mind, minp = np.inf, np.inf
    for k in range(n):
        x, y, w, Dn, Pn = step(x, y, w, phi, lam, gam)
        mind = min(mind, Dn); minp = min(minp, Pn)
        if not np.isfinite(x) or abs(x) > 1e8:
            return None, mind, minp
        traj.append(((k+1)*h, x, y))
    return traj, mind, minp

lam, x0, y0, Ntot = 3.0, 0.05, 0.01, 10.0

print("=== (1) Reference-solution accuracy: RK45 rtol=1e-8 vs rtol=1e-12 ===")
sol_a = solve_ivp(rhs, [0, Ntot], [x0, y0], args=(lam,), method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)
sol_b = solve_ivp(rhs, [0, Ntot], [x0, y0], args=(lam,), method='RK45', rtol=1e-12, atol=1e-14, dense_output=True)
ts = np.linspace(0, Ntot, 2000)
diffs = np.hypot(*(sol_a.sol(ts) - sol_b.sol(ts)))
print(f"max |ref(1e-8) - ref(1e-12)| over trajectory: {diffs.max():.3e}  (vs NSFD discretization error ~1e-2-1e-1)")

print()
print("=== (2) Step-refinement: empirical convergence order ===")
href_fine = 1e-4
from_rk4 = True
def rk4_step(x, y, h, lam, gam=GAM):
    def f(x,y):
        S = (2-gam)*x**2 - gam*y**2 + gam
        return -3*x + np.sqrt(6)/2*lam*y**2 + 1.5*x*S, -np.sqrt(6)/2*lam*x*y + 1.5*y*S
    k1x,k1y = f(x,y)
    k2x,k2y = f(x+h/2*k1x, y+h/2*k1y)
    k3x,k3y = f(x+h/2*k2x, y+h/2*k2y)
    k4x,k4y = f(x+h*k3x, y+h*k3y)
    return x+h/6*(k1x+2*k2x+2*k3x+k4x), y+h/6*(k1y+2*k2y+2*k3y+k4y)

x,y = x0,y0
for _ in range(int(Ntot/href_fine)):
    x,y = rk4_step(x,y,href_fine,lam)
xref, yref = x, y

hs_ref = [0.08, 0.04, 0.02, 0.01, 0.005]
errs_nsfd, errs_rk4 = [], []
for h in hs_ref:
    traj, _, _ = nsfd_traj(x0, y0, h, lam, Ntot)
    xn, yn = traj[-1][1], traj[-1][2]
    errs_nsfd.append(np.hypot(xn-xref, yn-yref))
    x,y = x0,y0
    for _ in range(int(Ntot/h)):
        x,y = rk4_step(x,y,h,lam)
    errs_rk4.append(np.hypot(x-xref,y-yref))

print(f"{'h':>8} {'err_NSFD':>14} {'err_RK4':>14}")
for h,en,er in zip(hs_ref, errs_nsfd, errs_rk4):
    print(f"{h:8.4f} {en:14.6e} {er:14.6e}")

orders_nsfd = [np.log(errs_nsfd[i]/errs_nsfd[i+1])/np.log(hs_ref[i]/hs_ref[i+1]) for i in range(len(hs_ref)-1)]
orders_rk4 = [np.log(errs_rk4[i]/errs_rk4[i+1])/np.log(hs_ref[i]/hs_ref[i+1]) for i in range(len(hs_ref)-1)]
print("empirical order NSFD (successive halvings):", [f"{o:.2f}" for o in orders_nsfd])
print("empirical order RK4  (successive halvings):", [f"{o:.2f}" for o in orders_rk4])

print()
print("=== (3) Maximum-in-time error (not just at N=10) ===")
h = 0.01
phi = phi_of_h(h)
x, y, w = x0, y0, np.sqrt(max(1-x0**2-y0**2, 0))
sol_ref = sol_b  # high-accuracy reference, dense output
ts_check = []
errs_time = []
n = int(Ntot/h)
for k in range(n):
    x, y, w, Dn, Pn = step(x, y, w, phi, lam)
    t = (k+1)*h
    xr, yr = sol_ref.sol(t)
    errs_time.append(np.hypot(x-xr, y-yr))
    ts_check.append(t)
errs_time = np.array(errs_time); ts_check = np.array(ts_check)
imax = np.argmax(errs_time)
print(f"error at N=10 (endpoint): {errs_time[-1]:.4e}")
print(f"MAXIMUM error over whole trajectory: {errs_time[imax]:.4e}, attained at N={ts_check[imax]:.2f}")

print()
print("=== (4) Behaviour near the denominator restrictions (Delta_n, P_n small) ===")
# Near-threshold case: lambda close to lambda_A(gamma) at gamma=1 (lambda_A(1) is the
# boundary beyond which Delta_n>0 is only conditional in h, per Proposition 1).
# lambda_A(gamma)=sqrt(6)(2+gamma)/2 ; at gamma=1, lambda_A=3*sqrt(6)/2 ~ 3.674
lamA = np.sqrt(6)*(2+GAM)/2
print(f"lambda_A(gamma=1) = {lamA:.6f} (Proposition 1 threshold)")
# h=0.6 is past the true critical h for this lambda (~0.489), where
# Delta_n is confirmed negative; h=0.3 used previously was still below
# it, so it never demonstrated the conditional failure this case needs.
for lam_test, h_test in [(lamA*0.999, 0.05), (lamA*1.2, 0.05), (lamA*1.2, 0.6)]:
    traj, mind, minp = nsfd_traj(-0.99, 0.01, h_test, lam_test, 5.0)
    # Delta_n/P_n sign and divergence are reported separately: a
    # trajectory can diverge for reasons unrelated to Delta_n, P_n.
    well_posed = mind > 0 and minp > 0
    bits = []
    bits.append("Delta_n,P_n>0 throughout" if well_posed else "Delta_n or P_n turned non-positive")
    bits.append("no divergence" if traj is not None else "trajectory diverged")
    print(f"  lambda={lam_test:.4f} ({'below' if lam_test<lamA else 'above'} lambda_A), h={h_test}: "
          f"min Delta_n={mind:.6f}, min P_n={minp:.6f} -> {', '.join(bits)}")
