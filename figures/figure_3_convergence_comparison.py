"""
figure_convergence_final.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Generates Figure 3, the error-vs-step-size comparison of Euler, RK4 and
NSFD at (lambda,gamma)=(3,1). The NSFD trajectory tracks
w=sqrt(Omega_gamma) and projects after every step; it previously tracked
only (x,y) without projecting, which was not the scheme the paper's
well-posedness and invariance proposition actually covers.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def step_2comp_final(x, y, w, phi, lam, gam=1.0):
    Kn = 1 + phi*(3*x**2 + 1.5*gam)
    Deltan = 1 + phi*(np.sqrt(6)/2*lam*x + 1.5*gam*(x**2+y**2))
    Pn = 1 + phi*(3 + 1.5*gam*(x**2+y**2))
    y1 = Kn*y/Deltan
    x1 = Kn*(x*Deltan + phi*np.sqrt(6)/2*lam*y**2)/(Pn*Deltan)
    S = (2-gam)*x**2 - gam*y**2 + gam
    Omega1 = w**2*(1+3*phi*S)/(1+3*gam*phi)
    w1 = np.sqrt(max(Omega1, 0.0))
    norm = np.sqrt(x1**2+y1**2+w1**2)
    return x1/norm, y1/norm, w1/norm

def phi_of_h(h):
    return (1-np.exp(-3*h))/3

def rhs(x, y, lam, gam=1.0):
    S = (2-gam)*x**2 - gam*y**2 + gam
    dx = -3*x + np.sqrt(6)/2*lam*y**2 + 1.5*x*S
    dy = -np.sqrt(6)/2*lam*x*y + 1.5*y*S
    return dx, dy

def euler_step(x, y, h, lam, gam=1.0):
    dx, dy = rhs(x, y, lam, gam)
    return x+h*dx, y+h*dy

def rk4_step(x, y, h, lam, gam=1.0):
    k1x, k1y = rhs(x, y, lam, gam)
    k2x, k2y = rhs(x+h/2*k1x, y+h/2*k1y, lam, gam)
    k3x, k3y = rhs(x+h/2*k2x, y+h/2*k2y, lam, gam)
    k4x, k4y = rhs(x+h*k3x, y+h*k3y, lam, gam)
    return x+h/6*(k1x+2*k2x+2*k3x+k4x), y+h/6*(k1y+2*k2y+2*k3y+k4y)

lam, gam = 3.0, 1.0
x0, y0 = 0.05, 0.01
Ntot = 10.0

href = 1e-4
n = int(Ntot/href)
x, y = x0, y0
for _ in range(n):
    x, y = rk4_step(x, y, href, lam, gam)
xref, yref = x, y

hs = np.array([0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,1.0,1.3,1.5,1.8,2.0,2.05,2.1,2.2,2.4])
err_euler, err_rk4, err_nsfd = [], [], []
for h in hs:
    n = int(Ntot/h)
    x, y, ok = x0, y0, True
    for _ in range(n):
        x, y = euler_step(x, y, h, lam, gam)
        if not np.isfinite(x) or abs(x) > 1e8:
            ok = False; break
    err_euler.append(np.hypot(x-xref, y-yref) if ok else np.nan)

    x, y, ok = x0, y0, True
    for _ in range(n):
        with np.errstate(over='ignore', invalid='ignore'):
            x, y = rk4_step(x, y, h, lam, gam)
        if not np.isfinite(x) or abs(x) > 1e8:
            ok = False; break
    err_rk4.append(np.hypot(x-xref, y-yref) if ok else np.nan)

    x, y, w, phi = x0, y0, np.sqrt(1-x0**2-y0**2), phi_of_h(h)
    ok = True
    for _ in range(n):
        x, y, w = step_2comp_final(x, y, w, phi, lam, gam)
        if not np.isfinite(x) or abs(x) > 1e8:
            ok = False; break
    err_nsfd.append(np.hypot(x-xref, y-yref) if ok else np.nan)

fig, ax = plt.subplots(figsize=(6.5,4.8))
ax.loglog(hs, err_euler, 'o-', label='Euler', color='tab:red', markersize=4)
ax.loglog(hs, err_rk4, 's-', label='RK4', color='tab:green', markersize=4)
ax.loglog(hs, err_nsfd, '^-', label='NSFD', color='tab:blue', markersize=4)
ax.axvline(0.6, color='tab:red', ls=':', lw=1, alpha=0.6)
ax.axvline(2.05, color='tab:green', ls=':', lw=1, alpha=0.6)
ax.text(0.63, 2e-5, 'Euler\ndiverges', color='tab:red', fontsize=8, va='bottom')
ax.text(2.15, 2e-5, 'RK4\ndiverges', color='tab:green', fontsize=8, va='bottom')
ax.set_xlabel(r'step size $h$')
ax.set_ylabel(r'error at $N=10$ (Euclidean distance from reference)')
ax.set_title(r'Error vs.\ step size, $\lambda=3,\ \gamma=1$')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim(1e-11, 1e2)
plt.tight_layout()
os.makedirs('figs', exist_ok=True)  # so a fresh checkout doesn't crash on savefig
plt.savefig('figs/convergence.pdf')
plt.savefig('figs/convergence.png', dpi=150)
print('saved')
for h,e,r,ns in zip(hs,err_euler,err_rk4,err_nsfd):
    print(f'h={h:.3f}  euler={e}  rk4={r}  nsfd={ns:.4e}')
