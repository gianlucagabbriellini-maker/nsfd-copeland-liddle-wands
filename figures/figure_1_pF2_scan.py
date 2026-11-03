"""
figure_pF2_final.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Generates Figure 1, the p.F2 scan along the transcritical threshold. Uses
the ambient two-component map, without the fluid variable or the
projection step, matching how this figure is described in the paper.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def step_2comp_final(x, y, phi, lam, gam):
    Kn = 1 + phi*(3*x**2 + 1.5*gam)
    Deltan = 1 + phi*(np.sqrt(6)/2*lam*x + 1.5*gam*(x**2+y**2))
    Pn = 1 + phi*(3 + 1.5*gam*(x**2+y**2))
    y1 = Kn*y/Deltan
    x1 = Kn*(x*Deltan + phi*np.sqrt(6)/2*lam*y**2)/(Pn*Deltan)
    return x1, y1

def pF2_numeric(theta, phi, eps=1e-5):
    gam = 2*np.cos(theta)**2
    lam = np.sqrt(6)*np.cos(theta)
    xC, yC = np.cos(theta), np.sin(theta)
    def f(dx, dy):
        return np.array(step_2comp_final(xC+dx, yC+dy, phi, lam, gam)) - np.array([xC, yC])
    fx = (f(eps, 0) - f(-eps, 0))/(2*eps)
    fy = (f(0, eps) - f(0, -eps))/(2*eps)
    Amat = np.array([fx, fy]).T
    a, b = Amat[0]
    c, d = Amat[1]
    q = np.array([b, 1-a])
    if np.linalg.norm(q) < 1e-7:
        q = np.array([1-d, c])
    if q[0] < 0:
        q = -q
    q = q/np.linalg.norm(q)
    p = np.array([d-1, -b])
    if np.linalg.norm(p) < 1e-7:
        p = np.array([-c, a-1])
    p = p/np.dot(p, q)
    def g(e):
        return f(e*q[0], e*q[1])
    d2 = (g(eps) - 2*g(0) + g(-eps))/eps**2
    F2 = d2/2
    return np.dot(p, F2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
phis = np.linspace(0.01, 1/3-0.01, 60)
for gam_target in [0.3, 0.7, 1.0, 1.4, 1.8]:
    theta = np.arccos(np.sqrt(gam_target/2))
    vals = [pF2_numeric(theta, ph) for ph in phis]
    ax.plot(phis, vals, label=fr'$\gamma={gam_target}$')
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\mathbf{p}\cdot\mathbf{F}_2$')
ax.axhline(0, color='gray', lw=0.5)
ax.legend(fontsize=8)
ax.set_title('vs. step size')

ax = axes[1]
gams = np.linspace(0.05, 1.95, 80)
for phi_target in [0.05, 0.15, 0.25, 0.32]:
    vals = []
    for g in gams:
        theta = np.arccos(np.sqrt(g/2))
        vals.append(pF2_numeric(theta, phi_target))
    ax.plot(gams, vals, label=fr'$\phi={phi_target}$')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\mathbf{p}\cdot\mathbf{F}_2$')
ax.axhline(0, color='gray', lw=0.5)
ax.legend(fontsize=8)
ax.set_title('vs. fluid equation of state')

plt.tight_layout()
os.makedirs('figs', exist_ok=True)  # so a fresh checkout doesn't crash on savefig
plt.savefig('figs/pF2_scan.pdf')
plt.savefig('figs/pF2_scan.png', dpi=150)
print('saved')
