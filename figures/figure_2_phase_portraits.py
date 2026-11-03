"""
figure_phase_final.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Generates Figure 2, four phase portraits comparing NSFD steps to a
high-precision reference trajectory. No closed-form solution exists for
this system, so the reference is numerical (adaptive RK45, rtol=1e-8),
not analytical; its error is several orders of magnitude below the NSFD
discretization error being shown, so it stands in for the true
trajectory. Uses the full projected map for both models.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

GAM_M = 1.0
H_DISCRETE = 0.01

def rhs_2comp(t, state, lam):
    x, y = state
    S = (2-GAM_M)*x**2 - GAM_M*y**2 + GAM_M
    dx = -3*x + np.sqrt(6)/2*lam*y**2 + 1.5*x*S
    dy = -np.sqrt(6)/2*lam*x*y + 1.5*y*S
    return [dx, dy]

def rhs_3comp(t, state, lam):
    x, y, u = state
    S = x**2 - y**2 + 1 + u**2/3
    dx = -3*x + np.sqrt(6)/2*lam*y**2 + 1.5*x*S
    dy = -np.sqrt(6)/2*lam*x*y + 1.5*y*S
    du = 0.5*u*(3*S-4)
    return [dx, dy, du]

def phi_of_h(h):
    return (1-np.exp(-3*h))/3

def step_2comp_final(x, y, w, phi, lam, gam=GAM_M):
    """Same fix as in figure_3_convergence_comparison.py: this used to
    return the pre-projection (x,y) only, unlike step_3comp_final just
    below. Now it tracks w=sqrt(Omega_gamma) too and projects after
    every step, matching the real scheme."""
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

def step_3comp_final(x, y, u, v, phi, lam):
    Kn = 1 + phi*(1.5*x**2 + 1.5 + 0.5*u**2)
    Deltan = 1 + phi*(np.sqrt(6)/2*lam*x + 1.5*y**2)
    Pn = 1 + phi*(3 + 1.5*y**2)
    y1 = Kn*y/Deltan
    x1 = Kn*(x*Deltan + phi*np.sqrt(6)/2*lam*y**2)/(Pn*Deltan)
    S = x**2 - y**2 + 1 + u**2/3
    Omega_r1 = u**2*(1+3*phi*S)/(1+4*phi)
    Omega_m1 = v**2*(1+3*phi*S)/(1+3*phi)
    u1 = np.sqrt(max(Omega_r1,0))
    v1 = np.sqrt(max(Omega_m1,0))
    norm = np.sqrt(x1**2+y1**2+u1**2+v1**2)
    return x1/norm, y1/norm, u1/norm, v1/norm

def fixed_points_2comp(lam, gam=GAM_M):
    pts = {'O': (0,0), 'B_+': (1,0), 'B_-': (-1,0)}
    if lam**2 < 6:
        pts['C'] = (lam/np.sqrt(6), np.sqrt(1-lam**2/6))
    if lam**2 > 3*gam:
        pts['D'] = (np.sqrt(1.5)*gam/lam, np.sqrt(3*gam*(2-gam)/2)/lam)
    return pts

np.random.seed(0)
ICS = []
for _ in range(8):
    r0 = np.random.uniform(0.15, 0.85)
    th0 = np.random.uniform(0, np.pi)
    ICS.append((r0*np.cos(th0), r0*np.sin(th0)))
U0 = 0.2

def plot_panel(ax, lam, three_comp, title):
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', lw=0.7, zorder=1)

    for (x0, y0) in ICS:
        if not three_comp:
            sol = solve_ivp(rhs_2comp, [0, 15], [x0, y0], args=(lam,), method='RK45',
                             dense_output=True, rtol=1e-8, atol=1e-10)
            ts = np.linspace(0, 15, 400)
            xy = sol.sol(ts)
            ax.plot(xy[0], xy[1], color='tab:blue', lw=1.1, alpha=0.85, zorder=2)

            phi = phi_of_h(H_DISCRETE)
            x, y, w = x0, y0, np.sqrt(max(1-x0**2-y0**2, 0))
            xs, ys = [x], [y]
            for _ in range(1200):
                xn, yn, wn = step_2comp_final(x, y, w, phi, lam)
                if np.hypot(xn-x, yn-y) < 1e-4:
                    break
                x, y, w = xn, yn, wn
                xs.append(x); ys.append(y)
            ax.plot(xs, ys, 'o', color='tab:orange', markersize=2, alpha=0.7, zorder=3, markeredgewidth=0)
        else:
            v0 = np.sqrt(max(1-x0**2-y0**2-U0**2, 0))
            sol = solve_ivp(rhs_3comp, [0, 15], [x0, y0, U0], args=(lam,), method='RK45',
                             dense_output=True, rtol=1e-8, atol=1e-10)
            ts = np.linspace(0, 15, 400)
            xyu = sol.sol(ts)
            ax.plot(xyu[0], xyu[1], color='tab:blue', lw=1.1, alpha=0.85, zorder=2)

            phi = phi_of_h(H_DISCRETE)
            x, y, u, v = x0, y0, U0, v0
            xs, ys = [x], [y]
            for _ in range(1200):
                xn, yn, un, vn = step_3comp_final(x, y, u, v, phi, lam)
                if np.hypot(xn-x, yn-y) < 1e-4:
                    break
                x, y, u, v = xn, yn, un, vn
                xs.append(x); ys.append(y)
            ax.plot(xs, ys, 'o', color='tab:orange', markersize=2, alpha=0.7, zorder=3, markeredgewidth=0)

        ax.plot(x0, y0, 'o', color='k', markersize=2.5, zorder=4)

    pts = fixed_points_2comp(lam)
    markers = {'O':'o', 'B_+':'s', 'B_-':'s', 'C':'*', 'D':'*'}
    colors = {'O':'black', 'B_+':'dimgray', 'B_-':'dimgray', 'C':'tab:red', 'D':'tab:green'}
    for name, (px, py) in pts.items():
        ms = 13 if name in ('C','D') else 7
        ax.plot(px, py, markers[name], color=colors[name], markersize=ms,
                 markeredgecolor='k', markeredgewidth=0.6, zorder=5)
        ax.annotate(f'${name}$', (px,py), textcoords="offset points", xytext=(6,6), fontsize=9)

    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.1,1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title(title, fontsize=11)

fig, axes = plt.subplots(2, 2, figsize=(11,6.8))

plot_panel(axes[0,0], lam=1.0, three_comp=False,
           title=r'Two-component, $\lambda=1<\sqrt{3}\ (\to C)$')
plot_panel(axes[0,1], lam=3.0, three_comp=False,
           title=r'Two-component, $\lambda=3>\sqrt{3}\ (\to D)$')
plot_panel(axes[1,0], lam=1.0, three_comp=True,
           title=r'Three-component (proj.\ on $x,y$), $\lambda=1\ (\to C_4)$')
plot_panel(axes[1,1], lam=3.0, three_comp=True,
           title=r'Three-component (proj.\ on $x,y$), $\lambda=3\ (\to D_3)$')

from matplotlib.lines import Line2D
legend_elems = [Line2D([0],[0], color='tab:blue', lw=1.5, label='RK45  '),
                Line2D([0],[0], color='tab:orange', lw=0, marker='o', markersize=5, label=f'NSFD, $h={H_DISCRETE}$ ')]
fig.legend(handles=legend_elems, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=10)

plt.subplots_adjust(hspace=0.28, wspace=0.15)
plt.tight_layout(rect=[0,0.035,1,1])
os.makedirs('figs', exist_ok=True)  # so a fresh checkout doesn't crash on savefig
plt.savefig('figs/phase_portrait.pdf')
plt.savefig('figs/phase_portrait.png', dpi=150)
print('saved')
