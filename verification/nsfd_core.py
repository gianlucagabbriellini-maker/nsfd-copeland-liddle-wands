"""
nsfd_core.py
Author: Gianluca Gabbriellini
Last modified: 2026-11-02

Reference implementation of the NSFD scheme for the Copeland-Liddle-Wands
system, both the two- and three-component versions. Every figure script and
verification script builds on these functions, so there is one definitive
version of the discrete map.

Comments below cite the paper by LaTeX label name rather than equation
number, since the numbering has shifted across revisions and labels have not.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Continuous systems (for reference trajectories / comparison only)
# ---------------------------------------------------------------------------

def rhs_2comp(t, state, lam, gam=1.0):
    """Right-hand side of the continuous two-component system
    (paper's eq:continuous-2comp), used only to generate reference
    trajectories to compare the discrete scheme against, not part of
    the NSFD scheme itself. S here is the Omega_gamma-eliminated form,
    S = (2-gam)*x^2 - gam*y^2 + gam (the two forms of S agree once the
    Friedmann constraint holds, as the paper notes).
    """
    x, y = state
    S = (2 - gam) * x**2 - gam * y**2 + gam
    dx = -3*x + np.sqrt(6)/2*lam*y**2 + 1.5*x*S
    dy = -np.sqrt(6)/2*lam*x*y + 1.5*y*S
    return [dx, dy]


def rhs_3comp(t, state, lam):
    """Right-hand side of the continuous three-component system
    (x, y, u=sqrt(Omega_r)), paper's eq:continuous-3comp, with S as in
    eq:S-3comp: S = x^2 - y^2 + 1 + Omega_r/3, gamma_m=1. Like
    rhs_2comp above, this is only for building a reference trajectory,
    not the discrete scheme.
    """
    x, y, u = state
    S = x**2 - y**2 + 1 + u**2/3
    dx = -3*x + np.sqrt(6)/2*lam*y**2 + 1.5*x*S
    dy = -np.sqrt(6)/2*lam*x*y + 1.5*y*S
    du = 0.5 * u * (3*S - 4)
    return [dx, dy, du]


# ---------------------------------------------------------------------------
# Denominator function (paper's eq:phi-def): phi(h) = (1-e^{-3h})/3
# ---------------------------------------------------------------------------

def phi_of_h(h):
    """Mickens' nonstandard denominator function (eq:phi-def). This is
    the step size that actually appears throughout the scheme below,
    in place of the plain h a standard finite-difference method would
    use.
    """
    return (1 - np.exp(-3*h)) / 3


# ---------------------------------------------------------------------------
# Two-component NSFD map, Section 3.2 (sec:costruzione).
# S is split term by term instead of kept as one block, the coupling
# term uses x_n*y_{n+1} (not x_{n+1}*y_n), and the whole system is then
# solved by direct substitution, with no need for Cramer's rule.
# Paper's eq:y-solve, eq:x-solve, eq:solution-new.
# ---------------------------------------------------------------------------

def step_2comp(x, y, phi, lam, gam=1.0):
    """One NSFD step for the two-component model (eq:solution-new).

    Parameters
    ----------
    x, y : float
        Current state (x_n, y_n).
    phi : float
        phi(h), see phi_of_h().
    lam, gam : float
        Physical parameters lambda, gamma.

    Returns
    -------
    (x_next, y_next) : tuple of float

    Notes
    -----
    Delta_n (eq:y-solve) stays positive, so the step is well-posed,
    for every h, as long as lam <= lambda_A(gam) = sqrt(6)*(2+gam)/2
    (the paper's Proposition on well-posedness/positivity/invariance,
    eq:lambdaA); beyond that threshold it only stays positive up to a
    finite h_crit(lam, gam), see h_crit_2comp() below.
    """
    Kn = 1 + phi * (3*x**2 + 1.5*gam)
    Deltan = 1 + phi * (np.sqrt(6)/2*lam*x + 1.5*gam*(x**2 + y**2))
    Pn = 1 + phi * (3 + 1.5*gam*(x**2 + y**2))
    y_next = Kn * y / Deltan
    x_next = Kn * (x*Deltan + phi*np.sqrt(6)/2*lam*y**2) / (Pn*Deltan)
    return x_next, y_next


def lambda_A(gam=1.0):
    """The threshold lambda_A(gamma) = sqrt(6)*(2+gamma)/2 (eq:lambdaA)
    below which Delta_n stays positive for every step size h. This is
    the exact boundary, not just a sufficient condition, as the paper's
    well-posedness proposition shows.
    """
    return np.sqrt(6) * (2 + gam) / 2


def h_crit_2comp(lam, gam=1.0):
    """Beyond lambda_A, Delta_n only stays positive up to some largest
    step size; this returns that critical h, by solving
    1 + phi*(1.5*gam - sqrt(6)/2*lam) = 0 (same well-posedness
    proposition as lambda_A above). Returns np.inf when lam is already
    within the unconditional range (lam <= lambda_A(gam)).
    """
    if lam <= lambda_A(gam):
        return np.inf
    coeff = np.sqrt(6)/2*lam - 1.5*gam  # > 0 in this regime
    phi_crit = 1 / coeff
    if phi_crit >= 1/3:
        return np.inf
    return -np.log(1 - 3*phi_crit) / 3


def step_2comp_projected(x, y, w, phi, lam, gam=1.0):
    """One full NSFD step for the two-component model: step_2comp above
    for (x,y), together with the matching update of the fluid variable
    w = sqrt(Omega_gamma), followed by projecting all three back onto
    the Friedmann constraint sphere x^2+y^2+w^2=1 (eq:projection-new,
    eq:full-map-2comp).

    step_2comp on its own returns the PRE-projection (x,y), which is
    not actually x_{n+1},y_{n+1} of the real scheme: the projection
    step normally changes them too, since the raw update does not keep
    x^2+y^2+w^2 exactly at 1 (only close to it, for small phi). Call
    this function, not step_2comp directly, to generate an actual
    trajectory of the two-component scheme, as the figure scripts and
    the point-5 numerical checks all do.
    """
    x1, y1 = step_2comp(x, y, phi, lam, gam)
    S = (2 - gam) * x**2 - gam * y**2 + gam
    Omega1 = step_Omega(w**2, S, phi, gamma_i=gam)
    w1 = np.sqrt(max(Omega1, 0.0))
    norm = np.sqrt(x1**2 + y1**2 + w1**2)
    return x1/norm, y1/norm, w1/norm


# ---------------------------------------------------------------------------
# Three-component NSFD map, Section 4.2 (sec:tre-componenti).
# Same term-by-term splitting of S as in the two-component case, now
# applied to S_3comp = x^2 - y^2 + 1 + Omega_r/3, which only has ONE
# sign-risky term (-y^2), unlike the two-component S which has two
# (-x^2 and -y^2). Omega_r and Omega_m are each updated with growth
# explicit and decay implicit (eq:Omegar-def, eq:Omegam-def), which is
# what makes them stay non-negative for every h, with no threshold at
# all needed for that part.
# ---------------------------------------------------------------------------

def step_3comp_xy(x, y, u, phi, lam):
    """(x_{n+1}, y_{n+1}) for the three-component model, before the
    projection step is applied. Paper's eq:solution-3comp and
    eq:PQK-3comp for Delta_n, P_n, K_n.
    """
    Deltan = 1 + phi * (np.sqrt(6)/2*lam*x + 1.5*y**2)
    Pn = 1 + phi * (3 + 1.5*y**2)
    Kn = 1 + phi * (1.5*x**2 + 1.5 + 0.5*u**2)
    y_next = Kn * y / Deltan
    x_next = Kn * (x*Deltan + phi*np.sqrt(6)/2*lam*y**2) / (Pn*Deltan)
    return x_next, y_next


def h_crit_3comp_xy(lam):
    """Critical step for the three-component (x,y) block, unconditional
    only for lam <= sqrt(6) (weaker than the two-component lambda_A(gam),
    because S_3comp has only one sign-risky term; see Section 4.2)."""
    if lam <= np.sqrt(6):
        return np.inf
    phi_crit = 2 / (np.sqrt(6)*lam)
    if phi_crit >= 1/3:
        return np.inf
    return -np.log(1 - 3*phi_crit) / 3


def step_Omega(Omega, S, phi, gamma_i):
    """One step of a fluid density Omega_i, discretizing
    dOmega_i/dN = Omega_i*(3S - 3*gamma_i) as
    Omega_{n+1} = Omega_n*(1 + 3*phi*S) / (1 + 3*gamma_i*phi)
    (eq:Omegar-def / eq:Omegam-def, depending on which fluid gamma_i
    is for). Whenever Omega_n and S both start non-negative, so does
    Omega_{n+1}, for every h and every lambda, with no threshold needed
    (the paper's fluid-positivity proposition for the three-component
    model).
    """
    return Omega * (1 + 3*phi*S) / (1 + 3*gamma_i*phi)


def step_3comp_projected(x, y, u, v, phi, lam):
    """One full NSFD step for the three-component model: the raw
    update above, followed by the projection back onto the (now
    generalized) Friedmann constraint sphere (eq:projection,
    eq:full-map-3comp) that keeps x^2+y^2+Omega_r+Omega_m=1 exact at
    every step.

    u = sqrt(Omega_r), v = sqrt(Omega_m), gamma_r = 4/3, gamma_m = 1.
    """
    x1, y1 = step_3comp_xy(x, y, u, phi, lam)
    S = x**2 - y**2 + 1 + u**2/3
    Omega_r1 = step_Omega(u**2, S, phi, gamma_i=4/3)
    Omega_m1 = step_Omega(v**2, S, phi, gamma_i=1.0)
    u1 = np.sqrt(max(Omega_r1, 0.0))
    v1 = np.sqrt(max(Omega_m1, 0.0))
    norm = np.sqrt(x1**2 + y1**2 + u1**2 + v1**2)
    return x1/norm, y1/norm, u1/norm, v1/norm


# ---------------------------------------------------------------------------
# Continuous fixed points (for validation / plotting)
# ---------------------------------------------------------------------------

def fixed_points_2comp(lam, gam=1.0):
    """Returns a dict of the continuous system's fixed points, matching
    the paper's fixed-point table for the two-component model
    (tab:fixedpoints)."""
    pts = {'O': (0.0, 0.0), 'B_+': (1.0, 0.0), 'B_-': (-1.0, 0.0)}
    if lam**2 < 6:
        pts['C'] = (lam/np.sqrt(6), np.sqrt(1 - lam**2/6))
    if lam**2 > 3*gam:
        pts['D'] = (np.sqrt(1.5)*gam/lam, np.sqrt(3*gam*(2-gam)/2)/lam)
    return pts


if __name__ == '__main__':
    # quick self-test: fixed points should map to themselves for any phi
    print("Self-test: fixed points are exact fixed points of the map\n")
    lam, gam = 3.0, 1.0
    for h in [0.05, 0.3, 1.0]:
        phi = phi_of_h(h)
        pts = fixed_points_2comp(lam, gam)
        for name, (x0, y0) in pts.items():
            x1, y1 = step_2comp(x0, y0, phi, lam, gam)
            err = np.hypot(x1-x0, y1-y0)
            status = "OK" if err < 1e-12 else "FAIL"
            print(f"  h={h:.2f}  {name:4s} = ({x0:+.4f},{y0:.4f})  "
                  f"-> ({x1:+.4f},{y1:.4f})  err={err:.2e}  [{status}]")
            # The printed [FAIL] above is easy to miss when this script runs
            # unattended (e.g. as part of a reproducibility check); an
            # assert makes a broken fixed point a hard failure (non-zero
            # exit code) instead of a line of text nobody reads.
            assert status == "OK", f"{name} is not a fixed point of step_2comp at h={h}!"
    print(f"\nlambda_A(gamma=1) = {lambda_A(1.0):.4f}")
    print(f"h_crit_2comp(lambda=3, gamma=1) = {h_crit_2comp(3.0, 1.0)} "
          f"(should be inf, since 3 < lambda_A)")
    print(f"h_crit_2comp(lambda=4, gamma=1) = {h_crit_2comp(4.0, 1.0):.4f} "
          f"(should be finite, since 4 > lambda_A={lambda_A(1.0):.3f})")
