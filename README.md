# Code accompanying "A Nonstandard Finite Difference Scheme for the Copeland--Liddle--Wands Cosmological Dynamical System"

Tested with Python 3.13, package versions pinned in `requirements.txt`
(`pip install -r requirements.txt`). Every computation is either exact
symbolic algebra (SymPy) or standard double-precision floating-point
arithmetic (NumPy/SciPy); results are not sensitive to minor version
changes in these packages.

## Layout

- `figures/` -- one self-contained script per paper figure. Each writes
  `figs/<name>.pdf` and `.png` relative to its own working directory
  (create a `figs/` subfolder, or one is created automatically by
  matplotlib's `savefig` only if the parent exists -- run
  `mkdir -p figs` first in each script's directory, or run from this
  repository's `figures/` folder after creating `figures/figs/`).
  - `figure_1_pF2_scan.py` -- Figure 1, the p·F2 sign scan.
  - `figure_2_phase_portraits.py` -- Figure 2, the four phase-portrait
    panels (two-component and three-component, both λ regimes).
  - `figure_3_convergence_comparison.py` -- Figure 3, Euler/RK4/NSFD
    error vs. step size at (λ,γ)=(3,1).
- `verification/` -- independent SymPy/NumPy re-derivations of every
  closed-form proposition and theorem in the paper, plus the four
  additional numerical checks requested in the second review round
  (reference-solution accuracy, step-refinement, maximum-in-time error,
  behaviour near the denominator restrictions).

## Verification scripts -> paper claims

- `nsfd_core.py` -- reference implementation of both maps (two- and
  three-component), used as the common base for the checks below.
- `verify_propositions.py` -- general-purpose symbolic cross-check
  harness for the closed-form propositions.
- `verify_2comp_D_jury.py`, `verify_2comp_stability.py` -- Theorems
  "Elementary stability at C/D" (two-component model).
- `verify_2comp_completeness_quartic.py` -- Proposition "Completeness
  of the fixed-point set" (two-component model), including the
  degree-4 polynomial in R and the y=0 sub-case closed-form identities
  added in this revision round.
- `verify_2comp_consistency.py` -- Rule I (local-truncation-error
  consistency check).
- `derive_a2b2_3comp_analytic.py` -- Proposition "Center manifold at
  the threshold" (two- and three-component models).
- `verify_3comp_stability.py` -- stability at the three-component
  fixed points, including $O_r,O_m,B_\pm$ (the $O_r$ eigenvalues
  corrected in this revision round, Section 4's Eq. for $O_r$).
- `derive_D3_block.py`, `derive_D3_jury.py`, `prove_D3_jury.py`,
  `verify_D3_tangential_text.py` (reads `D3block_TrC2.pkl`, produced
  by `derive_D3_block.py`) -- the new Proposition "Tangential stability
  at $D_3$" added in this revision round (reviewer point 2): builds the
  projected $(x,y,v)$ Jacobian block at $D_3$, extracts the Jury
  quantities as polynomials in $s=\lambda^2-3$, confirms every
  coefficient is manifestly positive, and cross-checks the exact
  closed-form expressions transcribed into the paper against the
  direct symbolic computation (all three differences simplify to
  exactly 0). `prove_D3_jury.py` only expands and prints Conditions 1-2
  (N1, N2) for a by-eye sign check; `verify_3comp_D3_jury.py` is the one
  that actually asserts pass/fail, independently and numerically
  (mpmath, 50-digit precision, not reusing `derive_D3_block.py`'s
  symbolic construction), for all three conditions including N3, the
  one `prove_D3_jury.py` does not cover.
- `verify_point5_checks.py` -- the four additional checks requested in
  the second review round (reviewer point 5): (i) reference-solution
  accuracy (RK45 rtol=1e-8 vs. rtol=1e-12), (ii) step-refinement
  (empirical convergence order of NSFD and RK4), (iii) maximum-in-time
  error (not only at the reported endpoint $N=10$), and (iv) behaviour
  near the denominator restrictions $\Delta_n,P_n$ on both sides of the
  threshold $\lambda_A(\gamma)$ of Proposition "Well-posedness,
  positivity and invariance of the physical state space". Run with
  `python3 verify_point5_checks.py`; it prints all four results used
  in Section 5 of the paper.

Run order where one script depends on another's output:
`derive_D3_block.py` must run before `derive_D3_jury.py` or
`verify_D3_tangential_text.py` (it writes `D3block_TrC2.pkl`); a
pre-computed copy of that file is already included so the later
scripts can be run directly without re-running the first.
