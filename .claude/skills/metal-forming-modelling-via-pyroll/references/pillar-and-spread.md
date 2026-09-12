# Pillar model and Domanti spreading

Two repos, read via their `docs/docs.tex` and (for Domanti) source this
session - both concern how a profile's *width* evolves through a roll
pass, at two different levels of resolution.

## `pyroll-pillar-model` — base package for width-wise discretization

Not itself a spread model — a **base package** other plugins build on by
dividing the profile cross-section into `n` collinear "pillars" along the
width direction, each an independent column tracked through the pass.
Most material-flow, groove-filling, and spread plugins in this ecosystem
depend on it for a common interface (Domanti spreading, above, is a
separate, non-pillar-based approach — the two are alternatives, not
layers on top of each other).

**Discretization** (two schemes, `docs.tex` §"Discretization of Profile
Cross-Sections"):
- **Equidistant pillars**: centers `z_i = i * dz`, `dz = w/(n - 1/2)`
  (`w` = max profile width). Simple, but positions can drift during
  deformation, so don't assume they stay equidistant downstream.
- **Uniform pillars**: centers solved so each pillar carries equal
  cross-sectional area (`A_i = A_{i+1}`) summing to the half-width.
  Boundaries `Z_j` sit halfway between consecutive centers (innermost
  boundary `Z_0 = z_0 = 0`); pillar width `w_i = Z_{i+1} - Z_i`.

**Behavior in a roll pass disk element**: only pillars whose height
exceeds the local roll-gap height are "in contact" and actually deform;
others just shift outward to make room. Each pillar's new width comes
from its own spread coefficient: `w_i^1 = beta_i * w_i^0` (`beta_i`
defaults to 1 — actual spread values are supplied by another plugin, this
base package doesn't predict spread itself). New boundaries/positions
follow cumulatively from the updated widths.

**Elongation correction** (the reason this is more than bookkeeping):
treating each pillar's elongation independently violates overall volume
conservation, since pillars aren't mechanically independent in reality.
The correction couples them via a **mean elongation** `lambda_m` -
Gorecki's volume-preserving formula (identified by Wusatowski's 1957
comparison of seven candidate formulas against rolling trials as the most
accurate, since it's the one *derived* to satisfy volume preservation
rather than fitted):

    lambda_m = sum(A_i) / sum(A_i / lambda_i)

applied via a fix-point iteration (first pass: elongation coefficients
=1, evaluating volume preservation globally; each subsequent pass updates
correction coefficients `lambda_{a,i}` against the previous pass's
values, divided by disk-element count, with a relaxation factor for
stability). Enabled by default (`ELONGATION_CORRECTION` in the plugin's
`CONFIG`).

**Key hooks**:
- `Profile.pillars` / `pillar_boundaries` / `pillar_heights` /
  `pillar_widths` / `pillar_sections` — the discretization itself
  (heights via cross-section-polygon intersection at each `z_i`; sections
  via clipping).
- `RollPass.DiskElement.pillars_in_contact` (bool array), `pillar_spreads`
  (`beta_i`, default 1), `pillar_draughts` (`gamma_i`, default
  `h_i^1/h_i^0`), `pillar_elongations` (`lambda_i`, default
  `(beta_i*gamma_i)^-1`).
- To add a new per-pillar property: define `Profile.pillar_<name>s`
  (plural, matching `Profile.rings`' length convention) or the
  `RollPass.DiskElement` equivalent.

A spread-prediction plugin built on this base should implement
`RollPass.DiskElement.pillar_spreads` with real physics instead of the
default-1.0 fallback.

## `pyroll-domanti-spreading` — differential/ODE spread model

A **different, non-pillar** approach: predicts the profile *width as a
continuous function of position along the roll gap* (`x`), not just the
entry/exit values or a per-pillar discretization. Implemented in
`DomantiAnalysis` (`pyroll/domanti_analysis.py`), attached via the
`RollPass.domanti_analysis` hook.

**Mechanism** (an ODE built from four "hill functions" of the local
normalized relative draught):

    normed_relative_draught(xi) = 1 - rel_draught * (2*xi - xi^2)

with `xi` the position normalized along the roll gap. Four integrals of
this quantity (`log(d)/d`, `d*log(d)`, `1/d`, `d`) feed a "shape
integral" constant `B` (closed-form combination of their values over
`[0,1]`, also involving the *relative back tension* — back tension over a
mean flow stress computed as a `(in + 2*out)/3`-weighted average) and a
per-position ODE right-hand side combining the same four integrals
evaluated up to the current position. Solved with `scipy.solve_ivp` over
the roll gap's actual disk-element coordinates, giving a raw normalized
width distribution.

**Scaling step**: the raw ODE solution is dimensionless and only
qualitatively shaped correctly; it's rescaled so its endpoint exactly
matches the profile's *actually known* entry and exit widths (from
whatever spread hook is otherwise in effect) — i.e., this model predicts
the *shape* of the width evolution along `x`, not the absolute magnitude
of overall spread, which still comes from elsewhere. Useful when you need
the intra-gap width profile (e.g., for a groove-filling or edge-defect
check at some specific down-gap position), not just the final width.

Depends on: `roll_pass.rel_draught`, `back_tension`, `in_profile`/`out_profile.flow_stress`,
`roll.contact_length`, and disk-element `in_profile`/`out_profile.x` and
`normed_position` (the last presumably from the pillar model or a similar
per-disk-element position hook — check `RollPass.DiskElement.Profile.normed_position`
if extending this).

**Source note**: this repo's own `docs/docs.tex` and `README.md` were
still the unedited plugin template when read this session (no author
content beyond the one-line PyPI description, "local spread analysis of
Domanti") — everything above came from reading `domanti_analysis.py`
directly. If you need the underlying Domanti paper's own equation
numbers/notation, they aren't transcribed anywhere in this repo yet;
you'd be reconstructing from the code same as this session did.
