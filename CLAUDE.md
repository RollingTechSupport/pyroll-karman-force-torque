# Flat-rolling slab theory and material-flow models — domain reference

This file is durable domain knowledge, not just repo-specific instructions.
It summarizes the classical and extended slab-theory models for flat
(strip/foil) rolling that this repository implements, so any future session
— here or elsewhere — has the physics background without re-deriving it or
re-reading every paper from scratch. The plugin in this repo
(`pyroll/flat_rolling_slab_model/`, package `pyroll-flat-rolling-slab-model`)
is a concrete, tested implementation of most of what's described below;
treat the summaries here as the theory, and the plugin's own module
docstrings/`docs/docs.tex` as the implementation detail.

## 1. The classical elementary slab theory (von Kármán / Siebel / Orowan)

All "elementary" flat-rolling theories share one core idealization: cut the
roll gap into thin vertical strip elements (slabs) perpendicular to the
rolling direction, and write a 1D horizontal force balance on each element
as it moves through the gap, assuming:

- Plane strain (no spread) — valid for wide strip.
- Vertical stress `sigma_y` (or normal pressure `p_N`) is uniform through
  the element's thickness at each `x` (Karman's own simplification —
  Orowan's model, below, relaxes this).
- The strip is rigid-plastic (or, in extended versions, elastic-plastic)
  and obeys a yield criterion linking `sigma_x` and `sigma_y`.
- Friction acts tangentially at the roll/strip interface only.

**Von Kármán's equilibrium equation** (1925, citing Siebel), the
foundational ODE nearly every flat-rolling model reduces to:

    d(sigma_x * h)/dx = 2*p_N*tan(alpha) -/+ 2*tau_R

with `h(x)` the local gap height, `alpha(x)` the roll surface angle at
position `x`, `tau_R` the interfacial shear stress, and the sign in front of
`tau_R` flipping at the neutral point (friction opposes relative sliding:
material is slower than the roll on the entry side, faster on the exit
side). This is a first-order linear ODE in `sigma_x` once a yield criterion
and friction law close it — closed-form only for simple geometry/friction
assumptions; solved numerically (shooting/integration) in general.

**Yield criterion** (Mises, plane strain): `sigma_x - sigma_y = ± 2*k_f/sqrt(3)`,
where `k_f` is the material's flow stress (uniaxial yield stress) and the
sign again flips at the neutral point. `k_f` itself is a material property
depending on accumulated strain, strain rate, and temperature —
`k_f = k_f(phi, phi_dot, T)`, supplied by a separate flow-stress model, not
by the rolling theory itself.

**The neutral point** (aka neutral plane) is where strip velocity equals
roll surface velocity — friction direction reverses there. It is *not*
prescribed; it's found by solving from both the entry point (with the
known back-tension boundary condition) and the exit point (known
front-tension boundary condition) and searching for the neutral-point
position where both solutions agree (a shooting/root-finding problem,
typically solved with a bracketing method like Brent's — this is how every
solver in this repo's plugin finds it, not by any closed-form formula).
Equivalently: it's the point where the entry- and exit-side solutions'
horizontal stress matches, which is exactly the condition that both the
prescribed back tension and front tension are simultaneously satisfied.

**Orowan's (1943) refinement** drops the "uniform sigma_y through
thickness" simplification, using genuinely curved (circular-arc)
coordinate slices normal to the roll surface and integrating the stress
state across each arc, producing an inhomogeneity function `omega_O` that
corrects the closure between `sigma_x` and `sigma_y` for non-uniform
through-thickness shear (important once the contact is short and/or
friction/reduction is large relative to strip thickness — the exact same
"is the deformation really homogeneous through the thickness" regime that
motivates layer/slab-splitting models, below). `omega_O` depends on
friction state and geometry but only weakly, ranging from `1` (frictionless
limit) to `pi/4 ~ 0.785` (full sticking).

**Ekelund's (1933) correction** is a different, complementary refinement:
an empirical "redundant work"/shear-strain multiplier accounting for the
extra (non-ideal, non-uniform) shear deformation actually observed in
rolling beyond the idealized homogeneous-compression assumption — usually
applied as a multiplicative factor on `k_f` or on the computed force,
parametrized by geometry (contact-length/thickness ratio). Complementary
to, not a replacement for, Orowan's or the layer models' own
through-thickness treatments.

## 2. Friction laws

- **Plain Coulomb**: `tau_R = mu * p_N`. Breaks down (produces an
  unbounded "friction hill") once integrated pressure would require shear
  traction exceeding the material's own shear yield strength — this
  happens whenever contact length is long relative to strip thickness
  and/or friction is high, i.e. exactly the thick-slab/high-friction
  regime.
- **Sticking friction**: `tau_R = m * k_f/sqrt(3)` (shear-yield-limited,
  `m` a "sticking coefficient" in `[0,1]`, `m=1` full sticking) — caps
  shear traction at what the material can actually transmit.
- **Bay & Wanheim (1976) mixed law**: a smooth blend between Coulomb and
  sticking, switching (physically: real contact area saturates as normal
  pressure rises) once the Coulomb-law-implied shear would exceed the
  sticking limit. Given a sticking coefficient `m`, an equivalent Coulomb
  coefficient can be derived via `mu_eq = m / (1 + pi/2 + arccos(m) + sqrt(1-m^2))`
  for use where a model wants a single-regime Coulomb coefficient as input
  but the more physical mixed law internally. This is the standard modern
  choice — it removes the friction-hill singularity of plain Coulomb
  without switching to sticking-only (which underestimates force at low
  pressure/thick gauge).
- Friction coefficients can themselves depend on rolling speed (see
  Troost, below) — commonly *decreasing* with speed for lubricated cold
  rolling (less time for lubricant film breakdown/asperity contact).

## 3. Roll flattening (Hitchcock and beyond)

The rigid-roll assumption (roll surface stays circular, radius = nominal
roll radius) breaks down once roll pressure is high and/or the strip is
thin relative to the roll (cold rolling of thin gauge/foil is the classic
case). Two levels of correction, in increasing order of rigor:

- **Hitchcock's (1935) formula**: an empirically/analytically motivated
  single-number correction to an *effective, still-circular* working
  radius: `R' = R * (1 + C*P/(reduction))` for some constant `C` depending
  on the roll's elastic properties — cheap, closed-form, still assumes a
  circular arc (just a bigger one). Good down to moderate contact-
  length/thickness ratios; breaks down (and Troost's own paper notes a
  refined two-branch/power-law variant is needed) once flattening becomes
  severe.
- **Full elastic contour models** (Fleck/Johnson/Sutcliffe-type, e.g. the
  foil-rolling model in this repo, after Mauk & Overhagen 2013): drop the
  "stays circular" assumption entirely. The roll-gap shape is found by
  superposing elastic half-space (Johnson 1985) point-load influence
  functions over the strip's own normal-pressure distribution and
  iterating (pressure → deflection → new gap shape → new pressure → ...)
  to convergence. Necessary once contact length becomes comparable to or
  larger than strip thickness (`L_d/H_m` large — see dispatch criterion
  below) — the point where a "flattened but still circular" approximation
  is no longer good enough, because the true flattened shape can become
  genuinely non-circular (even locally concave) under a strongly peaked
  pressure distribution.

A widely-used, simple criterion for "is the rigid/lightly-flattened
assumption still good enough" is the ratio of contact length `L_d`
(`~ sqrt(R * reduction)`, the classical un-flattened estimate) to mean
strip thickness `H_m = (h0+h1)/2`:

- `L_d/H_m` small (~<1): thick slab, through-thickness inhomogeneity
  dominates → layer/slab-splitting models.
- `L_d/H_m` moderate: ordinary single-homogeneous-layer elastic-plastic
  slab theory (Kármán/mixed-friction/Orowan-type models) is appropriate.
- `L_d/H_m` large (~>10): thin foil regime, elastic roll flattening
  dominates → full elastic-contour foil models.

(This repo's plugin uses exactly this ratio, with two tunable thresholds,
to auto-dispatch between its three solvers — see `condition.py` and
`roll_pass.py`'s `karman_solution` hook.)

## 4. Elastic-plastic extensions (entry/exit zones)

The purely rigid-plastic assumption (whole contact plastic from first
touch) is itself an idealization. A more complete model treats the strip
as elastic before yielding: Hooke's law entry and exit zones (with
friction already acting, since contact starts before yield) bound the
plastic zone on both sides, with the elastic/plastic transition found via
a Mises-yield-crossing event during integration, and elastic *recovery*
at exit (springback) found via a zero-pressure/separation event. This
matters most in cold rolling with small reductions relative to elastic
strain, and is what distinguishes a "modern" elastic-plastic slab solver
from a textbook rigid-plastic one — it changes the effective contact
length and can be a significant fraction of total contact for very light
reductions.

## 5. Layer / slab-splitting models for thick strip

Once a slab is thick enough that a single "homogeneous cross-section"
assumption breaks down (low `L_d/H_m`), through-thickness position starts
to matter: surface material (near roll contact, feeling friction directly)
plastifies before core material (which only feels neighboring-layer
friction, weaker than roll/strip friction) — the "Schmiedekreuz"
(forging-cross) effect, well known in thick-plate rolling and forging.
Layer/slab-splitting models discretize the thickness into N horizontal
layers, each with its own stress/temperature/strain state, coupled via:

- an inter-layer friction law (same Bay & Wanheim mixed law, applied
  between adjacent layers rather than only at the roll interface),
- a shared vertical stress `sigma_y` (or a Mises-mean flow stress across
  active/plastic layers) enforcing overall equilibrium,
- (optionally) thermal coupling: each layer exchanges heat with its
  neighbors and the roll, generates heat from deformation and friction
  work, so surface and core can develop genuinely different temperatures
  (and hence different `k_f`) through the pass.

`N=1` recovers the ordinary single-homogeneous-layer elastic-plastic
model as a special case (this repo's `LayerRollingSolver` is
parametrized exactly this way). Layers plastify *sequentially* as
reduction proceeds — a genuinely different qualitative behavior from the
single-layer case, and the reason layer models need their own
per-layer shooting/root-finding logic (tracking each layer's own
plastification boundary) on top of the overall neutral-point search.

## 6. Foil / thin-strip rolling (full elastic-contour models)

At the opposite extreme from thick slabs: once strip is thin enough
relative to the roll (`L_d/H_m` large), the *roll* — not just the strip —
needs elastic treatment, per the roll-flattening discussion above. Full
foil-rolling models (Fleck/Johnson/Sutcliffe-type; this repo's
`FoilRollingSolver` after Mauk & Overhagen 2013) combine:

- the same elastic-plastic zone-chain physics as above (elastic entry →
  plastic sliding → sometimes a *sticking* sub-zone near the neutral
  point, since pressure-to-flow-stress ratios get large in thin foil →
  plastic sliding on the exit side → elastic recovery),
- solved in a *dimensionless* form (positions, pressures, stresses scaled
  by the roll's plane-strain modulus, strip flow stress, and roll
  radius), because the elastic contour and the stress solution are
  mutually coupled and this keeps the coupled system numerically
  well-scaled,
- an outer iteration alternating: solve the stress/pressure distribution
  for the current roll-gap shape → superpose elastic half-space
  deflections from that pressure onto the shape → re-solve → repeat to
  convergence (relaxed, since naive full updates tend to be unstable).

A key qualitative result specific to this regime: the converged roll-gap
shape can be measurably non-circular even though the deviation from a
rigid circular arc can *look* small on an overlaid plot — the effect is
real and load-bearing, not numerical noise, and is best visualized as the
deviation `gap_height - rigid_gap_height` on its own axis.

## 7. Speed/inertia effects (Troost's plastokinetic extension)

All of the above is *plastostatic* — it neglects the strip's own inertia
and any velocity dependence of flow stress beyond whatever a flow-stress
model's `strain_rate` argument already captures. Troost (1967) extended
the classical elementary theory to a "plastokinetic" one by adding the
volume element's inertial force (via d'Alembert's principle, `F = m*a`
with `a = U*dU/dx` from steady-flow continuity `U(x)*h(x) = const`) to
the same horizontal equilibrium equation, and folding both the inertial
term and any explicit strain-rate/temperature dependence of `k_f` into a
single "dynamic flow stress" `k_f*`, defined so that **every existing
static solution carries over unmodified once `k_f` is replaced by
`k_f*`** — a very convenient property: the extension is a pure additive
correction to flow stress, not a change to the ODE structure or solution
method.

Closed form (derived from Troost's eqs. 2 and 12, independently
re-verified numerically against his own worked example in this repo's
session history — see git log around the `dynamic_flow_stress.py` module):

    q_st0 = 0.5 * rho * U0^2                          (stagnation pressure, entry velocity U0)
    k_f*(x) - k_f(x) = 2 * q_st0 * [(h0/h(x))^2 - h0/h(x)]

Practically small in magnitude for ordinary cold-rolling speeds (well
under 1% of typical flow stress below ~10 m/s) and only reaching a low
single-digit percentage at speeds far beyond most mills' normal range
(~75+ m/s) — Troost's own conclusion was that friction's velocity
dependence (see above) dominates over inertia itself in practice. Worth
knowing about whenever "high-speed rolling" or "does speed affect flow
stress" comes up, even outside this specific repo.

## 8. Key references (see `docs/refs.bib` for full BibTeX)

- von Kármán, Th. (1925) — *Beitrag zur Theorie des Walzvorganges* — the
  foundational equilibrium equation.
- Siebel, E. — cited alongside von Kármán as a co-originator of the
  elementary theory.
- Orowan, E. (1943) — *The calculation of roll pressure in hot and cold
  flat rolling* — the inhomogeneity-corrected closure.
- Ekelund, S. (1933) — redundant-shear/work correction.
- Bay, N. & Wanheim, T. (1976) — *Real area of contact and friction
  stresses at high pressure sliding contact* — the mixed Coulomb/sticking
  friction law.
- Hitchcock, J. H. (1935) — *Roll neck bearings* — the flattened-radius
  formula.
- Johnson, K. L. (1985) — *Contact Mechanics* — elastic half-space
  point-load solutions underlying full elastic roll-contour models.
- Mauk, P. J. & Overhagen, C. (2013) — *Prozessmodell zum Kaltwalzen von
  Metallfolien mit keramischen Arbeitswalzen* — the foil-rolling model
  this repo's `FoilRollingSolver` implements.
- Overhagen, C. (2018), Weiner, M. (2021) — dissertations reconstructing/
  extending Orowan's and the layer model's equations respectively, used
  as primary sources where the original papers were unavailable or
  ambiguous.
- Troost, A. (1967) — *Berechnung der Walzkraft und des Drehmomentes
  beim Bandwalzen mit höheren Verformungsgeschwindigkeiten* — the
  plastokinetic (speed/inertia) extension.
- Alexander, J. M. (1972) — *On theory of rolling* — general survey/
  reference for the elementary theory's context.

## 9. How this maps onto this repo's plugin (quick index)

- `pyroll/flat_rolling_slab_model/_elastic_plastic_common.py` — shared
  elastic-plastic zone-chain machinery (§4 above) used by two solvers.
- `karman_mixed_friction_solver.py` — von Kármán + Bay & Wanheim +
  elastic zones, single homogeneous layer (§1, §2, §4).
- `orowan_solver.py` — same, but Orowan's inhomogeneity closure (§1).
- `layer_solver.py` — `LayerRollingSolver`, N-layer generalization with
  thermal coupling (§5); `layer_count=1` collapses to the homogeneous
  case.
- `foil_solver.py` — `FoilRollingSolver`, full elastic roll-contour model
  (§3, §6).
- `dynamic_flow_stress.py` — Troost's optional inertial correction (§7),
  off by default, pluggable into all four solvers.
- `condition.py` / `roll_pass.py` — the `L_d/H_m`-based auto-dispatch
  between the three main solvers (§3's dispatch criterion).

Consult `docs/docs.tex` (compiled to `docs/docs.pdf`) for the full
mathematical derivations, worked examples, and figures — this file is
meant as a faster-to-load orientation, not a replacement for it.
