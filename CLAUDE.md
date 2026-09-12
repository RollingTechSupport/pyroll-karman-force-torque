# Flat-rolling slab theory and material-flow models — domain reference

This file is durable domain knowledge, not just repo-specific instructions.
It's a deliberately thorough writeup of the classical and extended slab
theories for flat (strip/foil) rolling covered while building this
plugin — including the specific equations, numeric worked-example
checkpoints, and interpretive judgment calls pulled from the primary
sources (several read directly, page by page, via Google Drive/PDF this
session) — so a future session, here or elsewhere, has the physics and
its provenance without re-deriving it or re-reading every paper cold.

**On scope/persistence**: this is committed to git, so it reaches any
session that clones this repo (or the sibling `notebooks-foil-rolling`
repo, which points here) — not a magic memory reaching literally every
Claude session everywhere regardless of repo. That's the strongest
persistence mechanism actually available in this environment (containers
are ephemeral; git history is not).

The plugin in this repo (`pyroll/flat_rolling_slab_model/`, package
`pyroll-flat-rolling-slab-model`) is a concrete, tested implementation of
most of what's below. Treat this file as the theory-plus-provenance layer;
the plugin's own module docstrings and `docs/docs.tex` (compiled to
`docs/docs.pdf`) are the implementation and derivation detail, including
the full LaTeX math this file only sketches in plain text.

## 1. The classical elementary slab theory (von Kármán / Siebel / Orowan)

Core idealization shared by every "elementary" theory below: cut the roll
gap into thin vertical strip elements (slabs) perpendicular to the rolling
direction, and write a 1D horizontal force balance on each element as it
moves through the gap, assuming:

- Plane strain (no spread) — valid for wide strip.
- Vertical stress `sigma_y` (normal pressure `p_N`) uniform through the
  element's thickness at each `x` (von Kármán's own simplification —
  Orowan's model and the layer models below each relax this, in different
  ways).
- Rigid-plastic, or (extended) elastic-plastic, obeying a yield criterion
  linking `sigma_x` and `sigma_y`.
- Friction acts tangentially at the roll/strip interface only (an
  inter-*layer* friction law is a layer-model addition, §5).

**Von Kármán's equilibrium equation** (1925, *Beitrag zur Theorie des
Walzvorganges*, citing Siebel):

    d(sigma_x * h)/dx = 2*p_N*tan(alpha) -/+ 2*tau_R

`h(x)` local gap height, `alpha(x)` roll surface angle, `tau_R`
interfacial shear stress; the `tau_R` sign flips at the neutral point
(friction opposes relative sliding — material slower than the roll on
entry, faster on exit). A first-order linear ODE in `sigma_x` once a yield
criterion and friction law close it; closed-form only for the simplest
geometry/friction choices, solved numerically (shooting/integration) in
general — this is the one equation nearly every model below reduces to,
differing only in how `p_N`/`sigma_y` relates to `sigma_x` (the closure)
and how `tau_R` relates to `p_N` (the friction law).

**Yield criterion** (Mises, plane strain): `sigma_x - sigma_y = ± 2*k_f/sqrt(3)`
(sign flips at the neutral point too). `k_f` is flow stress
(uniaxial yield stress), itself `k_f = k_f(phi, phi_dot, T)` — a material
property from a *separate* flow-stress model, not part of rolling theory
proper. Equivalent Tresca-flavoured conventions (a bare `k_f/sqrt(3)`
offset without the Mises `2/sqrt(3)` factor) show up in some older
sources (Orowan's own, per Overhagen's reconstruction, §1a below) and
need explicit reconciliation when porting them.

**The neutral point** (neutral plane): where strip velocity equals roll
surface velocity, so friction reverses. Never prescribed in closed form —
found by integrating from the entry point (known back-tension boundary
condition) and from the exit point (known front-tension boundary
condition) and searching for the neutral-point position where both
solutions agree on the exit-tension residual (a shooting/root-finding
problem; every solver in this repo's plugin uses `scipy.optimize.brentq`
on a bracketed residual for exactly this).

### 1a. Orowan's (1943) refinement and its reconstruction

Orowan's *The calculation of roll pressure in hot and cold flat rolling*
(Proc. IMechE 150, 1943) drops the "`sigma_y` uniform through thickness"
simplification: genuinely curved (circular-arc) coordinate slices normal
to the roll surface, with a polar-angle coordinate system on each arc,
integrated to give an inhomogeneity function `omega_O(alpha, a_O)`
correcting the `sigma_x`/`sigma_y` closure — the extra rigor matters once
contact is short and/or friction/reduction is large relative to
thickness, the same "is through-thickness deformation really uniform"
question the layer models (§5) answer differently (by discretizing
instead of correcting a closed-form closure).

The original 1943 paper wasn't directly available; this repo's
`orowan_solver.py` reconstructs it from Overhagen's 2018 dissertation
(TU Duisburg-Essen), §4.4.1.2, Gl. (4.4/1), (4.4/6), (4.4/7) — and three
genuine interpretive ambiguities in that secondary source had to be
resolved by cross-checking against Overhagen's own published reference
values, worth recording since they're easy to get subtly wrong porting
from a dissertation rather than the original paper:

- **Gl. (4.4/7)'s integration upper limit**: printed as `a_O`, but a
  later restatement of the same function (Gl. 4.4/36) uses upper limit
  `alpha` instead; integrating to `a_O` is ill-defined in general (the
  integrand's square root goes negative once `a_O/alpha > 1`). Only the
  `alpha`-limit choice reproduces Overhagen's own two published reference
  values (Abb. 4.4/4): `omega_O -> 1` as `a_O -> 0`, `omega_O -> pi/4`
  as `a_O -> 1, alpha -> 0`. **Closed form actually used** (avoiding a
  numerical integration on every ODE right-hand-side evaluation), derived
  via the Fourier-Bessel cosine integral
  `integral_0^1 sqrt(1-t^2)*cos(x*t) dt = pi*J_1(x)/(2*x)` substituted at
  `a_O=1` (full sticking):

      omega_O(alpha, 1) = pi * J_1(alpha) / (2*sin(alpha))

  (`J_1` order-1 Bessel function of the first kind; `-> pi/4` as
  `alpha -> 0`, matches the reference value.) `omega_O` varies only
  weakly with its arguments (`1.0` down to `~0.785`), which is also what
  makes blending it smoothly against the Bay & Wanheim friction-transition
  weight (below) a reasonable pragmatic hybrid rather than a full
  rederivation of Orowan's own sliding-friction branch (his Gl. 4.4/5,
  parametrized by a friction ratio `a_O = 2*mu*sigma_N/k_f in [0,1]`).
- **Flow-stress convention**: Orowan's Gl. (4.4/5)/(4.4/6) use a bare
  `k_f/sqrt(3)` offset (Tresca-flavoured), not this plugin's Mises
  convention (`sigma_x - sigma_N = ∓2*k_f/sqrt(3)`); reconciled by
  doubling `k_f` throughout Gl. (4.4/6) so it reduces to the ordinary
  Mises relation in the frictionless limit and stays continuous with the
  elastic-zone yield-onset criterion.
- **Entry/exit sign of the `(1/alpha - 1/tan(alpha))` geometric term**:
  printed as "∓" without enough context to fix which sign belongs to
  which zone; resolved by using `abs(alpha)` throughout and letting the
  explicit zone sign carry the distinction instead (the term is small
  relative to `omega_O` for typical rolling angles anyway).

Treat any Orowan port from a secondary source as a documented
reconstruction, not a byte-for-byte transcription — validate against
whatever reference values the secondary source itself publishes before
trusting it beyond qualitative comparisons.

### 1b. Ekelund's (1933) redundant-shear correction

A different, complementary refinement from Orowan's: an empirical
"redundant work"/shear-strain multiplier for the extra (non-ideal,
non-uniform) shear deformation actually observed beyond idealized
homogeneous compression, applied multiplicatively on `k_f` or on computed
force, parametrized by contact-length/thickness geometry. Complementary
to, not a replacement for, Orowan's or the layer models' own
through-thickness treatments — noted in this repo as a documented,
deliberately out-of-scope refinement (like Troost's extension, §7).

## 2. Friction laws

- **Plain Coulomb**: `tau_R = mu*p_N`. Breaks down (unbounded "friction
  hill") once integrated pressure would require shear traction exceeding
  the material's shear yield strength — whenever contact length is long
  relative to thickness and/or friction is high (thick-slab/high-friction
  regime).
- **Sticking**: `tau_R = m*k_f/sqrt(3)` (shear-yield-limited, sticking
  coefficient `m in [0,1]`, `m=1` full sticking) — caps shear traction at
  what the material can transmit.
- **Bay & Wanheim (1976) mixed law** (*Real area of contact and friction
  stresses at high pressure sliding contact*, Wear 38): a smooth blend
  between Coulomb and sticking (physically: real contact area saturates
  as normal pressure rises), switching once the Coulomb-implied shear
  would exceed the sticking limit. This is the modern standard choice —
  removes plain Coulomb's friction-hill singularity without sticking-only's
  underestimate at low pressure/thick gauge. Given a sticking coefficient
  `m`, the equivalent single-regime Coulomb coefficient (for callers that
  want to configure friction with one number):

      mu_eq = m / (1 + pi/2 + arccos(m) + sqrt(1-m^2))

  This plugin's own smoothing of the hard Coulomb/sticking switch uses an
  `arctan`-based transition weight `Phi` (0 at low pressure, 1 once
  sticking-limited) around the critical pressure
  `p_crit = m*k_f/(mu*sqrt(3))`, with a tunable smoothness parameter — see
  `_mixed_friction` in `_elastic_plastic_common.py`/`layer_solver.py` for
  the exact closed form actually implemented (`docs/docs.tex`'s
  `eq:mixed-friction`).
- Friction coefficients can depend on rolling speed themselves (Troost,
  §7) — commonly *decreasing* with speed in lubricated cold rolling (less
  time for lubricant-film breakdown/asperity contact); M. D. Stone's and
  O. Pawelski's measurements are cited (via Troost) as the most reliable
  sources of this speed dependence.

## 3. Roll flattening (Hitchcock and beyond)

Rigid-roll (surface stays circular at nominal radius) breaks down once
pressure is high and/or strip is thin relative to the roll (cold rolling
of thin gauge/foil is the classic case). Two levels of correction:

- **Hitchcock's (1935) formula**: corrects to an *effective, still-
  circular* working radius, `R' = R*(1 + C*P/reduction)` for a constant
  `C` from the roll's elastic properties — cheap, closed-form, still a
  circular arc (just bigger). Good to moderate `L_d/H_m`; breaks down once
  flattening is severe (Troost's own paper independently notes needing a
  refined two-branch/power-law variant at that point, not just a bigger
  single-radius correction).
- **Full elastic contour models** (Fleck/Johnson/Sutcliffe-type, e.g. this
  repo's foil model after Mauk & Overhagen 2013): drop "stays circular"
  entirely — see §6 for the full mechanism. Necessary once contact length
  is comparable to or larger than thickness, where the true flattened
  shape can become genuinely non-circular (even locally concave) under a
  strongly peaked pressure distribution.

**Dispatch criterion** used in practice (this repo's plugin uses exactly
this, with two tunable thresholds): contact length `L_d ~ sqrt(R*reduction)`
(classical, un-flattened estimate) over mean thickness `H_m = (h0+h1)/2`.

- `L_d/H_m` small (default `<1.0`): thick slab, through-thickness
  inhomogeneity dominates → layer/slab-splitting models (§5).
- `L_d/H_m` moderate: ordinary single-homogeneous-layer elastic-plastic
  slab theory (§1/§1a + §4) is appropriate.
- `L_d/H_m` large (default `>10.0`): thin foil regime, elastic roll
  flattening dominates → full elastic-contour foil models (§6).

Roll flattening itself is typically kept as a *separate* concern from the
slab-mechanics solver: a solver reads `Roll.working_radius` once (however
that hook is implemented — Hitchcock or otherwise) rather than deriving
or iterating on flattening itself, *except* the full elastic-contour foil
model, which necessarily computes its own roll-gap shape since that's the
whole point of dropping the circular-arc assumption.

## 4. Elastic-plastic extensions (entry/exit zones)

Purely rigid-plastic (whole contact plastic from first touch) is itself
an idealization. A fuller model treats the strip as elastic before
yielding: Hooke's-law elastic entry and exit zones (friction already
acting, since contact starts before yield) bound the plastic zone on both
sides — elastic/plastic transition found via a Mises-yield-crossing event
during integration, elastic *recovery* (springback) at exit found via a
zero-pressure/separation event. Matters most for small reductions
relative to elastic strain (light cold-rolling passes); changes effective
contact length and can be a significant fraction of total contact length
at very light reductions. This is what distinguishes a "modern"
elastic-plastic slab solver from a textbook rigid-plastic one, and it's
genuinely non-trivial to get the event-detection right: a subtle,
previously-encountered bug class was a sign term applied twice at a
zone-sign switch point, silently squaring a `+1/-1` factor to always
`+1` — worth double-checking explicitly (e.g. via a git-HEAD numeric
diff, not just "tests still pass") after refactoring any zone-sign-carrying
right-hand-side function.

## 5. Layer / slab-splitting models for thick strip

Once a slab is thick enough that "homogeneous cross-section" breaks down
(low `L_d/H_m`), through-thickness position matters: surface material
(near roll contact, feeling friction directly) plastifies before core
material (only feeling weaker neighboring-layer friction) — the
"Schmiedekreuz" (forging-cross) effect familiar from thick-plate rolling
and forging. Layer/slab-splitting models discretize thickness into `N`
horizontal layers, each with its own stress/temperature/strain state,
this repo's implementation (porting Weiner's 2021 TU Bergakademie
Freiberg thesis, via two Wolfram Language reference files —
`RollingHomogeneousElasticPlastic.wl`, the `N=1` special case with a
slightly different constitutive formulation for the elastic/plastic
transition, and `RollingLayerModelElasticPlastic.wl`, the general `N`
case) couples layers via:

- **inter-layer friction**: the same Bay & Wanheim mixed law, applied
  between adjacent layers (not just at the roll interface) — a
  `_relative_sliding_direction` regularized-sign function (smooth
  `arctan`-based, avoiding a hard sign discontinuity at zero relative
  velocity) feeds which way each inter-layer friction force points.
- **shared vertical stress**: `sigma_y` derived from a *thickness-weighted
  Mises mean* flow stress over all `N` layers (elastic and plastic alike,
  each at its own current height/temperature) once at least one layer has
  plastified — not just a mean over the plastic subset. (A real bug class
  here: for `N=1` this mean trivially coincides with "active only" since
  `active` is always empty or the full single-layer set, which is exactly
  why a wrong "active-subset-only" mean can silently pass for a long time
  and only shows up once some but not all of several layers have
  plastified — a >1-layer-specific regression test is worth keeping
  precisely because of this.)
- **sequential plastification**: each layer's own plastification
  (entry-side) and unloading (exit-side) boundary found via its own
  shooting/root-find loop (mirroring `EntryKinematicBoundaryIterate` /
  `ExitKinematicBoundaryIterate` in Weiner's `.wl` source) layered on top
  of the overall neutral-point search — layers genuinely diverge (and
  yield out of order) even under *uniform* incoming conditions, since only
  outer layers feel direct roll-contact friction.
- **thermal coupling** (optional but implemented in this repo's layer
  model, not in the single-homogeneous-layer or Orowan solvers): per-layer
  conductive exchange with neighbors and the roll (boundary condition at
  the two outer faces), deformation heat generation, and *friction* heat
  generation split between adjacent layers by relative thermal effusivity
  `BR[i]/(BR[i-1]+BR[i])` where `BR = sqrt(thermal_conductivity *
  heat_capacity * density)` — solved *simultaneously* with the stress ODEs
  per section (not sequentially), since flow stress depends on
  temperature and temperature depends on deformation/friction work.
- **initial neutral-point guess**: Osborn's classical closed-form estimate
  (a function of contact geometry, friction, and front/back tension
  ratios to a representative `k_f`) is used to warm-start the
  root-finding search rather than scanning blind — worth reusing whenever
  porting/re-deriving a similar layer or slab model, since a good initial
  bracket matters far more for convergence robustness here than for the
  single-homogeneous-layer case (more coupled unknowns, more places for a
  blind wide scan to lock onto a spurious root).

`N=1` recovers the ordinary single-homogeneous-layer elastic-plastic model
as a special case (this repo's `LayerRollingSolver` is parametrized
exactly this way, rather than porting two separate classes for the `.wl`
files' two cases).

## 6. Foil / thin-strip rolling (full elastic-contour models)

Opposite extreme from thick slabs: once strip is thin enough relative to
the roll (`L_d/H_m` large), the *roll* — not just the strip — needs
elastic treatment (§3). This repo's `FoilRollingSolver` implements the
Fleck/Johnson/Sutcliffe-type model as described in Mauk & Overhagen
(2013, *Prozessmodell zum Kaltwalzen von Metallfolien mit keramischen
Arbeitswalzen auf Mehrwalzengerüsten*) and its accompanying MATLAB
reference implementation (`lee_sutcliffe/`, in the sibling
`notebooks-foil-rolling` repo — `sutcliffe.m`, `foil_new.m`, `Dij.m` and
the `dgl_*.m`/`event_*.m` ODE right-hand-sides/events).

**Dimensionless formulation** (necessary since the elastic contour and
stress solution are mutually coupled, and this keeps the coupled system
numerically well-scaled): with `E'_R = E_R/(1-nu_R^2)` the roll's
plane-strain modulus and `k_fe = k_f - 0.5*(back_tension+front_tension)`
the tension-adjusted representative flow stress,

    X = x * E'_R / (R * k_fe)          (position)
    T = h * E'_R^2 / (R * k_fe^2)       (height)
    P = p / k_fe                        (normal pressure)
    S = sigma_x / k_fe                  (horizontal stress)

All elastic material properties cancel out of the fully non-dimensional
problem (verifiable by direct substitution) — the only parameter left in
the elastic influence-coefficient matrix is the dimensionless element
width, not any elastic constant.

**Elastic half-space influence matrix**: point-load solutions (Johnson,
*Contact Mechanics*, 1985) superposed over the strip's own pressure
distribution, as a matrix `D_ij - D_1j` (differenced against the first
element so the arbitrary additive constant `D0` in each point-load
solution cancels exactly) built once per grid resolution.

**Five-zone chain** per outer iteration (more zones than the
single-homogeneous-layer model, because pressure-to-flow-stress ratios
get large enough in thin foil that a *sticking* sub-zone genuinely
appears near the neutral point, not just sliding throughout):

1. elastic entry (sliding, Hooke's law + friction already acting),
2. plastic sliding (entry side),
3. sticking (0, 1, or more sub-zones — a "stick/slip switch" event fires
   when the sticking shear *capacity* crosses the sliding shear *demand*;
   this event needs a *signed* direction, not a bare zero-crossing, or the
   stick/slip zones ping-pong in zero-length steps right after the
   segment boundary, since the zero-crossing quantity is ~0 there by
   construction immediately after the event fires),
4. plastic sliding (exit side, forward slip, until the roll-gap's own
   horizontal tangent — not `X=0`, which only marks the *undeformed
   reference parabola's* center and drifts arbitrarily once elastic
   correction is applied),
5. elastic recovery, until separation (`pressure=0`).

**Outer iteration**: solve the 5-zone stress/pressure chain for the
current roll-gap shape → superpose elastic deflections from that pressure
onto the shape → offset so the exit height matches the prescribed exit
gap → relax (partial update, since a naive full update is unstable) →
repeat to convergence. Warm-starting each inner root-find (entry point,
neutral point) from the *previous outer iteration's* value, rather than a
wide blind scan, matters a lot here — the roll-gap height can develop
local non-monotonicities once flattening is significant, and a wide scan
risks locking onto a spurious root far from the physically continued
solution.

**Key qualitative result**: the converged roll-gap shape can be
measurably non-circular even though the deviation from a rigid circular
arc *looks* small on an overlaid plot — real and load-bearing, not
numerical noise. Best visualized as `gap_height - rigid_gap_height` on
its own axis/scale, not overlaid with the two nearly-identical curves.

**Numeric checkpoint** (Mauk & Overhagen's own Abb. 9 worked example,
`k_f=1000 N/mm^2`, `d=25mm` roll, 50% reduction, reproduced qualitatively
in this repo's tests): a stiffer ceramic roll (StarCeram N8000,
`E=310 GPa`) flattens less and needs a *lower* roll force for the same
reduction than a steel roll (`E=210 GPa`) — `521 kN` ceramic vs. `690 kN`
steel, ~24% lower. Any reimplementation should reproduce this direction
(stiffer roll → less flattening → lower force for equal reduction) even
if exact numbers differ with a different friction/tension/geometry setup.

**Known limitation, worth knowing before re-attempting**: Abb. 9's own
*literal* parameters (50% reduction, zero tension — a considerably more
aggressive point than any convergent test case) do not converge with
this solver, even after three separate, individually-validated fixes:
`_gap_minimum` picking the roll-gap's true global minimum by height
value rather than by first-derivative sign change (a shallow local kink
from an incipient sticking sub-zone otherwise fools it); adaptive
relaxation (halving the damping factor whenever a step fails to shrink
the iterate-to-iterate change, breaking limit cycles a fixed factor
cannot); and Anderson(m) mixing (Walker & Ni, 2011) replacing the plain
relaxed blend, extrapolating from several past iterates' residuals via a
small least-squares fit — the standard fix once a Picard iteration's
convergence is dominated by a single near-degenerate direction, which is
exactly what the small-but-load-bearing elastic-flattening correction
against a much larger rigid baseline produces here. Every Anderson step
is re-validated by actually re-running the entry/neutral-point/zone-chain
physics before being accepted (falling back to a shrunk plain-relaxed
step, and ultimately raising rather than silently continuing, if even
that fails) — an unguarded/unvalidated Anderson step was tried first and
it crashed outright (an extrapolated shape landing somewhere the
neutral-point search couldn't bracket at all), which is why validation
before acceptance matters here, not just afterward. Even with all three
fixes, Abb. 9's exact case still fails: even an infinitesimal step off an
already-accepted shape breaks neutral-point bracketing there, indicating
a genuine structural fragility of the zone-chain search at this specific
extreme parameter combination, not merely slow convergence — treated as
a documented limitation rather than pursued further. The per-step
physics re-validation this required is not free: it roughly tripled
`tests/test_foil_solve.py`'s slow-suite wall-clock time (~50 min → ~2h13m),
an accepted cost for a solver that no longer stalls or crashes on harder
(but still convergent) passes. The qualitative direction above remains
validated on a less extreme, convergent pass
(`test_foil_solver_converges_and_reduces_force_with_stiffer_roll`) — that
is what this plugin is actually validated against, not Abb. 9's exact
force values.

## 7. Speed/inertia effects (Troost's plastokinetic extension)

Everything above is *plastostatic* (German: Plastostatik) — it neglects
the strip's own inertia and any velocity dependence of flow stress beyond
whatever a flow-stress model's `strain_rate` argument already captures.
Troost (1967, *Berechnung der Walzkraft und des Drehmomentes beim
Bandwalzen mit höheren Verformungsgeschwindigkeiten*, read in full via
Google Drive this session) extended the classical elementary theory
(explicitly citing von Kármán, Siebel, Orowan as its origin) to a
"plastokinetic" one covering both isothermal and adiabatic deformation at
elevated speed.

**Kinematics** (his Bild 1; the `Uy`/boundary-layer detail in his own
plates was too OCR-degraded this session to transcribe reliably, so only
the parts independently re-derived and cross-checked against his stated
results are given exact form here):

    Ux(x) = U0*h0/h(x)                              (x-velocity, mass continuity)
    dUx/dx = -U0*h0/h(x)^2 * dh/dx                   (differentiating the above)
    Ux_dot = Ux * dUx/dx = -U0^2*h0^2/h(x)^3 * dh/dx  (steady-flow material acceleration)
    phi = ln(h0/h)                                   (equivalent strain)
    phi_dot = dUx/dx                                 (strain rate)

`Uy` (the y-velocity) is linear in `y` across the thickness, satisfying
both incompressible continuity and the no-penetration roll-surface
boundary condition — his own plates state this qualitatively; the exact
coefficient wasn't independently re-derived this session and shouldn't be
trusted from memory alone if it's needed precisely (re-derive from
2D incompressibility, `dUx/dx + dUy/dy = 0`, or re-read the source PDF).

**Inertial force** (d'Alembert's principle applied to the volume element,
his eq. 2, re-derived independently this session from first principles —
matches exactly): with `q_st0 = 0.5*rho*U0^2` the "Staudruck" (stagnation
pressure) at entry velocity `U0`,

    T_x(x) = -2*q_st0*h0*(h0/h(x) - 1)

(magnitude example from the paper: for steel strip, `q_st0 ~ 1 kgf/cm^2`
at `U0=5 m/s`, `~1 kgf/mm^2` at `U0=50 m/s` — scales as `U0^2`, so a 10x
speed increase is a 100x stagnation-pressure increase, though the final
force/flow-stress effect stays small, below.)

**Dynamic flow stress** `k_f*` (his eq. 12, by construction so that every
existing static solution of the equilibrium equation carries over
unmodified once `k_f -> k_f*`):

    k_f*(x) - k_f(x) = -T_x(x)/h(x) = 2*q_st0*[(h0/h(x))^2 - h0/h(x)]

A mean-value approximation across the whole pass for a given reduction
`eps = (h0-h1)/h0` uses a helper function `zeta(eps)` (his eq. 14/15;
`k_f*_mean ~ k_fm + zeta(eps)*q_st0`, with `zeta(0.2) ~ 0.33` in his own
worked numbers) — this repo's re-derivation instead integrates the
pointwise closed form above directly over the parabolic contour
approximation (his eq. 1) rather than reconstructing his specific
`zeta(eps)` transcendental form, since the OCR of his eqs. 14/15 was too
degraded to trust blindly; the two approaches agree to within the
expected order of magnitude at comparable speeds (independently
cross-checked this session, both against his own worked numbers and via
a from-scratch pointwise-vs-mean numeric sweep).

**Also folded into `k_f*`** (not implemented in this repo — see below):
any explicit strain-rate/temperature dependence of flow stress beyond
what a flow-stress model's own `strain_rate` argument captures, via
transformation of isothermal flow curves (Manjoine's uniaxial tension
data for low-carbon steel, `0.1` to `1000/s`) into the walk-through-the-
gap strain-rate history, plus an adiabatic-heating correction (his eqs.
16-28) — found to shift mean flow stress by under 2% relative to isothermal
in his own examples, i.e. secondary to the inertial term itself at the
speeds he could examine.

**Practical magnitude** (his own worked example: `h0=1.8mm`, 20%
reduction, `mu` between 0.01-0.09 under four different assumed
friction-vs-speed relationships, exit velocity swept `1` to `50 m/s`, one
case extended to `100 m/s`): the correction stays small — under ~3 MPa
across the `1`-`50 m/s` range for that geometry (matches this session's
own re-derivation almost exactly: ~2.4-3.9 MPa depending on point-vs-mean
evaluation), rising to ~10-16 MPa at `100 m/s`; **roll force varies by only
about ±10%** across the whole `1`-`50 m/s` range for a first (unworked)
pass, and Troost's own conclusion was that this variation is *dominated
by the assumed friction-coefficient/speed relationship*, not by inertia
itself — inertia's own contribution is secondary in practice at speeds
achievable at the time. For strongly pre-strained passes (later stands of
a fast tandem mill), force is instead expected to *fall* with increasing
speed, since flow-stress strain-rate sensitivity weakens with prior
strain hardening while the friction coefficient keeps falling with speed
— qualitatively confirmed against H. Ford's experimental results, per
Troost's own comparison.

**Reproduced in this repo**: only the inertial term (`dynamic_flow_stress.py`),
as an opt-in `dynamic_flow_stress_correction_enabled` hook (default off)
applied identically at every flow-stress evaluation across all four
solvers — the height ratio `h0/h(x)` the correction needs is recovered
from the equivalent strain already computed at each call site
(`strain = 2/sqrt(3)*ln(h0/h(x))`, so `h0/h(x) = exp(strain*sqrt(3)/2)`),
needing no extra geometry plumbing. Reference velocity used is the roll's
own peripheral speed `2*pi*n*R`, not a self-consistently slip-derived
strip velocity — Troost's own worked example likewise treats speed as an
independent input rather than solving for it jointly with the pressure
distribution, and the correction's magnitude doesn't warrant added rigor
there. The friction-velocity dependence and the strain-rate/temperature
half of `k_f*` are *not* reproduced (already partly covered by whatever
`strain_rate` dependence a flow-stress model itself has) — noted as a
documented, deliberately out-of-scope refinement, like Ekelund's
correction (§1b).

Worth knowing about whenever "high-speed rolling" or "does speed affect
flow stress" comes up, even outside this specific repo — the answer is
"yes, but the effect is small and dominated by friction's own speed
dependence, not inertia" up to speeds well beyond most mills' normal
range.

## 8. Key references (full BibTeX in `docs/refs.bib`)

- von Kármán, Th. (1925) — *Beitrag zur Theorie des Walzvorganges* — the
  foundational equilibrium equation.
- Siebel, E. — cited alongside von Kármán as a co-originator of the
  elementary theory.
- Mises, R. von (1913, 1928) — the yield criterion underlying every
  closure in §1/§1a.
- Orowan, E. (1943) — *The calculation of roll pressure in hot and cold
  flat rolling* — the inhomogeneity-corrected closure (§1a); original not
  directly available this session, reconstructed via Overhagen (2018).
- Ekelund, S. (1933) — redundant-shear/work correction (§1b).
- Bay, N. & Wanheim, T. (1976) — *Real area of contact and friction
  stresses at high pressure sliding contact*, Wear 38 — the mixed
  Coulomb/sticking friction law (§2).
- Hitchcock, J. H. (1935) — *Roll neck bearings* — the flattened-radius
  formula (§3).
- Johnson, K. L. (1985) — *Contact Mechanics* — elastic half-space
  point-load solutions underlying full elastic roll-contour models (§6).
- Mauk, P. J. & Overhagen, C. (2013) — *Prozessmodell zum Kaltwalzen von
  Metallfolien mit keramischen Arbeitswalzen auf Mehrwalzengerüsten* —
  the foil-rolling model (§6); source PDF and the `lee_sutcliffe/`
  MATLAB reference implementation live in the sibling
  `notebooks-foil-rolling` repo.
- Overhagen, C. (2018) — dissertation (TU Duisburg-Essen), *Modelle zum
  Walzen von Flach- und Vollquerschnitten* — secondary source for
  Orowan's model (§1a).
- Weiner, M. (2021) — thesis (TU Bergakademie Freiberg), *Modellierung
  des Walzens dünner Bänder und Folien unter Berücksichtigung
  elastisch-plastischer Werkstoffeigenschaften* — source for the layer
  model (§5), via two Wolfram Language reference files.
- Troost, A. (1967) — *Berechnung der Walzkraft und des Drehmomentes beim
  Bandwalzen mit höheren Verformungsgeschwindigkeiten*, Archiv für das
  Eisenhüttenwesen 38(3) — the plastokinetic (speed/inertia) extension
  (§7); read in full this session (found via Google Drive search after
  the user mentioned adding it there).
- Alexander, J. M. (1972) — *On theory of rolling* — general survey/
  context reference for the elementary theory.

## 9. How this maps onto this repo's plugin (quick index)

- `pyroll/flat_rolling_slab_model/_elastic_plastic_common.py` — shared
  elastic-plastic zone-chain machinery (§4) used by two solvers.
- `karman_mixed_friction_solver.py` — von Kármán + Bay & Wanheim +
  elastic zones, single homogeneous layer (§1, §2, §4).
- `orowan_solver.py` — same, but Orowan's inhomogeneity closure (§1a); not
  wired into automatic dispatch, call directly.
- `layer_solver.py` — `LayerRollingSolver`, N-layer generalization with
  thermal coupling (§5); `layer_count=1` collapses to the homogeneous case.
- `foil_solver.py` — `FoilRollingSolver`, full elastic roll-contour model
  (§3, §6).
- `dynamic_flow_stress.py` — Troost's optional inertial correction (§7),
  off by default, pluggable into all four solvers.
- `condition.py` / `roll_pass.py` — the `L_d/H_m`-based auto-dispatch
  between the three main solvers (§3's dispatch criterion).

Consult `docs/docs.tex`/`docs/docs.pdf` for the full mathematical
derivations, worked examples, and figures — this file is a
faster-to-load, provenance-annotated orientation, not a replacement for it.
