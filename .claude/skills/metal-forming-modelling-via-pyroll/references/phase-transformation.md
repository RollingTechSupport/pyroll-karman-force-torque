# Phase transformation model (`pyroll-phase-transformation-model`)

Read via its `docs/docs.tex` (well-written, equation-level source — trust
this one at full depth, unlike the ecosystem-tools repos). Predicts the
austenite-decomposition microstructure of a rolled-and-cooled steel
profile from the thermal history it experiences through a PyRoll pass
sequence, plus derived hardness and mechanical properties.

## Coupling to PyRoll

Two steps: (1) reconstruct the continuous cooling path `T(t)` a material
point passes through, by collecting the time/temperature of every
disk element of every cooling unit in the sequence; (2) evaluate
competing transformation kinetics along that path and report final phase
fractions, hardness, and mechanical properties as profile hooks. All
material/composition inputs come from profile hooks, so it adapts to
whatever steel grade the profile carries.

## Diffusional transformations (ferrite, pearlite, bainite)

A sigmoidal reaction-rate function captures nucleation-and-growth
kinetics:

    S(X) = integral_0^X dx / [x^(0.4*(1-x)) * (1-x)^(0.4*x)]

(plus a related auxiliary `I(X)` with exponent `2/3` instead of `0.4`,
used elsewhere in the model). Both are evaluated once by quadrature over
`X in [0.001, 0.999]` and cached as cubic splines (with inverses), so
they apply to whole arrays efficiently rather than re-integrating per
call.

The modified Kirkaldy-Venugopalan isothermal transformation time (Li et
al. 1998 parametrization):

    tau_k(X, T) = S(X) * Phi_k(T)
    Phi_k(T) = C_k / [2^(g_k*G) * (Ts_k - T)^n_k * exp(-Q/(R*T))]

`Phi_k` the temperature-dependent "transformation factor", `C_k` a
composition factor (from a per-phase hook), `g_k` grain-size coefficient,
`G` the profile's ASTM grain-size number, `Ts_k - T` undercooling below
that phase's start temperature, `n_k` the diffusion exponent (2 =
boundary diffusion, 3 = volume diffusion), `Q ≈ 115060 J/mol` activation
energy. Per-phase constants: ferrite (`g=0.41, n=3`, active between
`Ae3` and `Bs`), pearlite (`g=0.32, n=3`, `Ae1` to `Bs`), bainite
(`g=0.29, n=2`, `Bs` to `Ms`).

**Continuous cooling** uses Scheil's (1935) additivity rule: incremental
incubation-time consumption accumulates along the cooling path, and the
fraction `X` is reached once `integral(dt/Phi_k(T(t))) >= S(X)` —
implemented as a running cumulative sum while `Tf_k < T(t) < Ts_k`, then
inverting `S` (via the cached spline) to recover the fraction, clamped at
the untransformed/fully-transformed limits outside the spline's range.

## Martensite

Athermal, below `Ms`, via Koistinen-Marburger (1959):

    X_m(T) = 1 - exp(-alpha_m * (Ms - T)),   T < Ms

`alpha_m` from the `martensite_koistinen_marburger_rate` hook.

## Phase competition

The four products compete for available austenite. Each first computes
an isolated/uncoupled fraction and increment, then couples via balance
residuals at each time step — **asymmetric by design** (preserved
deliberately from the underlying Li model, not a bug): ferrite and
pearlite normalize by their *own* saturation level, while bainite and
martensite apply directly against whatever austenite remains after the
other products. Remaining austenite = `1 - sum(all fractions)`.

## Hardness (Maynier 1978 correlations)

Evaluated at the cooling rate the cycle reaches at 700°C (`phi_700`,
converted to K/hour via `L = log10(phi_700 * 3600)`), separate linear
correlations in composition (mass-%) and `L` for martensite (`HV_M`),
bainite (`HV_B`), and a shared ferrite+pearlite correlation (`HV_FP`).
Overall Vickers hardness is the fraction-weighted rule-of-mixtures mean
across the three (retained austenite excluded, remaining fractions
renormalized). **Every quantity is its own independently-overridable
out-profile hook** (`cooling_rate_at_700`, `vickers_hardness_martensite`,
`_bainite`, `_ferrite_perlite`, `vickers_hardness`) — swap one
correlation and everything downstream picks it up automatically.

## Mechanical properties (Krause 2007 correlations)

From overall hardness `HV` and martensite fraction `y_m` (%):

    Rm = 3.04*HV                          (tensile strength, default)
    A50 = 40 - (0.03 - 0.001*y_m)*Rm       (elongation at fracture, %)
    Z = 100 - (0.06 - 0.00024*y_m)*Rm      (reduction of area, %)
    KU = 296 - (0.285 - 0.00098*y_m)*Rm    (ISO-U impact energy, J)

An alternate `Rm` correlation (`3.412*HV - 64.3`) is also documented. The
source's own yield-strength correlation is *deliberately omitted* — its
coefficients give physically impossible values, almost certainly a
typo in the original source; don't reintroduce it without fixing that
first. Again, each output (`tensile_strength`, `elongation_a50`,
`reduction_of_area`, `impact_energy_ku`) is its own overridable hook
reading `vickers_hardness`/`martensite_fraction`.

## When this matters

Anything downstream of hot rolling that cares about final microstructure,
hardness, or strength/ductility — cooling-bed/Stelmor design questions
(see `references/thermal-models.md`), quality prediction, or any
"what hardness will this grade end up at with this cooling schedule"
question.
