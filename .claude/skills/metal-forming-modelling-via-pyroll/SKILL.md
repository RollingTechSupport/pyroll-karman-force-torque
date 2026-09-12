---
name: metal-forming-modelling-via-pyroll
description: Reference knowledge for the RollingTechSupport PyRoll plugin ecosystem - what physics each plugin implements, which equations/sources it's based on, and where its hooks and code live. Use this whenever a task touches PyRoll model physics or rolling mechanics in any way, including but not limited to: flat-rolling slab theory (von Karman/Orowan/Ekelund, friction laws, roll flattening, foil rolling), material flow / spread / pillar models, groove filling, phase transformation or microstructure/hardness prediction after rolling, coil/conveyor/laying-head thermal and cooling models, or general "what models/plugins exist across our PyRoll repos" orientation questions - even if the user doesn't name a specific plugin or say "PyRoll" explicitly, just describes a rolling-mill physics problem (spread, temperature, hardness, roll force, cooling curve, microstructure, wire coil, laying head, Stelmor conveyor). Also consult this before writing a NEW PyRoll plugin, since related exploratory work may already exist in pyroll-sandbox.
---

# Metal forming modelling via PyRoll

This skill orients you across the RollingTechSupport PyRoll plugin
ecosystem: which repo implements which physics, what it's based on, and
where to look for the full derivation. It was assembled by surveying 15
repos in one session - **depth varies by section**, and that's stated
plainly below rather than papered over:

- **Deep, equation-level coverage**: flat-rolling slab theory (this repo
  itself already has a full `CLAUDE.md` for it - see below) and the
  sections in `references/` for the pillar/spread, phase-transformation,
  and thermal-model plugins, each read via their own `docs/docs.tex` or
  source.
- **Shallow coverage**: the GUI/app/tooling repos and `pyroll-sandbox`
  (`references/ecosystem-tools.md`) - read only at the README/directory-
  listing level, since they're applications and exploratory scripts
  rather than documented physics models. Don't treat statements there as
  more authoritative than a quick look at the actual repo would give you.

This skill is local to this repo only (not published as a shared plugin) -
it travels with clones of `pyroll-flat-rolling-slab-model`
(formerly `pyroll-karman-force-torque`), the same reach as this repo's
`CLAUDE.md`.

## The framework layer: `pyroll-core`

Every plugin below is a hook implementation on top of `pyroll-core`'s
object model. If you're new to PyRoll, understand this layer first:

- **Object model**: `Profile` (the rolled stock's cross-section/state) is
  transformed by a sequence of `Unit`s - `RollPass` (`TwoRollPass`,
  `ThreeRollPass`, `SymmetricRollPass`), `Transport`, `CoolingPipe`,
  `Spooler`, `Rotator` - chained in a `PassSequence`. Deformation units
  discretize the pass into `DiskElement`s for incremental (in rolling-
  direction) modelling.
- **The Hook system** (`pyroll/core/hooks.py`) is the extension mechanism
  every plugin below uses: a `Hook[T]()` declared on a class (e.g.
  `RollPass.roll_force`) can be given a default implementation, then
  *overridden* by any plugin or user config without touching the
  original - this is how e.g. `karman_solution`, `pillar_spreads`, or
  `domanti_analysis` slot in. When you need to know what a hook does,
  grep for `Hook[` and its `@Class.hook_name` implementations, don't
  assume from the name alone.
  - **Core's own inclusion policy** (why some hooks live in core but most
    don't): hooks useful to *many* plugins (e.g. `Profile.surface_temperature`,
    falling back to `self.temperature`) are included in core with a
    sensible fallback; hooks that only make sense for one specific model
    (flow-stress coefficients, spread coefficients) stay in that model's
    own plugin. If you're deciding where a new hook belongs, this is the
    test to apply.
- **Groove shapes**: ovals, diamonds, boxes, rounds, hexagonal, flat -
  `pyroll/core/grooves/`.
- **`generic_elongation.py`/`generic_elongation_solvers.py`**: core's own
  basic spread/elongation *fallback*, used absent a more specific plugin
  like the pillar model or Domanti spreading below.
- Versioning: plugin major version should track core's major version
  (v3.x plugin ↔ v3.x core) - check this when something seems
  incompatible across repos.

## Flat-rolling slab theory - this repo

Von Kármán/Siebel/Orowan elementary slab theory, Mises yield, the
neutral-point shooting method, Orowan's inhomogeneity closure (and the
three interpretive ambiguities resolved reconstructing it from a
secondary source), Ekelund's redundant-shear correction, Bay & Wanheim
mixed friction, Hitchcock/full-elastic roll flattening, elastic-plastic
entry/exit zones, the layer/slab-splitting model with thermal coupling,
full elastic-contour foil rolling, and Troost's speed/inertia extension -
**all fully written up, with equations and numeric checkpoints, in this
repo's own `CLAUDE.md`** (repo root, three directories up from this file).
Read that directly rather than a summary here; it's already the
comprehensive version and duplicating it would just create a second copy
to keep in sync.

## Other physics models - `references/`

Each file covers one or two related repos. Read the one relevant to your
task rather than all of them:

- **`references/pillar-and-spread.md`** — `pyroll-pillar-model` (profile
  discretization into width-wise pillars; the base package most
  spread/groove-filling models build on) and `pyroll-domanti-spreading`
  (Domanti's differential/ODE spread model giving the width distribution
  *along* the roll gap, not just entry/exit). Read this for anything
  about spread, groove filling, or profile width prediction.
- **`references/phase-transformation.md`** — `pyroll-phase-transformation-model`:
  austenite decomposition (ferrite/pearlite/bainite/martensite) from the
  cooling thermal history, plus derived hardness and mechanical
  properties. Read this for anything about microstructure, hardness, or
  post-rolling mechanical properties.
- **`references/thermal-models.md`** — `pyroll-bar-in-coil-temperature-model`
  (wire coil cooling on a spooler) and the conveyor/laying-head model
  (Stelmor-type roller conveyor cooling; repo named
  `pyroll-conveyor-temperature-model`, package `pyroll.laying_head`).
  Read this for anything about coil cooling, conveyor cooling, or the
  laying head.
- **`references/ecosystem-tools.md`** — the GUI/app repos
  (`pyroll-gui`, `pyroll-cooling-pipe-gui`, `pyroll-simple-process-evaluation`),
  data/tooling repos (`pyroll-pass-sequence-database`, `pyroll-validation`),
  and `pyroll-sandbox`'s exploratory scripts. Read this before starting a
  *new* model from scratch - `pyroll-sandbox` may already have relevant
  exploratory work (forward slip, roll-angle geometry, induction heating,
  tension analysis, and more), and it's worth checking there before
  re-deriving something from zero.

## Using this knowledge

- **Answering a physics question**: find the right section above, read
  the pointed-to file (this repo's `CLAUDE.md` or the relevant
  `references/*.md`), and cite it - don't answer from vague memory of
  "rolling theory" when a specific, sourced model exists here.
- **Extending or fixing a model**: the actual repo (cloned separately) has
  the real source and tests; this skill tells you *which* repo and *what
  to expect* there, not a substitute for reading its code.
- **Starting a new model**: check `references/ecosystem-tools.md`'s
  `pyroll-sandbox` section first - reinventing something that already has
  a rough exploratory version there wastes effort on both sides.
- **Don't over-claim depth**: if asked something about one of the
  shallow-coverage repos (`references/ecosystem-tools.md`) beyond what's
  written there, say so and offer to actually read that repo rather than
  extrapolating.
