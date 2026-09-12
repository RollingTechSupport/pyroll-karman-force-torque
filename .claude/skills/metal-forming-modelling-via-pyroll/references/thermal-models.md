# Coil and conveyor thermal models

Two repos, both post-rolling cooling models for long/wire products, read
via their READMEs (bar-in-coil's is thorough and detailed; conveyor's
README turned out to actually describe a differently-named internal
model — see below) this session.

## `pyroll-bar-in-coil-temperature-model` — wire coil cooling on a spooler

Transient temperature model for wire coils wound on a spooler: predicts
every individual winding's temperature during coiling and during the
subsequent free-standing cool-down (after removal from the mandrel), so
the full cross-section temperature field (hot core, cooler bore, mantle,
end faces) is available downstream.

**Mechanism**: the coil is discretized into its windings, each a lumped
thermal node on a regular layer/position grid. Each node exchanges heat
by:
- **conduction** with radial (layer-to-layer) and axial
  (winding-to-winding) neighbors, through an effective coil thermal
  conductivity,
- **contact conduction** into the mandrel (inner bore) and spooler plates
  (end faces) — only while still on the spooler, i.e. before a
  configurable `stripping_time`,
- **convection** to ambient — outer mantle only while on the spooler, all
  four faces (bore, mantle, top, bottom) once free-standing, each with
  its own adjustable heat transfer coefficient,
- **radiation** to ambient at the same air-exposed faces.

Windings enter the simulation sequentially at the bar's own temperature,
timed from the laid-down wire length and feed velocity. Integrated with
an *explicit* scheme with automatic stability sub-stepping, fully
vectorized over windings — thousands-of-windings coils solve in a
fraction of a second. **Calibrated** against digitized core/outer-surface
cooling curves from Nicola Simaz's thesis (coil wound at 740°C,
`tests/test_thesis_simaz.py`).

Registers hooks on `pyroll.core.Spooler` (import the package to activate;
`Spooler.winding_temperatures` is the key output, shape
`(windings, time_steps)`; also feeds `out_profile.temperature` as the
mass-weighted mean at end of cooling). Interoperates with
`pyroll-ring-model-thermal` if installed: an inhomogeneous incoming ring
temperature enters at its area-weighted mean, and the coil leaves with a
*homogeneous* ring distribution at the mass-weighted mean (overriding
`ring_temperatures`/`surface_temperature`/`core_temperature` on
`Spooler.OutProfile`) so downstream ring-resolved transports continue
seamlessly.

Extensive hook table in the repo's own `README.md` (convection/contact/
radiation coefficients per surface, ambient/mandrel/plate temperatures,
`simulation_time_step`, `cooling_duration`) — read that directly rather
than this summary if you need to actually configure a run.

## Conveyor / laying-head model

**Naming note** (worth knowing so you don't get confused searching): the
repo is named `pyroll-conveyor-temperature-model`, but its README and
internal package describe it as the **laying-head** model —
`pyroll.laying_head`, `StelmorConveyor`. The repo's own description
frames it more generically as "conveyor temperature" but it specifically
implements the Stelmor-type roller-conveyor case after a laying head, not
a generic conveyor.

**Model**: implements A. Lindemann's 2003 dissertation (*Simulation der
Drahtabkühlung nach dem Warmwalzen im Bereich der Luftkühlstrecke*, Otto-
von-Guericke-Universität Magdeburg). The wire is laid down by the laying
head in overlapping circular loops ("windings") carried through the
air-cooling section by the roller table. Because winding pitch decreases
from the conveyor's centre towards its edges, local packing density — and
with it achievable heat transfer — varies strongly across the table
width. The plugin resolves this across-the-table temperature field.
Physics: forced/free/mixed convection, radiation, and axial-conduction
equations (thesis Sec. 3.2-3.7).

**Package layout** (worth knowing before extending it):
- `lindemann_model.py` — standalone, PyRolL-independent implementation of
  the thesis equations (no PyRolL import at all, independently
  unit-testable — a good pattern to follow if you're implementing another
  literature model here).
- `profile.py` — Profile-level hooks (`lay_temperature_profile`,
  `lay_x_grid`).
- `transport.py` — `StelmorConveyor(Transport)`, behaves like any other
  PyRolL `Transport` unit.
- `hookimpls.py` — wires `lindemann_model` into the PyRolL hook/profile-
  chaining machinery.
- `ring_coupling.py` — optional `pyroll-ring-model-thermal` coupling,
  active only if that plugin is importable.
- `config.py` — the `PYROLL_LAYING_HEAD` config section.
- The repo-root `laying_head/` folder (Jupyter notebook + jupytext-paired
  script) is the *original research sandbox* used to derive/validate the
  model — kept for reference, not part of the installable package. (See
  also `references/ecosystem-tools.md`'s note on `pyroll-sandbox`, which
  has its own separate, earlier laying-head exploration predating this
  formalized plugin.)

## When these matter

Anything about coil/bundle cooling after coiling (bar-in-coil) or about
wire-rod cooling on the roller table right after the laying head
(conveyor/Stelmor) — both feed into the phase-transformation model
(`references/phase-transformation.md`) as the thermal history driving
final microstructure.
