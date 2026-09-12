# Ecosystem tooling, apps, and exploratory work

Read at README/directory-listing depth only this session — these are
applications and scaffolding built on the physics plugins above, or
not-yet-formalized research, not documented physics models themselves.
Treat anything here as a starting pointer to go read the actual repo,
not as settled fact.

## Apps (GUIs)

- **`pyroll-gui`** — "PyRolL - Rolling Mill Simulation Interface", an
  Electron desktop app (`package.json`, `electron/`, `src/`, a Python
  `backend/`). The main end-user-facing simulation interface for the
  ecosystem.
- **`pyroll-cooling-pipe-gui`** — small standalone web GUI (`backend/` +
  `frontend/`) "to evaluate the performance of a cooling pipe". Also has
  a `cooling-pipe-model.ipynb` notebook and an `in_profile/` folder —
  worth checking the notebook first if you need the actual cooling-pipe
  physics, since it likely predates/documents what the backend
  implements.
- **`pyroll-simple-process-evaluation`** — standalone web GUI (`backend/`
  + `frontend/`) "to set BGV / FFB roll speeds and minimise inter-stand
  tension" — a mill-control/process-optimization tool built on top of
  whatever mechanical models are installed, not a model itself.

## Data and infrastructure

- **`pyroll-pass-sequence-database`** — not a model: a collection of
  actual pass-schedule data (`pass_sequence_wwa`, `pass_sequence_wwb`,
  `wwa_products.json`, `wwb_products.json`) plus Jupyter notebooks for
  building the database and fitting material/waterbox coefficients
  (`wwb_fit_material_coefficient.ipynb`, `wwb-fitting-waterbox-coefficients.ipynb`).
  Useful for real-world pass-schedule examples/validation data, not
  physics.
- **`pyroll-validation`** — empty repository as of this session (no
  commits/content). Presumably intended as a home for cross-plugin
  validation cases; check again before assuming it's still empty.

## `pyroll-sandbox` — check here before building a new model

A personal research playground, not a package (no `pyroll/` namespace
directory, no formal plugin structure) — a flat collection of topic
folders and scripts:

- `domanti_spreading/` — an *earlier* exploration of Domanti spreading,
  predating and separate from the now-formalized `pyroll-domanti-spreading`
  plugin (`references/pillar-and-spread.md`). If working on spread
  models, check whether this sandbox version has ideas/data not yet
  ported into the real plugin.
- `laying_head/` — likewise, exploratory work related to (and possibly
  predating) the formalized laying-head/conveyor plugin
  (`references/thermal-models.md`).
- `coiling/` — coil-related exploration, possibly related to or predating
  `pyroll-bar-in-coil-temperature-model`.
- `forward_slip/` — forward-slip (rolling kinematics) analysis.
- `angles/` — roll/pass angle geometry.
- `induction-heating/` — induction heating exploration.
- `tension-analysis/` — inter-stand tension analysis (possibly related to
  what `pyroll-simple-process-evaluation` now does as a packaged tool).
- `kennedy_analysis/`, `bilz/`, `qiang_f/` — named after (presumably)
  people or specific analyses; not investigated further this session —
  read the folder directly if a task touches these names.
- `calculations.py` — a root-level loose calculations script.

**Why this matters**: if you're asked to build a new PyRoll model or
investigate a topic that sounds like it could already have exploratory
work (forward slip, tension analysis, induction heating, roll angles, or
anything coiling/spreading/laying-head related), check the matching
`pyroll-sandbox` folder *first* — reinventing something that already has
a rough version there wastes effort, and the sandbox version may reveal
data sources, references, or dead ends worth knowing about even if the
code itself isn't reusable directly.
