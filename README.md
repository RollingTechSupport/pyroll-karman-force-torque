# PyRolL Karman power and labour

PyRolL Plugin for calculation of power and labour solving von-Karman ODE for a equivalent flat pass.

The plugin automatically selects between three solvers depending on the pass,
covering the full range from thick slabs to metal foils:

- **`KarmanSolver`** (rigid roll, pure Coulomb friction, no elastic zones) -
  the classical model, used whenever elastic effects are negligible or the
  roll/profile elastic properties (`elastic_modulus`/`poissons_ratio`)
  aren't set at all.
- **`LayerRollingSolver`** (medium and thick passes) - adds elastic
  entry/exit zones, a mixed Coulomb/stiction friction law (Bay & Wanheim),
  and through-thickness resolution into multiple layers with full thermal
  coupling. Ports Max Weiner's thesis work (TU Bergakademie Freiberg).
  Single-layer (`layer_count=1`) is the "medium"/Orowan-like configuration;
  multiple layers resolve the "thick slab" case where surface and core
  develop meaningfully different flow stress through the pass.
- **`FoilRollingSolver`** (foil rolling) - computes the true,
  elastically-flattened (generally non-circular) roll-gap shape instead of
  assuming a circular contact, following Mauk & Overhagen, "Prozessmodell
  zum Kaltwalzen von Metallfolien mit keramischen Arbeitswalzen auf
  Mehrwalzengerüsten" (2013).

The dispatch order is: Hitchcock's flattened-radius ratio `r'/r` decides
foil vs. not (`RollPass.foil_rolling_hitchcock_limit`, default `2.0`, via
`RollPass.foil_rolling_condition`); otherwise the classical contact-length /
mean-thickness ratio `Ld/Hm` decides thick (multiple layers) vs. medium
(single layer) (`RollPass.layer_model_ld_hm_limit`, default `1.0`, via
`RollPass.thick_slab_condition`; layer count via
`RollPass.layer_model_layer_count`, default `5`). All of these are
overridable hooks, including forcing a specific model unconditionally. The
elastic-plastic models require `elastic_modulus`/`poissons_ratio` on both
the roll and the profile; the layer model's thermal coupling additionally
needs `specific_heat_capacity`/`thermal_conductivity`/`density` on both.
Without the elastic properties, the plugin falls back to the classical
rigid-roll model.

Roll flattening itself is out of this plugin's scope: all three solvers
just read `Roll.working_radius` once, the same hook `pyroll-core` already
provides, rather than deriving or iterating on a flattened radius
themselves. A separate plugin hooking `Roll.working_radius` (e.g. with
Hitchcock's relation) is picked up transparently by whichever solver
runs; this plugin's own Hitchcock ratio (`hitchcock_radius_ratio` in
`condition.py`) is used purely to classify a pass for dispatch
(`foil_rolling_condition`), never fed back into a calculation.

For the docs, see [here](docs/docs.pdf) (source in [`docs/docs.tex`](docs/docs.tex)).

This project is licensed under the [BSD-3-Clause license](LICENSE).

The package is available via [PyPi](https://pypi.org/project/pyroll-karman-power-and-labour/) and can be installed with

    pip install pyroll-karman-power-and-labour