# PyRolL Karman power and labour

PyRolL Plugin for calculation of power and labour solving von-Karman ODE for a equivalent flat pass.

The plugin automatically selects between three solvers depending on the pass,
covering the full range from thick slabs to metal foils. All three are
elastic-plastic (Hooke's-law entry/exit zones bounding the plastic zone) and
use Bay & Wanheim's smoothed mixed Coulomb/sticking friction law throughout
- there is no rigid-plastic or pure-Coulomb fallback:

- **`KarmanMixedFrictionSolver`** ("medium" passes) - a single homogeneous
  slab, elastic-plastic zones, Bay & Wanheim mixed friction. No thermal
  coupling. Also available as `OrowanSolver`, which uses the same
  elastic-plastic/mixed-friction construction but Orowan's own
  inhomogeneity-corrected closure (Orowan, "The Calculation of Roll Pressure
  in Hot and Cold Flat Rolling", Proc. IMechE 150, 1943) instead of the
  plain Mises one - a substantially different, circular-arc/inhomogeneity-
  function model; see `docs/docs.tex` for the distinction. `OrowanSolver` is
  not part of the automatic dispatch (available for direct use).
- **`LayerRollingSolver`** (thick passes) - through-thickness resolution
  into multiple layers with full thermal coupling, on top of the same
  elastic-plastic/mixed-friction construction. Ports Max Weiner's thesis
  work (TU Bergakademie Freiberg). Used for the "thick slab" case where
  surface and core develop meaningfully different flow stress through the
  pass.
- **`FoilRollingSolver`** (foil rolling) - computes the true,
  elastically-flattened (generally non-circular) roll-gap shape instead of
  assuming a circular contact, following Mauk & Overhagen, "Prozessmodell
  zum Kaltwalzen von Metallfolien mit keramischen Arbeitswalzen auf
  Mehrwalzengerüsten" (2013).

The dispatch order is: Hitchcock's flattened-radius ratio `r'/r` decides
foil vs. not (`RollPass.foil_rolling_hitchcock_limit`, default `2.0`, via
`RollPass.foil_rolling_condition`); otherwise the classical contact-length /
mean-thickness ratio `Ld/Hm` decides thick (`LayerRollingSolver`, multiple
layers) vs. medium (`KarmanMixedFrictionSolver`) (`RollPass.layer_model_ld_hm_limit`,
default `1.0`, via `RollPass.thick_slab_condition`; layer count via
`RollPass.layer_model_layer_count`, default `5`). All of these are
overridable hooks, including forcing a specific model unconditionally.
`elastic_modulus`/`poissons_ratio` on both the roll and the profile are
required for every pass (needed for the Hitchcock-ratio/Ld-Hm checks
themselves, and for every solver's own elastic zones) - there is no
fallback for passes that omit them, they simply error. The layer model's
thermal coupling additionally needs `specific_heat_capacity`/
`thermal_conductivity`/`density` on both.

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