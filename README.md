# PyRolL Karman power and labour

PyRolL Plugin for calculation of power and labour solving von-Karman ODE for a equivalent flat pass.

For thin strip/foil passes, the plugin automatically switches from the rigid-roll
von-Karman solution to a full elastic-plastic foil-rolling model (following
Mauk & Overhagen, "Prozessmodell zum Kaltwalzen von Metallfolien mit
keramischen Arbeitswalzen auf Mehrwalzengerüsten", 2013) that computes the
true, elastically-flattened roll-gap shape instead of assuming a circular
contact. The switch is governed by Hitchcock's flattened-radius ratio
`r'/r`: once it reaches `RollPass.foil_rolling_hitchcock_limit` (default `2.0`,
per the report), the rigid model is no longer valid and
`RollPass.foil_rolling_condition` triggers the foil model instead. Both hooks
are overridable if you want to force one model or tune the threshold. The
foil model requires `elastic_modulus`/`poissons_ratio` to be set on both the
roll and the profile; without them the plugin falls back to the classical
rigid-roll model.

For the docs, see [here](docs/docs.pdf).

This project is licensed under the [BSD-3-Clause license](LICENSE).

The package is available via [PyPi](https://pypi.org/project/pyroll-karman-power-and-labour/) and can be installed with

    pip install pyroll-karman-power-and-labour