# Non-transiting

A Python package for modelling and fitting optical phase curves of non-transiting exoplanets.

`non-transiting` computes synthetic phase curves including planetary reflection, thermal emission, Doppler beaming and stellar ellipsoidal distortion. The package can generate multidimensional grids of models for rapid parameter inference or compute individual phase curves using more detailed reflection prescriptions.

The code was developed for the analysis presented in

> Breton et al. (2026), *Hidden worlds: a non-transiting candidate planet in the Neptunian desert around the solar-type pulsator KIC 9139163*.

## Reflection models

Two reflection models are available.

### Global reflection

The global reflection model assumes a constant geometric albedo over the planetary surface. Reflection is controlled by

* geometric albedo,
* heat redistribution factor,
* cloud offset.

The cloud offset shifts the longitude of maximum reflected flux, allowing asymmetric phase curves that approximate non-Lambertian scattering. This prescription is computationally efficient and is used to generate the published model grids.

### Local reflection

The local reflection model computes the reflected flux by integrating the contribution of individual surface elements with a spatially varying albedo. This allows more realistic phase-dependent brightness distributions and is intended for modelling individual systems.

## Main capabilities

The package can

* compute synthetic phase curves of non-transiting exoplanets,
* generate multidimensional model grids,
* fit observed phase curves using least-squares or nested sampling,
* model observations in the **Kepler** and **TESS** bandpasses.

The total phase curve includes contributions from

* reflected light,
* thermal emission,
* Doppler beaming,
* stellar ellipsoidal distortion.

## Repository

```text
model/
    ExoplanetarySystem.py      Core phase-curve model
    create_grid.py             Grid generation
    Grid_fit.py                Grid fitting
    Grid_fit_for_joint_fit.py  Joint Kepler/TESS fitting
    diagnosis.py               Plotting utilities
    fit_parameters.yaml        Configuration

references/                    Instrument response functions
results_model/                 Generated grids
fit/                           Fitting results
```

## Computing model grids

Model grids are generated with

```bash
python model/create_grid.py
```

The published grids are available for both the **Kepler** and **TESS** instrumental bandpasses.

The default parameter space is

| Parameter                     | Range     |
| ----------------------------- | --------- |
| Planet radius ($R_{\rm Jup}$) | 0.1–1.0   |
| Inclination (°)               | 17–62     |
| Geometric albedo              | 0.05–0.99 |
| Heat redistribution           | 0.05–0.99 |
| Cloud offset (°)              | -90–90    |

The parameter ranges can be modified in `create_grid.py` to generate custom grids.

## Configuration

Fitting is controlled through `model/fit_parameters.yaml`, which specifies the stellar and orbital parameters, observational data, optimisation settings and output directories.

Metadata describing the generated grids are stored in the accompanying `.json` files, including the parameter names, sampled values and grid dimensions.

## Citation

If you use this package, please cite

> Breton et al. (2026), *Hidden worlds: a non-transiting candidate planet in the Neptunian desert around the solar-type pulsator KIC 9139163*.

If you use the published model grids, please also cite the accompanying Zenodo archive.
