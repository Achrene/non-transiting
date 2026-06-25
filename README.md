````markdown
# Non-transiting

A Python package for modelling, generating and fitting optical phase curves of non-transiting exoplanets.

This repository provides tools to compute physically motivated phase curves of non-transiting exoplanets, generate multidimensional model grids, and fit observed photometric phase curves through least-squares or Bayesian inference.

The code was developed for the analysis presented in

> Breton et al. (2026), *Hidden worlds: a non-transiting candidate planet in the Neptunian desert around the solar-type pulsator KIC 9139163*.

---

## Features

- Compute optical phase curves of non-transiting exoplanets
- Generate multidimensional grids of synthetic phase curves
- Fit observed phase curves using least-squares or nested sampling
- Includes the contributions from
  - Planetary reflection
  - Thermal emission
  - Doppler beaming
  - Stellar ellipsoidal distortion
- Supports **Kepler** and **TESS** instrumental response functions
- Parallel grid generation using `joblib`
- Publication-quality diagnostic plots

---

## Reflection models

Two different reflection prescriptions are implemented.

### Global reflection

The global reflection model assumes a constant geometric albedo over the planetary surface. Reflection is described by

- Geometric albedo
- Heat redistribution factor
- Cloud offset

The cloud-offset parameter shifts the longitude of maximum reflected flux, allowing asymmetric and non-Lambertian phase curves while remaining computationally efficient.

This reflection prescription is used to generate the multidimensional model grids published alongside the paper.

### Local reflection

The local reflection model computes the reflected flux by integrating the contribution of individual surface elements whose albedo can vary across the planetary surface. This produces self-consistent phase-dependent reflection patterns and allows longitudinal variations in planetary reflectivity.


---

## Repository structure

```text
.
├── model/
│   ├── ExoplanetarySystem.py      # Physical phase-curve model
│   ├── create_grid.py             # Generate model grids
│   ├── Grid_fit.py                # Fit observed phase curves
│   ├── Grid_fit_for_joint_fit.py  # Joint Kepler + TESS fitting
│   ├── diagnosis.py               # Diagnostic plots
│   ├── fit_parameters.yaml        # Fitting configuration
│   └── utils.py
│
├── references/                    # Instrument transmission curves
├── results_model/                 # Generated model grids
├── fit/                           # Fitting outputs
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Achrene/non-transiting.git
cd non-transiting
```

Install the required Python packages

```bash
pip install numpy scipy matplotlib pandas astropy joblib dynesty pyyaml
```

---

## Computing phase curves

The physical model is implemented in

```text
model/ExoplanetarySystem.py
```

and computes the total optical phase curve as the sum of

- Reflected planetary light
- Thermal emission
- Doppler beaming
- Stellar ellipsoidal distortion

Both global and local reflection prescriptions are available.

---

## Generating model grids

Model grids are generated using

```bash
python model/create_grid.py
```

The script computes synthetic phase curves over a multidimensional parameter space and stores the resulting grid together with metadata describing the sampled parameters.

Currently, grids can be generated using either the **Kepler** or **TESS** instrumental transmission curves.

---

## Published grids

The published grids span the following parameter space.

| Parameter | Range | Number of values |
|-----------|------:|-----------------:|
| Planet radius ($R_{\rm Jup}$) | 0.1–1.0 | 10 |
| Orbital inclination (°) | 17–62 | 45 |
| Geometric albedo | 0.05–0.99 | 10 |
| Heat redistribution factor | 0.05–0.99 | 10 |
| Cloud offset (°) | -90–90 | 20 |

The inclination range excludes transiting systems (90°), focusing on non-transiting geometries.

These parameter ranges can be modified directly in `create_grid.py` to generate custom grids.

---

## Configuration

The fitting procedure is configured through

```text
model/fit_parameters.yaml
```

The configuration file specifies

- Stellar parameters
- Planetary and orbital parameters
- Photometric datasets
- Instrument
- Model grid location
- Fitting method
- Optimisation settings
- Output directories

Changing this file is generally sufficient to analyse another target once an appropriate model grid has been generated.

---

## Output

Generated model grids consist of multidimensional arrays of synthetic phase curves together with metadata describing the parameter space.

The accompanying `.json` files contain

- Parameter names
- Parameter values
- Parameter ordering
- Grid dimensions

These metadata are used during interpolation when fitting observed phase curves.

---

## Fitting observations

Observed light curves can be fitted using either

- Least-squares optimisation
- Bayesian nested sampling (`dynesty`)

The fitting routines interpolate within the pre-computed model grids, allowing efficient exploration of the parameter space without recomputing phase curves at every likelihood evaluation.

---

## Citation

If you use this code in your research, please cite

> Breton et al. (2026), *Hidden worlds: a non-transiting candidate planet in the Neptunian desert around the solar-type pulsator KIC 9139163*.

If you use the published model grids, please also cite the corresponding Zenodo archive.

---

## License

This project is distributed under the terms of the MIT License.
````
