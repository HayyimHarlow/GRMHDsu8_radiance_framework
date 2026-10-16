# SU(8) Radiance Framework v2.0.0 — Production HPC Release

## Overview

Version 2.0.0 of the SU(8 Radiance Framework introduces full HPC support for
production-scale GRMHD and GRRT simulations, including SU(8 source-term
patches for HARM/BHAC, SU(8)-modified emissivities for ipole/RAPTOR, and
a complete parameter grid suitable for EHT-style synthetic imaging studies.

## New Features

- SU(8 source-term implementation in HARM/BHAC (energy, momentum, magnetic field)
- SU(8-modified synchrotron emissivities and EVPA rotation in ipole/RAPTOR
- SLURM-based HPC job scripts for GRMHD + GRRT runs
- Full production parameter grid over $(\kappa_{\mathrm{ac}}, \gamma_{\mathrm{ac}},
  T_{\mathrm{prec}}, \alpha_{\mathrm{SU8}})$
- Convergence tests at $256^3$ and $512^3$ resolutions
- Archival of FITS images, visibility tables, closure phases, EVPA time series,
  and posterior chains under DOI 10.5281/zenodo.23290505

## Changes

- Executor extended to handle external HPC-produced datasets
- Documentation updated with HPC migration and SU(8 patch details
- README updated with production workflow and data products

## Known Limitations

- SU(8) sector currently implemented for a subset of GRMHD base codes (HARM/BHAC)
- GRRT modifications tested primarily at 230 GHz and 345 GHz
- Bayesian inference runs limited to a subset of the full parameter grid

## How to Cite

Please cite:

Harlow, H. (2026).
SU(8 Radiance Framework (v2.0.0).
Zenodo. https://doi.org/10.5281/zenodo.23290505
