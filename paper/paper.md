---
title: 'cortecs: A Python package for compressing opacities'
tags:
  - Python
  - astronomy
  - radiative transfer
authors:
  - name: Arjun B. Savel
    orcid: 0000-0002-2454-768X
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Megan Bedell
    orcid: 0000-0001-9907-7742
    affiliation: 2
  - name: Eliza M.-R. Kempton
    orcid: 0000-0002-1337-9051
    affiliation: 1
affiliations:
  - name: Astronomy Department, University of Maryland, College Park, 4296 Stadium Dr., College Park, MD 207842 USA
    index: 1
  - name: Flatiron Institute, Simons Foundation, 162 Fifth Avenue, New York, NY 10010, USA
    index: 2
date: 26 August 2023
bibliography: paper.bib

---

# Summary

Observations of exoplanet atmospheres at a number of wavelengths encode details of the atmospheric
composition, temperature structure, and dynamics. Simulating these observations requires knowledge
of the opacity of gases within the atmosphere. When modeling broad wavelength ranges, opacity data for
even a single gas species can take up multiple gigabytes of memory. This feature can be a limiting
factor in determining the number of gases to consider in a simulation, or even the architecture of
the computing system used to perform the simulation. We present `cortecs`, a tool for compressing
opacity data used to simulate spectra. `cortecs` provides flexible methods for fitting the
temperature and pressure dependence of opacity data and for evaluating the opacity with accelerated,
GPU-friendly methods.

# Statement of need
Recent advances in high-resolution spectroscopy of exoplanet atmospheres has required simulations of spectra over
tens of thousands of wavelength points. To increase the speed of these computations, some models have parallelized
the problem on GPUs (cite a bunch). However, GPUs in general do not have large amounts of memory; only the cutting-edge,
expensive GPUs have memory in excess of 30 GB (cite). Memory management is therefore of concern when simulating
high-resolution spectra.

While the wavelength dependence of opacity is sharp for many gases, the temperature and pressure dependencies are generally smooth and
similar across wavelengths. This feature implies that the opacity data should be compressible without significant loss of
accuracy at the spectrum level.


# Methods
`cortecs` seeks to compress redundant information by representing opacity not as the
values themselves but as fits to the values. We provide three methods of increasing complexity (and flexibility) for
compressing and decompressing opacity: polynomial-fitting, PCA, and neural networks. Each compression method is paired
with a decompression method that can be used to evaluate opacity values. These decompression methods are tailored
for usage on GPUs and are accelerated with the `JAX` code transformation framework (cite).

In addition to these compression/decompression methods, `cortecs` provides utility scripts for working with large opacity files.
For instance, `cortecs` can convert opacity files between popular formats, "chunk" opacity files for parallel
computations across CPUs, and add overlap between chunked files for calculations that include Doppler shifts.

Test a citation `@line:2021`

# Example: High-resolution retrieval of WASP-77Ab?
As a proof of concept, we run a parameter inference code (a "retrieval") on the thermal emission spectrum of the
fiducial hot Jupiter WASP-77Ab (cite a bunch) with `cortecs`-compressed opacity. The retrieval pairs PyMultiNest sampling
with the CHIMERA radiative transfer code (cite). For this experiment, we use the PCA-based compression scheme implemented
in `cortecs`.

We find that our compressed-opacity retrieval yields posterior distributions and Bayesian evidences that are consistent with those from the benchmark
retrieval using uncompressed opacity. The results from this exercise indicate that our compression/decompression scheme
is accurate enough to be used in high-resolution retrievals.


# Acknowledgements

A.B.S. and E.M-R.K. acknowledge support from the Heising-Simons Foundation. We thank Max Isi for helpful discussions.

# References
