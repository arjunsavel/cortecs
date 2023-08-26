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
the problem on GPUs (e.g., @line:2021, @lee20223d). However, GPUs in general do not have large amounts of memory
(e.g., @ito2017ooc_cudnn); only the cutting-edge, most expensive GPUs have memory in excess of 30 GB
(such as the NVIDIA A100 or H100). Memory management is therefore a clear concern when simulating
high-resolution spectra.

While the wavelength dependence of opacity is sharp for many gases, the temperature and pressure dependencies are generally smooth and
similar across wavelengths. This feature implies that the opacity data should be compressible without significant loss of
accuracy at the spectrum level.


# Methods
`cortecs` seeks to compress redundant information by representing opacity not as the
values themselves but as fits to the values. We provide three methods of increasing complexity (and flexibility) for
compressing and decompressing opacity: polynomial-fitting, principal components analysis (PCA; e.g., @jolliffe2016principal)
and neural networks (e.g., @alzubaidi2021review). Each compression method is paired
with a decompression method that can be used to evaluate opacity values. These decompression methods are tailored
for usage on GPUs and are accelerated with the `JAX` code transformation framework [@jax2018github].

In addition to these compression/decompression methods, `cortecs` provides utility scripts for working with large opacity files.
For instance, `cortecs` can convert opacity files between popular formats, "chunk" opacity files for parallel
computations across CPUs, and add overlap between chunked files for calculations that include Doppler shifts.


# Example: High-resolution retrieval of WASP-77Ab?
As a proof of concept, we run a parameter inference code (a "retrieval") on the thermal emission spectrum of the
fiducial hot Jupiter WASP-77Ab [@line:2021; @mansfield2022confirmation; @august2023confirmation] with
`cortecs`-compressed opacity. The
retrieval pairs PyMultiNest [@buchner2014x] sampling with the CHIMERA radiative transfer code [@line2013systematic].
For this experiment, we use the PCA-based compression scheme implemented
in `cortecs`.

We find that our compressed-opacity retrieval yields posterior distributions (as plotted by the `corner` package; @corner)
and Bayesian evidences that are consistent with those from the benchmark
retrieval using uncompressed opacity  \autoref{fig:corner}. The results from this exercise indicate that our compression/decompression scheme
is accurate enough to be used in high-resolution retrievals.

![The posterior distributions for our baseline WASP-77A retrieval (teal).
and our retrieval using opacities compressed by `cortecs` (gold). \label{fig:corner}](pca_compress.png)


# Acknowledgements

A.B.S. and E.M-R.K. acknowledge support from the Heising-Simons Foundation. We thank Max Isi for helpful discussions.

# References
