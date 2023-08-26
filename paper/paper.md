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

The absorption and emission of light by exoplanet atmospheres encodes details of the atmospheric
composition, temperature structure, and dynamics. Simulating these processes requires detailed knowledge
of the opacity of gases within an atmosphere. When modeling broad wavelength ranges, such opacity data for
even a single gas can take up multiple gigabytes of memory. This feature can be a limiting
factor in determining the number of gases to consider in a simulation, or even choosing the architecture of
the system used for the simulation. We present `cortecs`, a Python tool for compressing
opacity data used to compute spectra. `cortecs` provides flexible methods for fitting the
temperature, pressure, and wavelength dependence of opacity data and for evaluating the opacity with accelerated,
GPU-friendly methods. The package is developed on GitHub (https://github.com/arjunsavel/cortecs), and it is
available for download with `pip` and `conda`.

# Statement of need
Observations with the latest high-resolution spectrographs [@mace2018igrins; @seifahrt2020sky; @pepe2021espresso]
have motivated memory-intensive simulations of exoplanet atmospheres. `cortecs` enables these simulations with more
gases and on a broader range of architectures by compressing opacity data.

Broadly, generating a spectrum to compare against recent cutting-edge high-resolution data requires evaluating the
radiative transfer equation over tens of thousands of wavelength points. To decrease computational runtime,
some codes have parallelized the problem on GPUs (e.g., @line:2021, @lee20223d). However, GPUs in general do not have large amounts of memory
(e.g., @ito2017ooc_cudnn); only the cutting-edge, most expensive GPUs have memory in excess of 30 GB
(such as the NVIDIA A100 or H100). Memory management is therefore a clear concern when producing
high-resolution spectra.

How do we decrease the memory footprint of these calculations? By far the largest contributor to the memory footprint,
at least as measured on disk, is the opacity data. For instance, the opacity data for a single gas species across
the IGRINS wavelength range [@mace2018igrins] takes up 2.5 GB of memory at a resolution of 400,000. It stands to reason
that decreasing the amount of memory consumed by opacity data would strongly decrease the total amount of memory consumed
by the radiative transfer calculation.

The key to the solution is redundancy: While the wavelength dependence of opacity is sharp for many gases,
the temperature and pressure dependencies are generally smooth and similar across wavelengths (cite a bunch of opacity papers).
This feature implies that the opacity data should be compressible without significant loss of
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
