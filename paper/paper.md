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

The absorption and emission of light by exoplanet atmospheres encode details of atmospheric
composition, temperature, and dynamics. Fundamentally, simulating these processes requires detailed knowledge
of the opacity of gases within an atmosphere. When modeling broad wavelength ranges at high resolution, such opacity data for
even a single gas can take up multiple gigabytes of system random-access memory (RAM). This aspect can be a limiting
factor in determining the number of gases to consider in a simulation, or even in choosing the architecture of
the system used for the simulation. Here, we present `cortecs`, a Python tool for compressing
opacity data. `cortecs` provides flexible methods for fitting the
temperature, pressure, and wavelength dependencies of opacity data and for evaluating the opacity with accelerated,
GPU-friendly methods. The package is actively developed on GitHub (<https://github.com/arjunsavel/cortecs>), and it is
available for download with `pip`.

# Statement of need
Observations with the latest high-resolution spectrographs [e.g., @mace:2018; @seifahrt:2020; @pepe:2021]
have motivated RAM-intensive simulations of exoplanet atmospheres at high spectral resolution.
`cortecs` enables these simulations with more
gases and on a broader range of computing architectures by compressing opacity data.

Broadly, generating a spectrum to compare against recent high-resolution data requires solving the
radiative transfer equation over tens of thousands of wavelength points [e.g., @savel:2022;
@beltz:2023; @line:2021; @wardenier:2023;  @gandhi:2023; @prinoth:2023; @maguire:2023].
To decrease computational runtime,
some codes have parallelized the problem on GPUs [e.g., @line:2021; @lee:2022]. However, GPUs cannot in general hold large amounts of
data their video random-access memory (VRAM) [e.g., @ito:2017]; only the cutting-edge, most expensive GPUs are equipped with VRAM in excess of 30 GB
(such as the NVIDIA A100 or H100). RAM and VRAM management is therefore a clear concern when producing
high-resolution spectra.

How do we decrease the RAM footprint of these calculations? By far the largest contributor to the RAM footprint,
at least as measured on disk, is the opacity data. For instance, the opacity data for a single gas species across
the wavelength range of the IGRINS spectrograph [@mace:2018] takes up 2.5 GB of non-volatile memory (i.e., the file size is 2.5 GB) at float64 precision and at a resolving power of 400,000
(as used in [@line:2021], with 39 temperature points and 18 pressure points, using, e.g., the [@polyansky:2018] water opacity tables). It stands to reason
that decreasing the amount of RAM/VRAM consumed by opacity data would strongly decrease the total amount of RAM/VRAM consumed
by the radiative transfer calculation.

One solution is to isolate redundancy: While the wavelength dependence of opacity is sharp for many gases,
the temperature and pressure dependencies are generally smooth and similar across wavelengths [e.g.,
@barber:2014; @polyansky:2018; @coles:2019].
This feature implies that the opacity data should be compressible without significant loss of
accuracy at the spectrum level.

While our benchmark case (see Benchmark) demonstrates the applicability of `cortecs` to high-resolution
opacity functions of molecular gas, the package is general and can be applied to any opacity data that has
pressure and temperature dependence, such as the opacity of neutral atoms or ions. Additionally, the code has
only been verified to produce reasonable amounts of error in the spectra of exoplanet atmospheres at pressures
greater than a microbar for a single composition. This caveat is important to note for a few reasons:

1. Based on error propagation, the error in the opacity function will be magnified in the spectrum based on
the number of cells that are traced during radiative transfer. The number of spatial cells used to simulate exoplanet
atmospheres (in our case, 100) is small enough that the `cortecs` error is not massive at the spectrum level.
2. Exoplanet atmospheres can justifiably be modeled as in hydrostatic equilibrium at pressures greater than a microbar.
When modeling atmospheres in hydrostatic equilibrium, the final spectrum essentially maps to the altitude at which
the gas becomes optically thick. If `cortecs`-compressed opacities were used to model an optically thin gas over
large path lengths, then smaller opacities would be more important. However, `cortecs` tends to perform worse at
modeling opacity functions that jump from very low to very high opacities, so it may not perform optimally for these
optically thin scenarios.
3. The program may perform poorly for opacity functions with sharp features in their temperature--pressure dependence.
That is, it may require so many parameters to fit the opacity function that the compression is no longer worthwhile.


# Methods
`cortecs` seeks to compress redundant information by representing opacity data not as the
opacity itself but as fits to the opacity as a function of temperature and pressure. We generally refer to this process
as ``compression'' as opposed to ``fitting'' to emphasize that we do not seek to construct physically motivated,
predictive, or comprehensible models of the opacity function. Rather, we simply seek representations of the opacity function
that consume less RAM/VRAM. The compression methods we use are ``lossy,’’ in that the original opacity data cannot be exactly recovered with our methods.
We find that the loss of accuracy is tolerable for at least the hot Jupiter application (see Benchmark below).

We provide three methods of increasing complexity (and flexibility) for
compressing and decompressing opacity: polynomial-fitting, principal components analysis [PCA, e.g., @jolliffe:2016]
and neural networks [e.g., @alzubaidi:2021]. The default neural network architecture is a fully connected
neural network; the user can specify the desired hyperparameters, such as number of layers, neurons per layer,
and activation function. Alternatively, any `keras` [@chollet:2015] model can be passed to the fitter. Each compression method is paired
with a decompression method for evaluating opacity as a function of temperature, pressure, and wavelength. These decompression methods are tailored
for GPUs and are accelerated with the `JAX` code transformation framework [@jax:2018]. An example of this reconstruction
is shown in \autoref{fig:dependencies}.

![The posterior distributions for our baseline WASP-77Ab retrieval (teal)
and our retrieval using opacities compressed by `cortecs` (gold). \label{fig:corner}](pca_compress.png)


# Workflow
A typical workflow with `cortecs` involves the following steps:

1. Acquiring: Download opacity data from a source such as the ExoMol database [@tennyson:2016] or the HITRAN database [@gordon:2017].
2. Fitting: Compress the opacity data with `cortecs`'s `fit` methods.
3. Saving: Save the compressed opacity data (the fitted parameters) to disk.
4. Loading: Load the compressed opacity data from disk in whatever program you're applying the data---e.g., within your radiative transfer code.
5. Decompression: Evaluate the opacity with `cortecs`'s `eval` methods.

The accuracy of these fits may or may not be suitable for a given application. It is important to test that
the error incurred using `cortecs` does not impact the results of your application. We provide an example of such
a benchmarking exercise below.

# Benchmark: High-resolution retrieval of WASP-77Ab
As a proof of concept, we perform a parameter inference exercise [a "retrieval", @madhusudhan:2009] on the high-resolution
thermal emission spectrum of the
fiducial hot Jupiter WASP-77Ab [@line:2021; @mansfield:2022; @august:2023] as observed at IGRINS.
The retrieval pairs `PyMultiNest` [@buchner:2014] sampling with the `CHIMERA` radiative transfer code [@line:2013].
The non-compressed retrieval uses the data and retrieval framework from [@line:2021], run in an upcoming article (Savel et al. 2024, submitted).
For this experiment, we use the PCA-based compression scheme implemented in `cortecs`, preserving 3 principal components
and their corresponding weights as a function for each wavelength as a lossy compression of the original opacity data.

Using `cortecs`, we compress the input opacity files by a factor of 13. These opacity data (cite them) were originally
stored as 2.0 GB .h5 files containing 39 temperature points, 18 pressure points, and 373,260 wavelength points. The compressed opacity data are stored
as 154 MB files of PCA coefficients and 1.1 KB files of PCA vectors (which are reused for each wavelength point).
These on-disk memory quotes are relatively faithful to the in-memory RAM footprint of the data when stored as `numpy`
arrays (2.1 GB for the uncompressed data and 160 MB for the compressed data). Reading in the original files takes
1.1 $\pm$ 0.1 seconds, while reading in the compressed files takes 24.4 $\pm$ 0.3 ms. Accessing/evaluating an opacity
value takes 174.0 $\pm$ 0.5 ns for the uncompressed data and 789 $\pm$ 5 ns for the compressed data. All of these timing
experiments are performed on a 2021 MacBook Pro with an Apple M1 Pro chip and 16 GB of RAM.


Importantly, we find that our compressed-opacity retrieval yields posterior distributions [as plotted by the `corner` package, @corner:2016]
and Bayesian evidences that are consistent with those from the benchmark
retrieval using uncompressed opacity (\autoref{fig:corner}) within a comparable runtime. The two posterior distributions exhibit
slightly different substructure, which we attribute to the compressed results requiring 10% more samples to converge and
residual differents between the compressed and uncompressed opacities.
The results from this exercise indicate that our compression/decompression scheme
is accurate enough to be used in at least some high-resolution retrievals.

![The posterior distributions for our baseline WASP-77Ab retrieval (teal)
and our retrieval using opacities compressed by `cortecs` (gold). \label{fig:corner}](pca_compress.png)


# Acknowledgements

A.B.S. and E.M-R.K. acknowledge support from the Heising-Simons Foundation. We thank Max Isi for helpful discussions.

# References
