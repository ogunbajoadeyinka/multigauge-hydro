---
title: 'multigauge: a Python package for assessing how multiple rain gauges affect LSTM streamflow prediction'
tags:
  - Python
  - hydrology
  - streamflow
  - machine learning
  - LSTM
  - rain gauges
authors:
  - name: Adeyinka Ogunbajo
    affiliation: 1
affiliations:
  - name: Department of Biosystems and Agricultural Engineering, Oklahoma State University, USA
    index: 1
date: 3 November 2026
bibliography: paper.bib
---

<!-- DRAFT: revise in your own words, add your ORCID, and check every reference before submitting to JOSS. -->

# Summary

Long Short-Term Memory (LSTM) networks are now among the most accurate tools for daily streamflow
prediction [@kratzert2018; @kratzert2019]. Most applications, however, feed the model a single
basin-averaged rainfall series, discarding information about where rain fell. `multigauge` is an
open-source Python package and web dashboard that, for any watershed with several rain gauges,
trains LSTMs on four rainfall representations (a single gauge, a simple average, a Thiessen/Voronoi
area-weighted average [@thiessen1911], and every gauge as a separate input), evaluates them with
standard hydrological metrics [@nash1970; @gupta2009], and ranks gauges by permutation importance
[@breiman2001] and station ablation.

# Statement of need

Whether spatially distributed rainfall improves data-driven streamflow models, and which gauges
provide that improvement, is a practical question for researchers, forecasters and the agencies that
fund monitoring networks. Answering it currently requires custom data preparation, spatial weighting,
repeated model training and importance analysis for each basin. In a study of the Washita River basin,
Oklahoma, using ten Oklahoma Mesonet gauges, the disaggregated representation gave the highest test
Nash-Sutcliffe Efficiency and the lumped average the lowest, motivating a reusable tool for testing
the effect elsewhere.

`multigauge` packages this workflow so it can be repeated on any basin with a few input files and a
single command, a few lines of Python, or a browser upload. It controls the factors that commonly
confound such comparisons: all representations share periods, gap filling, model settings and random
seeds; normalisation uses training-period statistics only; the single gauge is selected on the
validation period; and multiple seeds quantify training variability. A synthetic basin generator with
known, spatially heterogeneous storms and runoff supports testing and teaching. Helpers download U.S.
streamflow (USGS) and precipitation (NOAA GHCN-Daily) in the expected format.

The package complements NeuralHydrology [@kratzert2022], which provides a broad framework for
large-sample LSTM hydrology, by focusing on one question, the value of individual rain gauges and
their spatial arrangement, with an end-to-end workflow and reporting aimed at practitioners.

# Acknowledgements

<!-- Add advisor, committee, funding and data providers as appropriate. -->

# References
