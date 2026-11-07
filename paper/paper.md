---
title: 'multigauge: measuring the value of local rain gauges for pre-trained deep-learning streamflow models'
tags:
  - Python
  - hydrology
  - streamflow
  - machine learning
  - LSTM
  - rain gauges
  - flood forecasting
authors:
  - name: Adeyinka Ogunbajo
    orcid: 0009-0002-2887-0625
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 5 November 2026
bibliography: paper.bib
---

<!-- DRAFT for the author. Before submitting: (1) read every sentence and rewrite it in your own words;
(3) confirm the affiliation; (4) make the AI usage disclosure exactly true for how
you worked, and only keep the sentence that you reviewed and validated everything once you have; (5) re-check the national-survey numbers if the survey is re-run; (6) check every reference. -->

# Summary

Rivers are forecast more and more with deep-learning models trained on hundreds of basins at once. These
models are driven by gridded weather data that cover the whole country, while local rain-gauge networks run by
states, cities and agencies are usually left out. `multigauge` is an open-source Python package and web
dashboard that answers a practical question for any watershed in the conterminous United States: if the local
rain gauges are added to a state-of-the-art AI streamflow model, how much do the predictions improve, is the
improvement real, and where would a new gauge help most? Given a U.S. Geological Survey (USGS) streamgage
number, the package downloads the basin boundary, streamflow, gridded weather and rain-gauge records, fine-tunes
a pre-trained model with and without the gauges, compares the results against conventional benchmarks, and
writes an HTML report. A dashboard (https://multigauge-hydro.streamlit.app) shows pre-computed results and runs
the workflow on uploaded data without programming.

# Statement of need

Long Short-Term Memory (LSTM) networks trained on many basins are among the most accurate daily streamflow
models [@kratzert2019], are more reliable than models trained on a single basin [@kratzert2024], and underpin
operational flood forecasting [@nearing2024]. Pre-trained models can be adapted to a new basin by fine-tuning
[@ryd2025]. What is missing is a standard, reproducible way to measure the value of *local* observations on top
of such a model. Agencies that install and maintain rain gauges need this to justify networks and choose
locations, and researchers need it to compare spatial rainfall treatments fairly. Gauge-value studies that use
a weak baseline, such as a model trained on one basin, can overstate the value of gauges; studies that ignore
data quality can understate it.

`multigauge` targets hydrologists, water agencies, forecasters and students. It measures gauge value as the
change in skill of a fine-tuned, pre-trained LSTM that already uses gridded weather, the setting in which an
agency would deploy AI forecasts, and reports whether that change exceeds the variation between training runs.

# State of the field

NeuralHydrology [@kratzert2022] is the reference research framework for deep learning in hydrology. It trains
and evaluates many model types on large-sample data sets such as CAMELS [@newman2015; @addor2017], but it is a
general-purpose library configured through experiment files. It does not prepare data for an arbitrary
gauge, treat irregular rain-gauge networks as inputs, test the value of those inputs or provide a non-programmer
interface. Conceptual models such as GR4J [@perrin2003] are widely available but do not use pre-trained
deep learning. Rather than contribute these features to a general framework, `multigauge` packages one applied
question end to end: it uses a small, documented model whose inputs are available for every U.S. basin, so that
results are comparable across basins and reproducible by agencies without large-sample data sets.

# Software design

**Inputs that transfer.** The pre-trained LSTM uses five Daymet variables [@thornton2021] and nine static
attributes computed from those forcings and the basin area, so any U.S. basin can use it without an external
attribute database. Output is a countable mixture of asymmetric Laplace distributions, which gives prediction
intervals evaluated with the continuous ranked probability score [@hersbach2000].

**Gauges as neutral new inputs.** Gauge rainfall enters as additional input channels whose weights start at
zero, so fine-tuning begins exactly at the pre-trained model. Four treatments are compared: one gauge, the gauge
average, a Thiessen (Voronoi) average [@thiessen1911], and every gauge as its own input, ranked by permutation
importance [@breiman2001] and optional station ablation.

**Data quality checks.** Most NOAA cooperative observers record the gauge total in the morning, under the
day it was read; Daymet moves such reports back one day [@thornton2021], but raw GHCN-Daily records
[@menne2012] do not. `multigauge` detects each gauge's day convention by lagged correlation with basin-average
Daymet precipitation and corrects it, or uses a value the user supplies. On five validation basins this raised
the correlation between basin-average gauge rainfall and Daymet from 0.27–0.65 to 0.90–0.98.

**Honest comparisons.** Every model shares periods, gap filling and random seeds. Gauge value is computed seed
by seed and labelled as helping or hurting only when it exceeds twice its seed-to-seed spread and 0.01 in the
Nash–Sutcliffe efficiency (NSE) [@nash1970]. Benchmarks are a calibrated GR4J model, an LSTM trained on the basin
alone and the pre-trained model without fine-tuning; skill is reported as NSE, Kling–Gupta efficiency
[@gupta2009], bias, peak error, CRPS and interval coverage.

**Extensions.** Sub-basin LSTMs can be routed to the outlet with learned gamma travel-time distributions or
a graph attention network [@yu2024; @mosaffa2026], which shows where flood flow originates and which sub-basins
lack gauges. A hindcast/forecast LSTM issues 1–7-day probabilistic forecasts, and a differentiable HBV model
[@feng2022] can be combined with the LSTM [@li2025].

**Structure.** Data loading and validation (`io`, `prepare`, `download`), features and gauge strategies,
models (`nn`, `transfer`, `baselines`, `hybrid`, `routing`, `forecast`), the study runner (`study`), results
that load without PyTorch (`results`), reporting and a command-line interface are separate modules. A test
suite runs on every commit, and pre-trained models ship with the package.

# Research impact statement

The repository includes a validation on five USGS basins in five states that are outside the pre-training set
(the pre-trained model has a median test NSE of 0.68 over 531 CAMELS basins). It found that gauge value
depends on the basin and on model structure. NOAA gauges improved the fine-tuned model clearly in one basin
(Monocacy River, Maryland; NSE +0.058, about six times its seed spread) and were negligible or within noise in
the others. In three large basins, a routed four-sub-basin model raised NSE by 0.12 to 0.21 over the lumped
model with gridded weather alone, after which gauges added nothing measurable. A national survey with the same method, run on 88 basins covering all 18 U.S. hydrologic
regions with public data only, found that adding NOAA gauges raised test NSE in 80% of basins (median +0.021,
Wilcoxon p = 3 × 10⁻⁹), with the gain concentrated in rain-dominated basins (median +0.039, positive in 93%)
and absent in snow-dominated ones (\autoref{fig:map}). The day-convention check corrects a data error that otherwise makes gauge
networks look less valuable. All results, data-preparation code and notebooks needed to reproduce them are
in the repository, and the software is archived on Zenodo (https://doi.org/10.5281/zenodo.23986756).

![Change in test NSE when NOAA rain gauges are added to the fine-tuned model, 88 USGS basins.
Filled markers are changes larger than the variation between training runs.\label{fig:map}](national_map.png)

# AI usage disclosure

Generative AI was used in this project. Claude (Anthropic; Claude Opus 5.5, used through the Claude app,
October–November 2026) wrote most of the source code, tests, notebooks and documentation, and a first draft of
this paper, working from the author's instructions. The author defined the research question and scope, chose
among alternative designs proposed during development, ran the pre-training on cloud GPUs, and reviewed the
validation results. Correctness was checked by an automated test suite (unit tests and end-to-end tests on a
synthetic basin with known rainfall), by comparison of the pre-trained model with published CAMELS benchmarks and
by manual inspection of the validation results. The author has reviewed, edited and validated all AI-assisted
outputs, including the code, documentation and this manuscript, and takes full responsibility for them.
<!-- Edit this paragraph so that every statement is exactly true for how you worked. -->

# Acknowledgements

The author thanks the providers of the public data used here: the U.S. Geological Survey, the Oak Ridge National
Laboratory Distributed Active Archive Center (Daymet), the NOAA National Centers for Environmental Information
(GHCN-Daily) and the creators of the CAMELS data set. No specific funding was received.

# References
