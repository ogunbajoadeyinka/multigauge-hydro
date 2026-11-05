---
title: 'multigauge: measuring the value of local rain gauges for pre-trained deep-learning streamflow models'
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
date: November 2026
bibliography: paper.bib
---

<!-- DRAFT: revise in your own words, add your ORCID, and check every reference before submitting to JOSS. -->

# Summary

Deep-learning models trained on hundreds of basins are now the most accurate tools for daily streamflow
simulation [@kratzert2019], and pre-trained models can be fine-tuned to individual basins [@ryd2025].
Local rain-gauge networks, such as state mesonets, are not part of these large-sample models, and agencies
lack a practical way to judge how much their gauges improve predictions. `multigauge` is an open-source
Python package and web dashboard that fine-tunes a multi-basin Long Short-Term Memory (LSTM) network,
pre-trained on CAMELS-US, to a user's basin; adds the basin's rain gauges as new inputs (one gauge, a simple
average, a Thiessen/Voronoi average [@thiessen1911], or every gauge); and measures the resulting change in
deterministic (NSE, KGE; [@nash1970; @gupta2009]) and probabilistic (CRPS, interval coverage) skill against
a calibrated conceptual model (GR4J; [@perrin2003]), a single-basin LSTM and the un-tuned model. Gauges
are ranked by permutation importance [@breiman2001] and station ablation. Extensions route sub-basin LSTM
predictions to the outlet with learned travel times or a graph attention network [@yu2024; @mosaffa2026],
issue probabilistic multi-day forecasts with a hindcast/forecast LSTM [@nearing2024], and combine the LSTM
with a differentiable HBV model [@li2025].

# Statement of need

The value of a rain gauge depends on the model it feeds. Gauge-network studies typically use single-basin
models, which are known to underperform multi-basin models [@kratzert2024], so a gauge may appear valuable
only because the baseline is weak. `multigauge` measures gauge value on top of a strong, pre-trained,
fine-tuned model that already uses gridded weather, the setting in which agencies would deploy AI forecasts.
New inputs are added with zero initial weights, so fine-tuning starts exactly at the pre-trained model, and
every comparison shares periods, gap filling, model settings and random seeds. A synthetic basin generator
with known, spatially variable storms and nonlinear runoff lets users test the method where the truth is
known. Downloaders prepare USGS streamflow, NOAA gauge data and Daymet weather for any U.S. basin, and a
dashboard makes the workflow usable without programming.

The repository includes a validation on five USGS basins outside the training set (median CAMELS-US
test NSE of the pre-trained model 0.68). Two practical results motivate the design. First, most NOAA
cooperative gauges report morning readings under the following day, out of step with Daymet; the
package detects and corrects each gauge's day convention, without which measured gauge value is not
meaningful. Second, gauge value depends on model structure: in a 9,400 km² Oklahoma basin, gauges
added little to a lumped model but raised a routed sub-basin model from NSE 0.74 to 0.85, and the
sub-basin model identified the ungauged area producing most of the peak flow.

The package complements NeuralHydrology [@kratzert2022], a general research framework for large-sample deep
learning in hydrology, by packaging one applied question, the value of local rain gauges and of spatial
detail for pre-trained models, as an end-to-end, reproducible workflow for researchers and practitioners.

# Availability

The source code, pre-trained models and validation results are at
https://github.com/ogunbajoadeyinka/multigauge-hydro (MIT licence) and archived on Zenodo
(https://doi.org/10.5281/zenodo.23986756). A web dashboard is available at https://multigauge-hydro.streamlit.app.

# Acknowledgements

<!-- Add advisor, committee, funding and data providers as appropriate. -->

# References
