# Research basis for the design

Each design choice in multigauge rests on published evidence. Numbers are as reported by the cited
studies; see the papers for their exact settings. Two sources are preprints or workshop papers and are
marked as such.

## 1. Architecture: the LSTM remains the best-validated choice for daily streamflow

- Liu et al. (2025, *HESS* 29, 6811) benchmarked the LSTM against 11 attention-based (Transformer-type)
  models. The LSTM was best for daily streamflow regression on 531 CAMELS basins (median KGE 0.80 vs 0.73)
  and globally (0.75 vs about 0.64). Attention models helped mainly for long-horizon autoregression and
  spatial generalisation, at much higher computational cost.
- Wang et al. (2025, arXiv:2501.14980) found a deep state-space model (S4D-FT) slightly above the LSTM on
  median NSE (0.74 vs 0.72) on CAMELS, but worse on low-flow and volume bias.
- **Choice:** a multi-basin LSTM as the core model.

## 2. Training: pre-train on many basins, then fine-tune

- Kratzert et al. (2019, *HESS* 23, 5089) established regional LSTMs trained on hundreds of basins as a
  strong benchmark.
- Kratzert et al. ("Never train an LSTM on a single basin", *HESS* opinion preprint hess-2023-275) showed
  regional LSTMs outperform single-basin LSTMs on 531 CAMELS basins, and single-basin models are far more
  likely to cap their predictions below observed floods.
- Ryd & Nearing (2025, ICLR workshop paper, arXiv:2504.12559) fine-tuned a model pre-trained on Caravan:
  mean NSE rose from 0.473 to 0.541, with the largest gains where the global model was weakest.
- Ahouandjinou et al. (2026, EGUsphere preprint, under review) found pre-training on climatically similar
  basins raised median KGE from 0.65 (local only) to 0.75.
- **Choice:** a CAMELS-US pre-trained model, fine-tuned per basin; a single-basin LSTM kept as a baseline.

## 3. Rainfall inputs: combine sources rather than choose one

- Kratzert et al. (2021, *HESS* 25, 2685) fed three gridded forcing products to one LSTM at once and raised
  median NSE from 0.77 (best single product) to 0.82; the network learned how to weight the sources.
- **Choice:** local gauges are added *alongside* gridded Daymet weather, and their value is measured as the
  gain over the gridded-only fine-tuned model. New inputs start with zero weight, so fine-tuning begins at
  the pre-trained model.

## 4. Spatial detail: sub-basins and routing for large basins

- Yu et al. (2024, *HESS* 28, 2107) ran a regional LSTM per sub-basin and routed flows with a physical
  routing model; KGE improved by about 0.11 in basins over 2,000 km² at untrained locations, with little
  change in small basins.
- Mosaffa et al. (2026, *HESS* 30, 2079) added a graph attention routing module to sub-basin LSTMs in the
  Danube; mean NSE rose from 0.46 to 0.61.
- **Choice:** phase 2 offers learned gamma routing and graph-attention routing.

## 5. Uncertainty: probabilistic output

- Nearing et al. (2024, *Nature*) describe Google's global flood model: hindcast and forecast LSTMs with a
  CMAL mixture-density head, trained on 5,680 gauges, matching GloFAS nowcast reliability at 4–5 day leads.
- Zhang et al. (2023, *HESS* 27, 4529) compared CMAL-LSTM with quantile regression forests for
  probabilistic streamflow; CMAL-LSTM was preferable for large basins and deterministic accuracy.
- **Choice:** CMAL output for all LSTMs; CRPS and 90 % interval coverage reported; phase-3 forecasting uses
  the hindcast/forecast design.

## 6. Physics-guided hybrids

- Li et al. (2025, *HESS* 29, 6829) found a differentiable HBV (δHBV) about as accurate as an LSTM in time,
  better in ungauged-region tests, and that LSTM + δHBV ensembles improved spatial generalisation
  (median NSE +0.02 to +0.07).
- **Choice:** an optional δHBV model with neural parameter learning and an LSTM + δHBV ensemble.

## What is new in multigauge

The building blocks above are published. multigauge's contribution is to combine them into one open,
reproducible workflow that answers a practical question for agencies and researchers: **how much do local
rain gauges add to a pre-trained deep-learning streamflow model, and which gauges and which parts of a basin
matter most.**

## References

- Kratzert, F., et al. (2019). Towards learning universal, regional, and local hydrological behaviors via
  machine learning applied to large-sample datasets. *HESS*, 23, 5089–5110.
- Kratzert, F., Klotz, D., Hochreiter, S., & Nearing, G. S. (2021). A note on leveraging synergy in multiple
  meteorological data sets with deep learning for rainfall–runoff modeling. *HESS*, 25, 2685–2703.
- Kratzert, F., Gauch, M., Klotz, D., & Nearing, G. HESS Opinions: Never train an LSTM on a single basin.
  *HESS Discussions* preprint hess-2023-275.
- Li, P., Song, Y., Pan, M., Lawson, K., & Shen, C. (2025). Ensembling differentiable process-based and
  data-driven models with diverse meteorological forcing datasets to advance streamflow simulation.
  *HESS*, 29, 6829–6861.
- Liu, J., Shen, C., et al. (2025). From RNNs to Transformers: benchmarking deep learning architectures for
  hydrologic prediction. *HESS*, 29, 6811.
- Mosaffa, H., Pappenberger, F., Prudhomme, C., Chantry, M., Rüdiger, C., & Cloke, H. (2026). A GNN routing
  module is all you need for LSTM rainfall–runoff models. *HESS*, 30, 2079–2092.
- Nearing, G., et al. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature*.
- Ryd, E., & Nearing, G. (2025). Fine Flood Forecasts: incorporating local data into global models through
  fine-tuning. ICLR 2025 workshop paper, arXiv:2504.12559.
- Wang, Y., Zhang, L., Yu, A., Erichson, N. B., & Yang, T. (2025). A deep state space model for
  rainfall-runoff simulations. arXiv:2501.14980.
- Yu, Q., Tolson, B. A., Shen, H., Han, M., Mai, J., & Lin, J. (2024). Enhancing long short-term memory
  (LSTM)-based streamflow prediction with a spatially distributed approach. *HESS*, 28, 2107–2122.
- Zhang, Y., et al. (2023). Comparing quantile regression forest and mixture density long short-term memory
  models for probabilistic post-processing of satellite precipitation-driven streamflow simulations.
  *HESS*, 27, 4529–4550.
- Ahouandjinou, J. E., Bossa, A. Y., Hounkpè, J., & Taormina, R. (2026). A climate similarity-based transfer
  learning framework using global Caravan dataset. *EGUsphere* preprint egusphere-2026-4749.
