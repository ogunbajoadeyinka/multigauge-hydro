# multigauge v0.3.0: a pre-trained deep-learning workflow for measuring the value of local rain gauges in streamflow prediction

**Technical description and evaluation** · Adeyinka Ogunbajo, Independent Researcher, United States ·
Software archive: https://doi.org/10.5281/zenodo.23986756 · Code: https://github.com/ogunbajoadeyinka/multigauge-hydro (MIT licence)

## Abstract

multigauge is an open-source Python package and web dashboard that answers one question for any watershed in
the conterminous United States: does adding local rain-gauge records improve a pre-trained deep-learning
streamflow model, by how much, and is the improvement larger than the noise of training? Given a U.S.
Geological Survey (USGS) streamgage, the package downloads the basin boundary, daily streamflow, gridded Daymet
weather and NOAA rain-gauge records; corrects the gauges' day convention; fine-tunes a long short-term memory
(LSTM) network pre-trained on 531 basins with and without the gauges; compares the results against a calibrated
conceptual model and other benchmarks; and reports the gain seed by seed. Optional modules route sub-basin
models to the outlet and issue 1–7-day probabilistic forecasts. This document describes the equations and
numerical choices in enough detail to re-implement them, the verification tests, and the evaluation on a
531-basin benchmark, four detailed basins, three sub-basin studies and an 88-basin national survey.

## 1 Introduction

LSTM networks trained on hundreds of basins are among the most accurate daily rainfall–runoff models
(Kratzert et al., 2019) and are more reliable than models trained on a single basin (Kratzert et al., 2024).
They are usually driven by gridded weather. Local rain-gauge networks sample storms directly but are rarely used
as inputs, and there is no standard way to measure what they add on top of a strong, pre-trained model.
multigauge provides that measurement as a reproducible workflow: identical periods, gap filling, settings and
random seeds for every model; benchmarks that a gauge-value claim must beat; and an explicit test of whether
each gain exceeds the variation between training runs.

## 2 Model description

### 2.1 Workflow

1. **Prepare** a basin folder from public data (`prepare_usgs_basin`), or load the user's own files.
2. **Align** each gauge to the Daymet day convention (Section 2.3).
3. **Build** gauge inputs in four forms (Section 2.4).
4. **Fine-tune** the pre-trained LSTM once without gauges and once per gauge form, for each seed (Sections 2.5–2.6).
5. **Benchmark** against GR4J, a basin-only LSTM, the un-tuned pre-trained model and, optionally, a differentiable
   HBV model (Section 2.7).
6. **Evaluate** on a held-out test period and compute gauge value with its seed spread (Sections 2.10–2.11).
7. Optionally **route** sub-basins (Section 2.8) and **forecast** (Section 2.9). Results are written as CSV files and
   an HTML report and can be browsed in the dashboard.

### 2.2 Data and pre-processing

Dynamic inputs are the Daymet version 4 daily variables precipitation $P$ (mm d⁻¹), shortwave radiation
(W m⁻²), maximum and minimum temperature $T_{\max}$, $T_{\min}$ (°C) and vapour pressure (Pa) (Thornton et al.,
2021), averaged over 16 points spread over the basin. Streamflow $Q$ is converted to a depth over the basin,

$$q_t = \frac{86.4\,Q_t}{A},$$

with $Q_t$ in m³ s⁻¹, basin area $A$ in km² and $q_t$ in mm d⁻¹. Potential evapotranspiration is estimated with
the Hargreaves–Samani equation,

$$\mathrm{PET}_t = 0.0023 \cdot 0.408\, R_{a,t}\, (\bar T_t + 17.8)\, \sqrt{T_{\max,t} - T_{\min,t}},$$

where $R_{a,t}$ is extraterrestrial radiation (MJ m⁻² d⁻¹) for the basin latitude and day of year, $\bar T_t$ the
mean daily temperature and 0.408 converts energy to an equivalent water depth. Nine static attributes are computed
from the forcings so that any basin can be used without an attribute database: $\log_{10} A$, latitude, mean
precipitation $\bar P$, mean PET, aridity $\overline{\mathrm{PET}}/\bar P$, mean temperature, snow fraction (share
of precipitation on days with $\bar T_t < 0$ °C), and the frequencies (days per year) of high-precipitation days
($P_t \ge 5\bar P$) and dry days ($P_t < 1$ mm), following the CAMELS definitions (Addor et al., 2017). All inputs
are standardised with means and standard deviations from the pre-training basins.

Rain gauges are NOAA GHCN-Daily stations (Menne et al., 2012) inside the basin or within a buffer of its boundary
(5 km by default) whose records cover at least 80 % of the years and 70 % of the days of the study period; at
most 15 are kept, longest records first.

### 2.3 Gauge day alignment

Cooperative observers usually read gauges in the morning and report the total under the day of reading, so most
of that rain fell the previous day; Daymet assigns rain to the day it fell (Thornton et al., 2021). For each gauge
$i$ with record $g_{i,t}$ and basin Daymet precipitation $P_t$, the correlation at lag $k$ is

$$r_i(k) = \mathrm{corr}\big(g_{i,t+k},\, P_t\big), \qquad k \in \{-1, 0, +1\}.$$

The record is shifted by $k^\ast = \arg\max_k r_i(k)$ when $r_i(k^\ast) - r_i(0) \ge 0.1$ and the gauge has at least
365 days of data; otherwise it is left unchanged. Users can set the shift per gauge (`day_shift`).

### 2.4 Gauge inputs

With $G$ gauges and their records after alignment, the four gauge forms added as inputs are

- **single gauge:** $x_t = g_{j,t}$ for one gauge $j$ (the most important gauge by permutation, or the user's choice);
- **gauge average:** $x_t = \frac{1}{|R_t|} \sum_{i \in R_t} g_{i,t}$, where $R_t$ is the set of gauges reporting on day $t$;
- **Thiessen (Voronoi) average:** $x_t = \sum_{i \in R_t} \tilde w_i\, g_{i,t}$, with $w_i$ the share of the basin area closest to gauge $i$ (Thiessen, 1911) and $\tilde w_i = w_i / \sum_{j \in R_t} w_j$ re-normalised over the reporting gauges;
- **all gauges:** one input per gauge, $\mathbf{x}_t = (g_{1,t}, \dots, g_{G,t})$, with each missing value filled by the Thiessen average of the gauges reporting that day.

Gap filling is identical for every model of a study.

### 2.5 LSTM with a probabilistic output

For each day $t$, the input vector $\mathbf{u}_t$ concatenates the standardised dynamic inputs, the static
attributes and any gauge inputs. The LSTM (Hochreiter and Schmidhuber, 1997) updates its cell state
$\mathbf{c}_t$ and hidden state $\mathbf{h}_t$ (128 units) as

$$ \begin{aligned} \mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{u}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i), & \mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{u}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f), \\ \mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{u}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o), & \tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c \mathbf{u}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c), \\ \mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t, & \mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t), \end{aligned}$$

run over a 365-day window ending on the day predicted. The last hidden state, after dropout (rate 0.4), feeds a
countable mixture of $K = 3$ asymmetric Laplace distributions (CMAL; Klotz et al., 2022) for the standardised flow
$y$:

$$p(y) = \sum_{k=1}^{K} \pi_k\, \frac{\tau_k (1-\tau_k)}{b_k}\, \exp\!\left[-\rho_{\tau_k}\!\left(\frac{y-\mu_k}{b_k}\right)\right], \qquad \rho_\tau(u) = u\,\big(\tau - \mathbb{1}[u<0]\big),$$

where the location $\mu_k$, scale $b_k > 0$ (softplus), asymmetry $\tau_k \in (0.01, 0.99)$ (scaled sigmoid) and
weights $\pi_k$ (softmax) are linear functions of $\mathbf{h}_t$. Training minimises the negative log-likelihood
$-\log p(y_t)$. The point prediction is the mixture mean

$$\hat y_t = \sum_k \pi_k \left[\mu_k + b_k \frac{1 - 2\tau_k}{\tau_k (1-\tau_k)}\right],$$

and prediction intervals (5 %, 50 % and 95 % quantiles) are taken from samples of the mixture.

### 2.6 Pre-training and fine-tuning

The model was pre-trained on the 531-basin CAMELS subset (Newman et al., 2015) for October 1980 to September 1995
with Adam (learning rate 10⁻³, batch size 256, 200 000 random training windows per epoch, at most 30 epochs),
keeping the epoch with the best median validation Nash–Sutcliffe efficiency (NSE) on 1995–2000 (patience 6).

For a new basin the pre-trained weights are copied and fine-tuned on the basin's own training period (learning
rate 2 × 10⁻⁴, at most 20 epochs, early stopping on the validation period). Gauge inputs are added by enlarging
the input matrices $\mathbf{W}_{\{i,f,o,c\}}$ with new columns **initialised to zero**, so that at the start of
fine-tuning the model with gauges gives exactly the same output as the model without them; any difference after
fine-tuning is learned from the gauges.

### 2.7 Benchmarks

- **GR4J** (Perrin et al., 2003), a four-parameter conceptual model, with a degree-day snow routine, calibrated by
  differential evolution on the training period with the same inputs.
- **Basin-only LSTM:** the same architecture trained from scratch on the basin alone (Kratzert et al., 2024).
- **Pre-trained, not fine-tuned:** the pre-trained model applied as is, a measure of transfer.
- **δHBV** (optional): a differentiable HBV model whose parameters are predicted by a neural network (Feng et al.,
  2022), and an ensemble of the LSTM and δHBV.

### 2.8 Sub-basin routing

A basin is split into $N$ sub-basins (by default four bands of distance from the outlet), each with its own Daymet
forcings $\mathbf{u}^{(n)}_t$, area $A_n$ and, optionally, its own gauges. The fine-tuned LSTM predicts local runoff
$\hat q^{(n)}_t$ for each, and outlet flow is the area-weighted sum of each sub-basin's runoff convolved with a
learned gamma travel-time kernel:

$$\hat q_t = \sum_{n=1}^{N} \frac{A_n}{A} \sum_{s=0}^{W-1} h_n(s)\, \hat q^{(n)}_{t-s}, \qquad h_n(s) \propto \frac{(s+\tfrac12)^{\kappa_n - 1}\, e^{-(s+\frac12)/\theta_n}}{\Gamma(\kappa_n)\, \theta_n^{\kappa_n}}, \quad \sum_s h_n(s) = 1,$$

with shape $\kappa_n$ and mean travel time $\kappa_n\theta_n$ learned per sub-basin (initialised from the
sub-basin's distance to the outlet at 40 km d⁻¹) over a window of $W$ days. LSTM and routing parameters are trained
jointly on outlet streamflow. Each sub-basin's share of outlet flow, overall and on peak-flow days, is reported to
show where flood runoff comes from. A graph-attention routing option is also provided.

### 2.9 Forecasts

A hindcast LSTM, initialised from the fine-tuned model, runs over the past 365 days and hands its states to a
forecast LSTM that runs over the next $H = 7$ days on forecast weather, with a CMAL output. In the evaluation,
forecast weather is observed weather with errors that grow with lead time (log-normal precipitation error with
σ = 0.15 per lead day; temperature error 0.6 °C per lead day). The latest observed flow updates the forecast through
a per-lead error correction

$$\hat q_{t+h} \leftarrow \hat q_{t+h} + \phi_h \,(q_t - \hat q_t), \qquad \phi_h = \mathrm{clip}\!\left(\frac{\mathrm{cov}(e_t, e_{t+h})}{\mathrm{var}(e_t)},\, 0,\, 0.99\right),$$

with $e_t$ the simulation error on training days. Skill is compared with persistence, $\hat q_{t+h} = q_t$.

### 2.10 Evaluation metrics

With observations $q_t$, predictions $\hat q_t$ and means $\bar q$, $\bar{\hat q}$ over the test days:

$$\mathrm{NSE} = 1 - \frac{\sum_t (\hat q_t - q_t)^2}{\sum_t (q_t - \bar q)^2} \quad \text{(Nash and Sutcliffe, 1970)},$$

$$\mathrm{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}, \quad \alpha = \frac{\sigma_{\hat q}}{\sigma_q}, \quad \beta = \frac{\bar{\hat q}}{\bar q} \quad \text{(Gupta et al., 2009)},$$

$$\mathrm{PBIAS} = 100\, \frac{\sum_t (\hat q_t - q_t)}{\sum_t q_t}, \qquad \mathrm{Peak\ error} = 100\, \overline{\left(\frac{\hat q_t - q_t}{q_t}\right)}_{\,q_t \ge q_{0.98}},$$

where $r$ is the correlation and $q_{0.98}$ the 98th percentile of observed flow. Probabilistic skill is the
continuous ranked probability score (Hersbach, 2000), approximated from the predicted quantiles $\hat q^{(\ell)}_t$ at
levels $\ell \in L$ as twice the mean pinball loss,

$$\mathrm{CRPS} \approx \frac{2}{|L|} \sum_{\ell \in L} \overline{\big(\mathbb{1}[q_t < \hat q^{(\ell)}_t] - \ell\big)\big(\hat q^{(\ell)}_t - q_t\big)},$$

and the 90 % interval coverage is the share of days with $q_t$ between the 5 % and 95 % quantiles.

### 2.11 Gauge value and verdict

For gauge form $m$ and seed $s$, gauge value is the change in test NSE relative to the fine-tuned model without
gauges trained with the same seed,

$$\Delta_{m,s} = \mathrm{NSE}_{m,s} - \mathrm{NSE}_{0,s}, \qquad \bar\Delta_m = \frac{1}{S}\sum_s \Delta_{m,s}, \qquad \sigma_m = \mathrm{sd}_s(\Delta_{m,s}).$$

The gain is called **helps** (or **hurts**) when $|\bar\Delta_m| > 2\sigma_m$ and $|\bar\Delta_m| \ge 0.01$,
**negligible** when it exceeds $2\sigma_m$ but is smaller than 0.01, and **within noise** otherwise. The
contribution of each gauge in the all-gauge model is ranked by permutation importance: the drop in validation NSE
when that gauge's input is randomly reordered in time (Breiman, 2001). Across many basins, the national survey
tests whether the median gain differs from zero with the Wilcoxon signed-rank test and relates it to basin
attributes with Spearman's rank correlation.

## 3 Verification

An automated test suite (41 tests, run on every change) checks unit conversions, the PET and attribute equations,
day alignment on synthetic gauges with a known one-day lag, Thiessen weights, gap filling, the zero-initialised
input expansion (identical output before fine-tuning), saving and loading of models and results, period handling,
the national-survey runner and the dashboard's loading of saved results. A synthetic basin generator with spatially
variable storms and known rainfall provides an end-to-end test in which gauges must add skill: on the demo basin,
fine-tuning with all gauges raises test NSE from 0.823 to 0.848, while GR4J reaches 0.591.

## 4 Evaluation

**Benchmark.** On the standard CAMELS test years (October 2000 to September 2010) the pre-trained model has a median
NSE of 0.68, a median KGE of 0.70 and a median 90 % interval coverage of 0.91 across the 531 basins, slightly below
the best published single LSTMs (Kratzert et al., 2019), which use more static attributes.

**National survey.** On 88 USGS basins in all 18 U.S. hydrologic regions (test October 2017 to September 2023),
adding every NOAA gauge as an input raised NSE in 80 % of basins (median +0.021, Wilcoxon p = 3 × 10⁻⁹). The gain
was +0.039 in rain-dominated basins (snow < 10 % of precipitation; positive in 93 %), +0.017 in mixed basins and
−0.001 in snow-dominated ones, and it was largest where the model without gauges was weakest (Spearman ρ = −0.45).
83 % of basins contained a gauge that needed the one-day alignment.

**Sub-basins.** In three large basins (3,336–7,185 km²), four routed sub-basins with Daymet alone raised NSE by
0.12–0.21 over the whole-basin model; adding gauges then changed NSE by +0.001 to +0.006, within seed noise.

**Forecasts.** On the Iowa River the model beats persistence from day 3 and keeps NSE 0.67 at day 7 (persistence
0.46); on the flashier Flint River it beats persistence from day 1 and holds NSE 0.83 at day 7 (persistence −0.53).

Full results, with every number traceable to a file, are in `docs/validation_report.md` and can be recomputed with
`scripts/check_numbers.py`.

## 5 Scope and limitations

- Applies to USGS basins in the conterminous United States (where Daymet, NLDI and GHCN-Daily are available) and,
  with user-supplied files, to any basin with daily data.
- The pre-trained model was trained on basins under 2,000 km²; for larger basins, sub-basin routing is recommended.
- NOAA gauges also inform Daymet, so measured gauge value is conservative for independent networks (state mesonets,
  city networks, radar products), which users can load as their own gauge files.
- Forecast skill is evaluated with perturbed observations, not archived weather forecasts.
- Per-basin verdicts depend on the number of seeds; two seeds (national survey) give a rough spread, three or more are
  recommended for single-basin decisions.

## 6 Software, requirements and licence

Python ≥ 3.9 with NumPy, pandas, SciPy, Shapely, PyTorch and Plotly; Streamlit for the dashboard. Runs on a laptop
CPU (a single basin study takes about 1–2 h) or a free cloud GPU (Kaggle, Colab). Installation:
`pip install -e ".[download]"` from the repository. MIT licence.

## Code availability

The exact version described here (v0.3.0), the pre-trained models, scripts, notebooks and all results are archived at
https://doi.org/10.5281/zenodo.23986756 and developed at https://github.com/ogunbajoadeyinka/multigauge-hydro. The
dashboard is at https://multigauge-hydro.streamlit.app.

## References

- Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293–5313, https://doi.org/10.5194/hess-21-5293-2017, 2017.
- Breiman, L.: Random forests, Mach. Learn., 45, 5–32, https://doi.org/10.1023/A:1010933404324, 2001.
- Feng, D., Liu, J., Lawson, K., and Shen, C.: Differentiable, learnable, regionalized process-based models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy, Water Resour. Res., 58, e2022WR032404, https://doi.org/10.1029/2022WR032404, 2022.
- Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition of the mean squared error and NSE performance criteria: implications for improving hydrological modelling, J. Hydrol., 377, 80–91, https://doi.org/10.1016/j.jhydrol.2009.08.003, 2009.
- Hersbach, H.: Decomposition of the continuous ranked probability score for ensemble prediction systems, Weather Forecast., 15, 559–570, https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2, 2000.
- Hochreiter, S. and Schmidhuber, J.: Long short-term memory, Neural Comput., 9, 1735–1780, https://doi.org/10.1162/neco.1997.9.8.1735, 1997.
- Klotz, D., Kratzert, F., Gauch, M., Keefe Sampson, A., Brandstetter, J., Klambauer, G., Hochreiter, S., and Nearing, G.: Uncertainty estimation with deep learning for rainfall–runoff modeling, Hydrol. Earth Syst. Sci., 26, 1673–1693, https://doi.org/10.5194/hess-26-1673-2022, 2022.
- Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets, Hydrol. Earth Syst. Sci., 23, 5089–5110, https://doi.org/10.5194/hess-23-5089-2019, 2019.
- Kratzert, F., Gauch, M., Klotz, D., and Nearing, G.: HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin, Hydrol. Earth Syst. Sci., 28, 4187–4201, https://doi.org/10.5194/hess-28-4187-2024, 2024.
- Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E., and Houston, T. G.: An overview of the Global Historical Climatology Network-Daily database, J. Atmos. Ocean. Tech., 29, 897–910, https://doi.org/10.1175/JTECH-D-11-00103.1, 2012.
- Nash, J. E. and Sutcliffe, J. V.: River flow forecasting through conceptual models part I — A discussion of principles, J. Hydrol., 10, 282–290, https://doi.org/10.1016/0022-1694(70)90255-6, 1970.
- Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., Viger, R. J., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T., and Duan, Q.: Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA, Hydrol. Earth Syst. Sci., 19, 209–223, https://doi.org/10.5194/hess-19-209-2015, 2015.
- Perrin, C., Michel, C., and Andréassian, V.: Improvement of a parsimonious model for streamflow simulation, J. Hydrol., 279, 275–289, https://doi.org/10.1016/S0022-1694(03)00225-7, 2003.
- Thiessen, A. H.: Precipitation averages for large areas, Mon. Weather Rev., 39, 1082–1089, 1911.
- Thornton, P. E., Shrestha, R., Thornton, M., Kao, S.-C., Wei, Y., and Wilson, B. E.: Gridded daily weather data for North America with comprehensive uncertainty quantification, Sci. Data, 8, 190, https://doi.org/10.1038/s41597-021-00973-0, 2021.
