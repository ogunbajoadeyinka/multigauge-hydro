"""multigauge: how do multiple rain gauges change LSTM streamflow prediction?

Quick start
-----------
>>> import multigauge as mg
>>> data = mg.BasinData.from_folder("my_basin/")          # or mg.generate_basin()
>>> result = mg.run_experiment(data, mg.ExperimentConfig())
>>> result.summary                                         # NSE, KGE, ... per strategy
>>> mg.write_report(result, data, "report.html")
"""

__version__ = "0.1.0"

from .io import BasinData, DataError  # noqa: E402
from .metrics import evaluate, kge, nse  # noqa: E402
from .model import StreamflowLSTM, TrainConfig  # noqa: E402
from .spatial import voronoi_weights  # noqa: E402
from .strategies import LABELS, STRATEGIES, build_features  # noqa: E402
from .experiment import ExperimentConfig, ExperimentResult, run_experiment  # noqa: E402
from .synthetic import generate_basin, write_basin  # noqa: E402
from .report import write_report  # noqa: E402

__all__ = [
    "BasinData",
    "DataError",
    "ExperimentConfig",
    "ExperimentResult",
    "LABELS",
    "STRATEGIES",
    "StreamflowLSTM",
    "TrainConfig",
    "build_features",
    "evaluate",
    "generate_basin",
    "kge",
    "nse",
    "run_experiment",
    "voronoi_weights",
    "write_basin",
    "write_report",
]
