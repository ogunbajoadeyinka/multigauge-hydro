"""multigauge: how much do local rain gauges add to AI streamflow models?

Pre-trained multi-basin LSTMs, fine-tuning on your basin, gauge-value
analysis, sub-basin routing, probabilistic forecasts and a physics-guided
hybrid, for any U.S. watershed.

Quick start
-----------
>>> import multigauge as mg
>>> data = mg.BasinData.from_folder("my_basin/")
>>> base = mg.load_pretrained("camels-us")
>>> result = mg.run_study(data, base)
>>> result.summary.xs("test", level="period")[["NSE", "KGE", "CRPS"]]
>>> mg.write_report(result, data, "report.html")
"""
from importlib import import_module

__version__ = "0.3.0"

# public name -> module that defines it (imported on first use, so that
# importing multigauge does not load PyTorch until a model is needed)
_EXPORTS = {
    "BasinData": "io", "DataError": "io",
    "Basin": "basin",
    "evaluate": "metrics", "evaluate_probabilistic": "metrics", "nse": "metrics", "kge": "metrics",
    "voronoi_weights": "spatial",
    "STRATEGIES": "strategies", "build_features": "strategies",
    "ModelConfig": "nn", "StreamflowModel": "nn",
    "pretrain": "transfer", "finetune": "transfer", "train_local": "transfer",
    "load_pretrained": "transfer", "available_pretrained": "results",
    "StudyConfig": "study", "StudyResult": "study", "run_study": "study", "LABELS": "results", "load_saved_study": "results",
    "GR4JModel": "baselines",
    "SubbasinSet": "routing", "RoutedModel": "routing", "run_routed_study": "routing", "auto_zones": "routing",
    "ForecastModel": "forecast", "ForecastConfig": "forecast",
    "HybridModel": "hybrid", "HybridConfig": "hybrid", "ensemble": "hybrid",
    "generate_basin": "synthetic", "generate_pretraining_set": "synthetic", "write_basin": "synthetic",
    "load_camels": "camels",
    "write_report": "report",
    "run_national": "national", "national_summary": "national",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module 'multigauge' has no attribute '{name}'")


def __dir__():
    return __all__
