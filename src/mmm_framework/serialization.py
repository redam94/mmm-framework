"""Model serialization utilities for BayesianMMM.

This module provides functions to save and load BayesianMMM models,
separating the serialization logic from the main model class.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np

if TYPE_CHECKING:
    from .data_loader import PanelDataset
    from .model import BayesianMMM, TrendConfig


class MMMSerializer:
    """Handles save/load operations for BayesianMMM models.

    This class encapsulates all serialization logic, keeping the main
    BayesianMMM class focused on modeling.

    Examples
    --------
    >>> # Save a model
    >>> MMMSerializer.save(model, "models/my_mmm")

    >>> # Load a model
    >>> model = MMMSerializer.load("models/my_mmm", panel)
    """

    # Version for saved model format
    _FORMAT_VERSION = "1.0"

    @classmethod
    def save(
        cls,
        model: BayesianMMM,
        path: str | Path,
        save_trace: bool = True,
        compress: bool = True,
    ) -> None:
        """
        Save a BayesianMMM model to disk.

        Parameters
        ----------
        model : BayesianMMM
            The model instance to save.
        path : str or Path
            Directory path where the model will be saved.
        save_trace : bool, default True
            Whether to save the fitted trace. Set to False for a smaller
            save file if you only need configurations.
        compress : bool, default True
            Whether to compress the trace file with gzip.

        Raises
        ------
        ValueError
            If the model has no trace and save_trace is True.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata
        metadata = cls._collect_metadata(model)
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 2. Save configurations
        configs = cls._collect_configs(model)
        with open(path / "configs.json", "w") as f:
            json.dump(configs, f, indent=2, default=str)

        # 3. Save scaling parameters
        scaling_params = cls._collect_scaling_params(model)
        with open(path / "scaling_params.json", "w") as f:
            json.dump(scaling_params, f, indent=2)

        # 4. Save trace (if fitted and requested)
        if save_trace and model._trace is not None:
            cls._save_trace(model._trace, path, compress)

        # 5. Save trend features if they exist
        cls._save_trend_features(model, path)

        # 6. Save seasonality features
        cls._save_seasonality_features(model, path)

        print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        panel: PanelDataset,
        rebuild_model: bool = True,
    ) -> BayesianMMM:
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : str or Path
            Directory path where the model was saved.
        panel : PanelDataset
            Panel data to use with the loaded model. Must be compatible
            with the original data (same channels, controls, dimensions).
        rebuild_model : bool, default True
            Whether to rebuild the PyMC model. Set to False if you only
            need access to the trace and don't need to make predictions.

        Returns
        -------
        BayesianMMM
            Loaded model instance with fitted trace (if available).

        Raises
        ------
        ValueError
            If the panel data is incompatible with the saved model.
        FileNotFoundError
            If the model files are not found.
        """
        # Import here to avoid circular imports
        from .model import BayesianMMM, TrendConfig, geometric_adstock_2d

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        # 1. Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Version check
        cls._check_version(metadata)

        # 2. Load configurations
        with open(path / "configs.json", "r") as f:
            configs = json.load(f)

        # Reconstruct configs
        from .config import ModelConfig

        model_config = ModelConfig(**configs["model_config"])
        trend_config = TrendConfig.from_dict(configs["trend_config"])

        # 3. Validate panel compatibility
        cls._validate_panel_compatibility(panel, metadata)

        # 4. Create instance
        adstock_alphas = metadata.get("adstock_alphas", [0.0, 0.3, 0.5, 0.7, 0.9])

        instance = BayesianMMM(
            panel=panel,
            model_config=model_config,
            trend_config=trend_config,
            adstock_alphas=adstock_alphas,
        )

        # 5. Load scaling parameters
        with open(path / "scaling_params.json", "r") as f:
            scaling_params = json.load(f)

        cls._restore_scaling_params(instance, scaling_params)

        # Re-standardize y with loaded params
        instance.y = (instance.y_raw - instance.y_mean) / instance.y_std

        # Re-normalize media with loaded max values
        for alpha in instance.adstock_alphas:
            adstocked = geometric_adstock_2d(instance.X_media_raw, alpha)
            normalized = np.zeros_like(adstocked)
            for c, ch_name in enumerate(instance.channel_names):
                normalized[:, c] = adstocked[:, c] / (
                    instance._media_max[ch_name] + 1e-8
                )
            instance.X_media_adstocked[alpha] = normalized

        # Re-standardize controls with loaded params
        if instance.X_controls_raw is not None and "control_mean" in scaling_params:
            instance.X_controls = (
                instance.X_controls_raw - instance.control_mean
            ) / instance.control_std

        # 6. Load trend features if present
        cls._load_trend_features(instance, path)

        # 7. Load seasonality features if present
        cls._load_seasonality_features(instance, path)

        # 8. Load trace (if available)
        trace = cls._load_trace(path)
        if trace is not None:
            instance._trace = trace

        # 9. Build model if requested
        if rebuild_model:
            instance._model = instance._build_model()

        print(f"Model loaded from {path}")
        if instance._trace is not None:
            print(
                f"  Trace loaded: {instance._trace.posterior.dims['chain']} chains, "
                f"{instance._trace.posterior.dims['draw']} draws"
            )

        return instance

    @classmethod
    def save_trace_only(
        cls,
        trace: az.InferenceData,
        path: str | Path,
    ) -> None:
        """
        Save only a trace to a file.

        Parameters
        ----------
        trace : az.InferenceData
            The trace to save.
        path : str or Path
            File path for the trace (should end in .nc or .nc.gz).
        """
        path = Path(path)

        if str(path).endswith(".gz"):
            # Save compressed
            base_path = Path(str(path)[:-3])  # Remove .gz
            trace.to_netcdf(str(base_path))

            with open(base_path, "rb") as f_in:
                with gzip.open(path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            base_path.unlink()
        else:
            trace.to_netcdf(str(path))

        print(f"Trace saved to {path}")

    @classmethod
    def load_trace_only(cls, path: str | Path) -> az.InferenceData:
        """
        Load a trace from a file.

        Parameters
        ----------
        path : str or Path
            File path to the trace (.nc or .nc.gz).

        Returns
        -------
        az.InferenceData
            The loaded trace.
        """
        path = Path(path)

        if str(path).endswith(".gz"):
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name

            with gzip.open(path, "rb") as f_in:
                with open(tmp_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            trace = az.from_netcdf(tmp_path)
            os.unlink(tmp_path)
        else:
            trace = az.from_netcdf(str(path))

        print(f"Trace loaded from {path}")
        return trace

    # =========================================================================
    # Private helper methods
    # =========================================================================

    @classmethod
    def _collect_metadata(cls, model: BayesianMMM) -> dict[str, Any]:
        """Collect model metadata for saving."""
        metadata = {
            "version": model._VERSION,
            "format_version": cls._FORMAT_VERSION,
            "n_obs": model.n_obs,
            "n_channels": model.n_channels,
            "n_controls": model.n_controls,
            "n_time_periods": model.n_time_periods,
            "channel_names": model.channel_names,
            "control_names": model.control_names,
            "has_geo": model.has_geo,
            "has_product": model.has_product,
            "adstock_alphas": model.adstock_alphas,
        }

        if model.has_geo:
            metadata["geo_names"] = model.geo_names
        if model.has_product:
            metadata["product_names"] = model.product_names

        return metadata

    @classmethod
    def _collect_configs(cls, model: BayesianMMM) -> dict[str, Any]:
        """Collect model configurations for saving."""
        return {
            "model_config": model.model_config.model_dump(),
            "trend_config": model.trend_config.to_dict(),
            "mff_config": model.mff_config.model_dump(),
        }

    @classmethod
    def _collect_scaling_params(cls, model: BayesianMMM) -> dict[str, Any]:
        """Collect scaling parameters for saving."""
        scaling_params = {
            "y_mean": model.y_mean,
            "y_std": model.y_std,
            "media_max": {k: float(v) for k, v in model._media_max.items()},
        }

        if model.X_controls_raw is not None:
            scaling_params["control_mean"] = model.control_mean.tolist()
            scaling_params["control_std"] = model.control_std.tolist()

        return scaling_params

    @classmethod
    def _save_trace(
        cls,
        trace: az.InferenceData,
        path: Path,
        compress: bool,
    ) -> None:
        """Save trace to disk, optionally compressed."""
        trace_path = path / "trace.nc"
        trace.to_netcdf(str(trace_path))

        if compress:
            with open(trace_path, "rb") as f_in:
                with gzip.open(str(trace_path) + ".gz", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            trace_path.unlink()  # Remove uncompressed file

    @classmethod
    def _save_trend_features(cls, model: BayesianMMM, path: Path) -> None:
        """Save trend features if they exist."""
        trend_features_to_save = {}
        for key, value in model.trend_features.items():
            if isinstance(value, np.ndarray):
                trend_features_to_save[key] = value.tolist()
            elif isinstance(value, dict):
                trend_features_to_save[key] = value
            else:
                trend_features_to_save[key] = value

        if trend_features_to_save:
            with open(path / "trend_features.json", "w") as f:
                json.dump(trend_features_to_save, f, indent=2)

    @classmethod
    def _save_seasonality_features(cls, model: BayesianMMM, path: Path) -> None:
        """Save seasonality features."""
        season_features_to_save = {}
        for key, value in model.seasonality_features.items():
            if isinstance(value, np.ndarray):
                season_features_to_save[key] = value.tolist()

        if season_features_to_save:
            with open(path / "seasonality_features.json", "w") as f:
                json.dump(season_features_to_save, f, indent=2)

    @classmethod
    def _check_version(cls, metadata: dict[str, Any]) -> None:
        """Check version compatibility and warn if needed."""
        from .model import BayesianMMM

        saved_version = metadata.get("version", "0.0.0")
        if saved_version != BayesianMMM._VERSION:
            warnings.warn(
                f"Model was saved with version {saved_version}, "
                f"current version is {BayesianMMM._VERSION}. "
                "There may be compatibility issues."
            )

    @classmethod
    def _validate_panel_compatibility(
        cls,
        panel: PanelDataset,
        metadata: dict[str, Any],
    ) -> None:
        """Validate that the panel is compatible with the saved model."""
        if panel.coords.channels != metadata["channel_names"]:
            raise ValueError(
                f"Panel channels {panel.coords.channels} don't match "
                f"saved model channels {metadata['channel_names']}"
            )

        if panel.coords.controls != metadata["control_names"]:
            raise ValueError(
                f"Panel controls {panel.coords.controls} don't match "
                f"saved model controls {metadata['control_names']}"
            )

    @classmethod
    def _restore_scaling_params(
        cls,
        instance: BayesianMMM,
        scaling_params: dict[str, Any],
    ) -> None:
        """Restore scaling parameters to the model instance."""
        instance.y_mean = scaling_params["y_mean"]
        instance.y_std = scaling_params["y_std"]
        instance._media_max = scaling_params["media_max"]
        instance._scaling_params["y_mean"] = scaling_params["y_mean"]
        instance._scaling_params["y_std"] = scaling_params["y_std"]
        instance._scaling_params["media_max"] = scaling_params["media_max"]

        if "control_mean" in scaling_params:
            instance.control_mean = np.array(scaling_params["control_mean"])
            instance.control_std = np.array(scaling_params["control_std"])
            instance._scaling_params["control_mean"] = instance.control_mean
            instance._scaling_params["control_std"] = instance.control_std

    @classmethod
    def _load_trend_features(cls, instance: BayesianMMM, path: Path) -> None:
        """Load trend features if present."""
        trend_features_path = path / "trend_features.json"
        if trend_features_path.exists():
            with open(trend_features_path, "r") as f:
                trend_features = json.load(f)

            # Convert lists back to numpy arrays
            for key, value in trend_features.items():
                if isinstance(value, list):
                    instance.trend_features[key] = np.array(value)
                else:
                    instance.trend_features[key] = value

    @classmethod
    def _load_seasonality_features(cls, instance: BayesianMMM, path: Path) -> None:
        """Load seasonality features if present."""
        season_features_path = path / "seasonality_features.json"
        if season_features_path.exists():
            with open(season_features_path, "r") as f:
                season_features = json.load(f)

            for key, value in season_features.items():
                if isinstance(value, list):
                    instance.seasonality_features[key] = np.array(value)

    @classmethod
    def _load_trace(cls, path: Path) -> az.InferenceData | None:
        """Load trace if available."""
        trace_path_gz = path / "trace.nc.gz"
        trace_path = path / "trace.nc"

        if trace_path_gz.exists():
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name

            with gzip.open(trace_path_gz, "rb") as f_in:
                with open(tmp_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            trace = az.from_netcdf(tmp_path)
            os.unlink(tmp_path)  # Clean up temp file
            return trace

        elif trace_path.exists():
            return az.from_netcdf(str(trace_path))

        return None
