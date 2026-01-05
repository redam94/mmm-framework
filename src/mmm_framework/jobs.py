"""
Job management system for running MMM models in separate processes.

Provides:
- Background model fitting with progress tracking
- Multiple concurrent job management
- Job persistence and recovery
- Result storage and retrieval
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_JOBS_DIR = Path.home() / ".mmm_framework" / "jobs"

logger = logging.getLogger(__name__)


# =============================================================================
# Job Status Enum
# =============================================================================


class JobStatus(str, Enum):
    """Status of a model fitting job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Job Data Classes
# =============================================================================


@dataclass
class JobProgress:
    """Progress information for a running job."""

    stage: str = "initializing"
    current_step: int = 0
    total_steps: int = 100
    message: str = ""
    started_at: str | None = None
    updated_at: str | None = None

    @property
    def percent_complete(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> JobProgress:
        return cls(**data)


@dataclass
class JobConfig:
    """Configuration for a model fitting job."""

    # Data
    data_path: str | None = None  # Path to pickled panel data

    # Model settings
    n_chains: int = 4
    n_draws: int = 1000
    n_tune: int = 1000
    target_accept: float = 0.95
    use_numpyro: bool = False
    random_seed: int = 42

    # Trend settings
    trend_type: str = "linear"
    trend_settings: dict = field(default_factory=dict)

    # Seasonality
    yearly_order: int = 2

    # Hierarchical
    pool_geo: bool = True

    # Additional metadata
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> JobConfig:
        return cls(**data)


@dataclass
class JobResult:
    """Results from a completed model fitting job."""

    # Diagnostics
    divergences: int = 0
    rhat_max: float = 1.0
    ess_bulk_min: float = 0.0

    # Fit statistics
    r_squared: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0

    # Timing
    fit_duration_seconds: float = 0.0

    # Paths to stored results
    trace_path: str | None = None
    contributions_path: str | None = None
    summary_path: str | None = None

    # Error info (if failed)
    error_message: str | None = None
    error_traceback: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> JobResult:
        return cls(**data)


@dataclass
class Job:
    """Represents a model fitting job."""

    id: str
    status: JobStatus
    config: JobConfig
    progress: JobProgress
    result: JobResult | None = None

    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None

    # Process info
    pid: int | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def is_active(self) -> bool:
        return self.status in [JobStatus.PENDING, JobStatus.RUNNING]

    @property
    def display_name(self) -> str:
        return self.config.name or f"Job {self.id[:8]}"

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            return (end - start).total_seconds()
        elif self.started_at:
            start = datetime.fromisoformat(self.started_at)
            return (datetime.now() - start).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "pid": self.pid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        return cls(
            id=data["id"],
            status=JobStatus(data["status"]),
            config=JobConfig.from_dict(data["config"]),
            progress=JobProgress.from_dict(data["progress"]),
            result=JobResult.from_dict(data["result"]) if data.get("result") else None,
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            pid=data.get("pid"),
        )


# =============================================================================
# Worker Process
# =============================================================================


def _update_progress(
    job_dir: Path, stage: str, current_step: int, total_steps: int, message: str = ""
):
    """Update progress file from worker process using atomic write."""
    progress = JobProgress(
        stage=stage,
        current_step=current_step,
        total_steps=total_steps,
        message=message,
        updated_at=datetime.now().isoformat(),
    )

    progress_path = job_dir / "progress.json"
    temp_path = job_dir / "progress.json.tmp"

    with open(temp_path, "w") as f:
        json.dump(progress.to_dict(), f)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(progress_path)


def _worker_process(
    job_id: str,
    jobs_dir: Path,
    config_dict: dict,
    panel_data: bytes,  # Pickled panel data
):
    """
    Worker process that runs model fitting.

    This runs in a separate process to avoid blocking the main thread.
    """
    import warnings

    warnings.filterwarnings("ignore")

    job_dir = jobs_dir / job_id
    config = JobConfig.from_dict(config_dict)

    result = JobResult()
    start_time = time.time()

    try:
        # Update status to running
        _update_status(job_dir, JobStatus.RUNNING)
        _update_progress(job_dir, "loading", 0, 100, "Loading data...")

        # Unpickle panel data
        panel = pickle.loads(panel_data)

        _update_progress(job_dir, "building", 5, 100, "Building model configuration...")

        # Import here to avoid multiprocessing issues
        from mmm_framework import (
            BayesianMMM,
            TrendConfig,
            TrendType,
            ModelConfigBuilder,
            HierarchicalConfigBuilder,
            SeasonalityConfigBuilder,
            TrendConfigBuilder,
        )

        # Build model config
        model_builder = ModelConfigBuilder()

        if config.use_numpyro:
            model_builder.bayesian_numpyro()
        else:
            model_builder.bayesian_pymc()

        model_builder.with_chains(config.n_chains)
        model_builder.with_draws(config.n_draws)
        model_builder.with_tune(config.n_tune)
        model_builder.with_target_accept(config.target_accept)

        # Seasonality
        season_builder = SeasonalityConfigBuilder()
        if config.yearly_order > 0:
            season_builder.with_yearly(config.yearly_order)
        model_builder.with_seasonality_builder(season_builder)

        # Hierarchical
        if config.pool_geo:
            hier_builder = HierarchicalConfigBuilder().enabled().pool_across_geo()
            model_builder.with_hierarchical_builder(hier_builder)

        model_config = model_builder.build()

        _update_progress(
            job_dir, "building", 10, 100, "Building trend configuration..."
        )

        # Build trend config
        trend_builder = TrendConfigBuilder()
        trend_settings = config.trend_settings

        if config.trend_type == "none":
            trend_builder.none()
        elif config.trend_type == "linear":
            trend_builder.linear()
            if "growth_prior_mu" in trend_settings:
                trend_builder.with_growth_prior(
                    mu=trend_settings.get("growth_prior_mu", 0.0),
                    sigma=trend_settings.get("growth_prior_sigma", 0.1),
                )
        elif config.trend_type == "piecewise":
            trend_builder.piecewise()
            if "n_changepoints" in trend_settings:
                trend_builder.with_n_changepoints(trend_settings["n_changepoints"])
            if "changepoint_range" in trend_settings:
                trend_builder.with_changepoint_range(
                    trend_settings["changepoint_range"]
                )
            if "changepoint_prior_scale" in trend_settings:
                trend_builder.with_changepoint_prior_scale(
                    trend_settings["changepoint_prior_scale"]
                )
        elif config.trend_type == "spline":
            trend_builder.spline()
            if "n_knots" in trend_settings:
                trend_builder.with_n_knots(trend_settings["n_knots"])
            if "spline_degree" in trend_settings:
                trend_builder.with_spline_degree(trend_settings["spline_degree"])
            if "spline_prior_sigma" in trend_settings:
                trend_builder.with_spline_prior_sigma(
                    trend_settings["spline_prior_sigma"]
                )
        elif config.trend_type == "gaussian_process":
            trend_builder.gaussian_process()
            if "gp_lengthscale_mu" in trend_settings:
                trend_builder.with_gp_lengthscale(
                    mu=trend_settings.get("gp_lengthscale_mu", 0.3),
                    sigma=trend_settings.get("gp_lengthscale_sigma", 0.2),
                )
            if "gp_amplitude_sigma" in trend_settings:
                trend_builder.with_gp_amplitude(trend_settings["gp_amplitude_sigma"])
            if "gp_n_basis" in trend_settings:
                trend_builder.with_gp_n_basis(trend_settings["gp_n_basis"])

        trend_config = trend_builder.build()

        _update_progress(job_dir, "building", 15, 100, "Creating model...")

        # Create model
        mmm = BayesianMMM(panel, model_config, trend_config)

        _update_progress(job_dir, "fitting", 20, 100, "Starting MCMC sampling...")

        # Fit model
        # Note: Progress updates during MCMC are tricky without callbacks
        # We'll do coarse updates based on estimated timing
        fit_results = mmm.fit(random_seed=config.random_seed)

        _update_progress(job_dir, "postprocessing", 90, 100, "Computing diagnostics...")

        # Extract diagnostics
        result.divergences = fit_results.diagnostics.get("divergences", 0)
        result.rhat_max = fit_results.diagnostics.get("rhat_max", 1.0)
        result.ess_bulk_min = fit_results.diagnostics.get("ess_bulk_min", 0.0)

        # Compute fit statistics
        try:
            pred = mmm.predict(return_original_scale=True)
            y_obs = mmm.y_raw
            y_pred = pred.y_pred_mean

            ss_res = np.sum((y_obs - y_pred) ** 2)
            ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
            result.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            result.rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
            result.mape = np.mean(np.abs((y_obs - y_pred) / (y_obs + 1e-8))) * 100
        except Exception as e:
            logger.warning(f"Could not compute fit statistics: {e}")

        _update_progress(job_dir, "saving", 95, 100, "Saving results...")

        # Save trace
        trace_path = job_dir / "trace.nc"
        fit_results.trace.to_netcdf(str(trace_path))
        result.trace_path = str(trace_path)

        # Save summary
        summary_path = job_dir / "summary.csv"
        summary = fit_results.summary()
        summary.to_csv(summary_path)
        result.summary_path = str(summary_path)

        # Save the fitted model (for later use)
        model_path = job_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "mmm": mmm,
                    "results": fit_results,
                    "panel": panel,
                },
                f,
            )

        # Compute and save contributions
        try:
            _update_progress(
                job_dir, "contributions", 97, 100, "Computing contributions..."
            )
            contrib = mmm.compute_counterfactual_contributions(
                compute_uncertainty=True, random_seed=config.random_seed
            )
            contrib_path = job_dir / "contributions.pkl"
            with open(contrib_path, "wb") as f:
                pickle.dump(contrib, f)
            result.contributions_path = str(contrib_path)
        except Exception as e:
            logger.warning(f"Could not compute contributions: {e}")

        result.fit_duration_seconds = time.time() - start_time

        _update_progress(job_dir, "complete", 100, 100, "Model fitting complete!")

        # Save result
        _save_result(job_dir, result)
        _update_status(job_dir, JobStatus.COMPLETED)

    except Exception as e:
        result.error_message = str(e)
        result.error_traceback = traceback.format_exc()
        result.fit_duration_seconds = time.time() - start_time

        _save_result(job_dir, result)
        _update_status(job_dir, JobStatus.FAILED)
        _update_progress(job_dir, "failed", 0, 100, f"Error: {str(e)}")


def _update_status(job_dir: Path, status: JobStatus):
    """Update job status file using atomic write."""
    status_path = job_dir / "status.json"
    temp_path = job_dir / "status.json.tmp"

    # Read existing status (with error handling for empty/corrupted files)
    data = {}
    if status_path.exists():
        try:
            with open(status_path) as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            # File is empty or corrupted, start fresh
            data = {}

    data["status"] = status.value
    data["updated_at"] = datetime.now().isoformat()

    if status == JobStatus.RUNNING and "started_at" not in data:
        data["started_at"] = datetime.now().isoformat()

    if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        data["completed_at"] = datetime.now().isoformat()

    # Atomic write: write to temp file, then rename
    with open(temp_path, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())

    # Atomic rename
    temp_path.replace(status_path)


def _save_result(job_dir: Path, result: JobResult):
    """Save job result to file using atomic write."""
    result_path = job_dir / "result.json"
    temp_path = job_dir / "result.json.tmp"

    with open(temp_path, "w") as f:
        json.dump(result.to_dict(), f)
        f.flush()
        os.fsync(f.fileno())
    temp_path.replace(result_path)


# =============================================================================
# Job Manager
# =============================================================================


class JobManager:
    """
    Manages multiple model fitting jobs.

    Jobs are persisted to disk so they survive app restarts.
    Each job runs in a separate process.
    """

    def __init__(self, jobs_dir: Path | str | None = None):
        self.jobs_dir = Path(jobs_dir) if jobs_dir else DEFAULT_JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._processes: dict[str, mp.Process] = {}

        # Clean up any stale "running" jobs on startup
        self._cleanup_stale_jobs()

    def _cleanup_stale_jobs(self):
        """Mark any stale 'running' jobs as failed."""
        for job_id in self._list_job_ids():
            job = self.get_job(job_id)
            if job and job.status == JobStatus.RUNNING:
                # Check if process is still alive
                if job.pid:
                    try:
                        os.kill(job.pid, 0)  # Check if process exists
                    except OSError:
                        # Process doesn't exist, mark as failed
                        job_dir = self.jobs_dir / job_id
                        _update_status(job_dir, JobStatus.FAILED)
                        _update_progress(
                            job_dir,
                            "failed",
                            0,
                            100,
                            "Job was interrupted (process no longer running)",
                        )
                else:
                    # No PID recorded, mark as failed
                    job_dir = self.jobs_dir / job_id
                    _update_status(job_dir, JobStatus.FAILED)

    def _list_job_ids(self) -> list[str]:
        """List all job IDs."""
        if not self.jobs_dir.exists():
            return []

        return [
            d.name
            for d in self.jobs_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]

    def create_job(
        self,
        panel,  # PanelDataset
        config: JobConfig,
    ) -> Job:
        """
        Create a new job (but don't start it yet).

        Parameters
        ----------
        panel : PanelDataset
            The panel data to fit.
        config : JobConfig
            Job configuration.

        Returns
        -------
        Job
            The created job.
        """
        job_id = str(uuid.uuid4())
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(job_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f)

        # Save panel data
        panel_path = job_dir / "panel.pkl"
        with open(panel_path, "wb") as f:
            pickle.dump(panel, f)

        # Initialize status
        _update_status(job_dir, JobStatus.PENDING)

        # Initialize progress
        _update_progress(job_dir, "pending", 0, 100, "Waiting to start...")

        # Create job object
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            config=config,
            progress=JobProgress(stage="pending", message="Waiting to start..."),
        )

        return job

    def start_job(self, job_id: str) -> bool:
        """
        Start a pending job.

        Parameters
        ----------
        job_id : str
            ID of the job to start.

        Returns
        -------
        bool
            True if job was started successfully.
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return False

        if job.status != JobStatus.PENDING:
            logger.error(f"Job {job_id} is not pending (status: {job.status})")
            return False

        job_dir = self.jobs_dir / job_id

        # Load panel data
        panel_path = job_dir / "panel.pkl"
        with open(panel_path, "rb") as f:
            panel_data = f.read()  # Keep as bytes for passing to process

        # Create and start process
        process = mp.Process(
            target=_worker_process,
            args=(job_id, self.jobs_dir, job.config.to_dict(), panel_data),
            daemon=False,  # Allow cleanup
        )
        process.start()

        self._processes[job_id] = process

        # Update with PID (with error handling and atomic write)
        status_path = job_dir / "status.json"
        temp_path = job_dir / "status.json.tmp"

        # Read existing status (handle empty/corrupted files)
        status_data = {"status": "running"}
        if status_path.exists():
            try:
                with open(status_path) as f:
                    content = f.read().strip()
                    if content:
                        status_data = json.loads(content)
            except (json.JSONDecodeError, OSError):
                pass

        status_data["pid"] = process.pid
        status_data["started_at"] = datetime.now().isoformat()
        status_data["status"] = "running"

        # Atomic write
        with open(temp_path, "w") as f:
            json.dump(status_data, f)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(status_path)

        return True

    def submit_job(
        self,
        panel,
        config: JobConfig,
    ) -> Job:
        """
        Create and immediately start a job.

        Parameters
        ----------
        panel : PanelDataset
            The panel data to fit.
        config : JobConfig
            Job configuration.

        Returns
        -------
        Job
            The created and started job.
        """
        job = self.create_job(panel, config)
        self.start_job(job.id)
        return job

    def get_job(self, job_id: str) -> Job | None:
        """
        Get a job by ID.

        Parameters
        ----------
        job_id : str
            The job ID.

        Returns
        -------
        Job or None
            The job, or None if not found.
        """
        job_dir = self.jobs_dir / job_id

        if not job_dir.exists():
            return None

        # Load config (required - return None if missing or corrupted)
        config_path = job_dir / "config.json"
        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                content = f.read().strip()
                if not content:
                    return None
                config = JobConfig.from_dict(json.loads(content))
        except (json.JSONDecodeError, OSError):
            return None

        # Load status (with error handling for empty/corrupted files)
        status_path = job_dir / "status.json"
        status = JobStatus.PENDING
        started_at = None
        completed_at = None
        pid = None

        if status_path.exists():
            try:
                with open(status_path) as f:
                    content = f.read().strip()
                    if content:
                        status_data = json.loads(content)
                        status = JobStatus(status_data.get("status", "pending"))
                        started_at = status_data.get("started_at")
                        completed_at = status_data.get("completed_at")
                        pid = status_data.get("pid")
            except (json.JSONDecodeError, OSError):
                # File is empty or corrupted, use defaults
                pass

        # Load progress (with error handling)
        progress_path = job_dir / "progress.json"
        progress = JobProgress()
        if progress_path.exists():
            try:
                with open(progress_path) as f:
                    content = f.read().strip()
                    if content:
                        progress = JobProgress.from_dict(json.loads(content))
            except (json.JSONDecodeError, OSError):
                pass

        # Load result (with error handling)
        result_path = job_dir / "result.json"
        result = None
        if result_path.exists():
            try:
                with open(result_path) as f:
                    content = f.read().strip()
                    if content:
                        result = JobResult.from_dict(json.loads(content))
            except (json.JSONDecodeError, OSError):
                pass

        # Get created_at from directory
        created_at = datetime.fromtimestamp(job_dir.stat().st_ctime).isoformat()

        return Job(
            id=job_id,
            status=status,
            config=config,
            progress=progress,
            result=result,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            pid=pid,
        )

    def list_jobs(
        self,
        status_filter: list[JobStatus] | None = None,
        limit: int | None = None,
        order_by: str = "created_at",
        ascending: bool = False,
    ) -> list[Job]:
        """
        List all jobs.

        Parameters
        ----------
        status_filter : list[JobStatus], optional
            Only return jobs with these statuses.
        limit : int, optional
            Maximum number of jobs to return.
        order_by : str
            Field to sort by (created_at, started_at, completed_at).
        ascending : bool
            Sort order.

        Returns
        -------
        list[Job]
            List of jobs.
        """
        jobs = []

        for job_id in self._list_job_ids():
            job = self.get_job(job_id)
            if job:
                if status_filter and job.status not in status_filter:
                    continue
                jobs.append(job)

        # Sort
        def get_sort_key(j: Job):
            val = getattr(j, order_by, j.created_at)
            return val or ""

        jobs.sort(key=get_sort_key, reverse=not ascending)

        if limit:
            jobs = jobs[:limit]

        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        Parameters
        ----------
        job_id : str
            The job ID.

        Returns
        -------
        bool
            True if job was cancelled.
        """
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            return False

        # Kill process if running
        if job_id in self._processes:
            process = self._processes[job_id]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
            del self._processes[job_id]
        elif job.pid:
            try:
                os.kill(job.pid, 9)  # SIGKILL
            except OSError:
                pass

        job_dir = self.jobs_dir / job_id
        _update_status(job_dir, JobStatus.CANCELLED)
        _update_progress(job_dir, "cancelled", 0, 100, "Job was cancelled")

        return True

    def delete_job(self, job_id: str, force: bool = False) -> bool:
        """
        Delete a job and its files.

        Parameters
        ----------
        job_id : str
            The job ID.
        force : bool
            If True, delete even if running.

        Returns
        -------
        bool
            True if job was deleted.
        """
        job = self.get_job(job_id)
        if not job:
            return False

        if job.is_active and not force:
            logger.error("Cannot delete active job without force=True")
            return False

        if job.is_active:
            self.cancel_job(job_id)

        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)

        return True

    def load_job_results(self, job_id: str) -> dict | None:
        """
        Load full results for a completed job.

        Parameters
        ----------
        job_id : str
            The job ID.

        Returns
        -------
        dict or None
            Dictionary with 'mmm', 'results', 'panel', 'contributions' keys,
            or None if not found or not completed.
        """
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.COMPLETED:
            return None

        job_dir = self.jobs_dir / job_id

        # Load model and results
        model_path = job_dir / "model.pkl"
        if not model_path.exists():
            return None

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # Load contributions if available
        contrib_path = job_dir / "contributions.pkl"
        if contrib_path.exists():
            with open(contrib_path, "rb") as f:
                data["contributions"] = pickle.load(f)

        return data

    def get_active_jobs(self) -> list[Job]:
        """Get all currently active (pending or running) jobs."""
        return self.list_jobs(status_filter=[JobStatus.PENDING, JobStatus.RUNNING])

    def get_completed_jobs(self) -> list[Job]:
        """Get all completed jobs."""
        return self.list_jobs(status_filter=[JobStatus.COMPLETED])

    def cleanup_old_jobs(self, max_age_days: int = 30, max_jobs: int = 50):
        """
        Remove old completed/failed jobs.

        Parameters
        ----------
        max_age_days : int
            Remove jobs older than this many days.
        max_jobs : int
            Keep at most this many completed jobs.
        """
        completed_jobs = self.list_jobs(
            status_filter=[JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
            order_by="completed_at",
            ascending=False,
        )

        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 3600)

        for i, job in enumerate(completed_jobs):
            should_delete = False

            # Delete if too old
            if job.completed_at:
                completed_ts = datetime.fromisoformat(job.completed_at).timestamp()
                if completed_ts < cutoff:
                    should_delete = True

            # Delete if too many jobs
            if i >= max_jobs:
                should_delete = True

            if should_delete:
                self.delete_job(job.id)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: JobManager | None = None


def get_job_manager(jobs_dir: Path | str | None = None) -> JobManager:
    """Get or create the default job manager."""
    global _default_manager

    if _default_manager is None or (
        jobs_dir and Path(jobs_dir) != _default_manager.jobs_dir
    ):
        _default_manager = JobManager(jobs_dir)

    return _default_manager


def submit_model_job(
    panel,
    name: str = "",
    description: str = "",
    n_chains: int = 4,
    n_draws: int = 1000,
    n_tune: int = 1000,
    target_accept: float = 0.95,
    use_numpyro: bool = False,
    trend_type: str = "linear",
    trend_settings: dict | None = None,
    yearly_order: int = 2,
    pool_geo: bool = True,
    random_seed: int = 42,
    tags: list[str] | None = None,
) -> Job:
    """
    Convenience function to submit a model fitting job.

    Parameters
    ----------
    panel : PanelDataset
        The panel data.
    name : str
        Job name.
    description : str
        Job description.
    n_chains : int
        Number of MCMC chains.
    n_draws : int
        Number of draws per chain.
    n_tune : int
        Number of tuning samples.
    target_accept : float
        Target acceptance rate.
    use_numpyro : bool
        Use NumPyro backend.
    trend_type : str
        Trend type (none, linear, piecewise, spline, gaussian_process).
    trend_settings : dict
        Trend-specific settings.
    yearly_order : int
        Fourier order for yearly seasonality.
    pool_geo : bool
        Enable hierarchical geo pooling.
    random_seed : int
        Random seed.
    tags : list[str]
        Tags for the job.

    Returns
    -------
    Job
        The submitted job.
    """
    config = JobConfig(
        name=name,
        description=description,
        n_chains=n_chains,
        n_draws=n_draws,
        n_tune=n_tune,
        target_accept=target_accept,
        use_numpyro=use_numpyro,
        trend_type=trend_type,
        trend_settings=trend_settings or {},
        yearly_order=yearly_order,
        pool_geo=pool_geo,
        random_seed=random_seed,
        tags=tags or [],
    )

    manager = get_job_manager()
    return manager.submit_job(panel, config)
