"""
Tests for the jobs module.

Tests cover:
- JobStatus enum
- JobProgress, JobConfig, JobResult, Job dataclasses
- Helper functions (_update_progress, _update_status, _save_result)
- JobManager class
- Convenience functions (get_job_manager, submit_model_job)
"""

import json
import pickle
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from mmm_framework.jobs import (
    JobStatus,
    JobProgress,
    JobConfig,
    JobResult,
    Job,
    JobManager,
    get_job_manager,
    submit_model_job,
    _update_progress,
    _update_status,
    _save_result,
    DEFAULT_JOBS_DIR,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_jobs_dir(tmp_path):
    """Create a temporary jobs directory."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    return jobs_dir


class MockPanel:
    """Picklable mock PanelDataset for testing."""

    def __init__(self):
        self.y = pd.Series([100, 200, 300])
        self.X_media = pd.DataFrame({"TV": [10, 20, 30], "Digital": [5, 10, 15]})
        self.X_controls = pd.DataFrame({"Price": [1.0, 1.1, 1.2]})


@pytest.fixture
def mock_panel():
    """Create a mock PanelDataset."""
    return MockPanel()


@pytest.fixture
def sample_job_config():
    """Create a sample JobConfig."""
    return JobConfig(
        name="Test Job",
        description="A test job",
        n_chains=2,
        n_draws=500,
        n_tune=250,
        target_accept=0.9,
        use_numpyro=False,
        trend_type="linear",
        yearly_order=2,
        pool_geo=True,
        random_seed=42,
        tags=["test", "unit"],
    )


@pytest.fixture
def job_manager(tmp_jobs_dir):
    """Create a JobManager with temporary directory."""
    return JobManager(tmp_jobs_dir)


# =============================================================================
# TestJobStatus
# =============================================================================


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_enum_values(self):
        """Test all JobStatus values exist."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_status_is_string_enum(self):
        """Test JobStatus inherits from str."""
        assert isinstance(JobStatus.PENDING, str)
        assert JobStatus.PENDING == "pending"

    def test_status_from_string(self):
        """Test creating JobStatus from string value."""
        assert JobStatus("pending") == JobStatus.PENDING
        assert JobStatus("running") == JobStatus.RUNNING
        assert JobStatus("completed") == JobStatus.COMPLETED

    def test_status_invalid_raises(self):
        """Test invalid status string raises error."""
        with pytest.raises(ValueError):
            JobStatus("invalid")


# =============================================================================
# TestJobProgress
# =============================================================================


class TestJobProgress:
    """Tests for JobProgress dataclass."""

    def test_creation_with_defaults(self):
        """Test JobProgress creation with default values."""
        progress = JobProgress()

        assert progress.stage == "initializing"
        assert progress.current_step == 0
        assert progress.total_steps == 100
        assert progress.message == ""
        assert progress.started_at is None
        assert progress.updated_at is None

    def test_creation_with_values(self):
        """Test JobProgress creation with custom values."""
        progress = JobProgress(
            stage="fitting",
            current_step=50,
            total_steps=100,
            message="Fitting model...",
            started_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:30:00",
        )

        assert progress.stage == "fitting"
        assert progress.current_step == 50
        assert progress.total_steps == 100
        assert progress.message == "Fitting model..."

    def test_percent_complete_calculation(self):
        """Test percent_complete property."""
        progress = JobProgress(current_step=25, total_steps=100)
        assert progress.percent_complete == 25.0

        progress = JobProgress(current_step=50, total_steps=200)
        assert progress.percent_complete == 25.0

        progress = JobProgress(current_step=100, total_steps=100)
        assert progress.percent_complete == 100.0

    def test_percent_complete_with_zero_total(self):
        """Test percent_complete with zero total steps."""
        progress = JobProgress(current_step=50, total_steps=0)
        assert progress.percent_complete == 0.0

    def test_percent_complete_caps_at_100(self):
        """Test percent_complete does not exceed 100."""
        progress = JobProgress(current_step=150, total_steps=100)
        assert progress.percent_complete == 100.0

    def test_to_dict_serialization(self):
        """Test to_dict produces correct dictionary."""
        progress = JobProgress(
            stage="fitting",
            current_step=50,
            total_steps=100,
            message="Test",
        )

        d = progress.to_dict()

        assert d["stage"] == "fitting"
        assert d["current_step"] == 50
        assert d["total_steps"] == 100
        assert d["message"] == "Test"
        assert "started_at" in d
        assert "updated_at" in d

    def test_from_dict_deserialization(self):
        """Test from_dict recreates JobProgress."""
        data = {
            "stage": "postprocessing",
            "current_step": 90,
            "total_steps": 100,
            "message": "Computing...",
            "started_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T01:00:00",
        }

        progress = JobProgress.from_dict(data)

        assert progress.stage == "postprocessing"
        assert progress.current_step == 90
        assert progress.total_steps == 100
        assert progress.message == "Computing..."

    def test_round_trip_serialization(self):
        """Test serialization round-trip preserves data."""
        original = JobProgress(
            stage="fitting",
            current_step=75,
            total_steps=100,
            message="Almost done",
            started_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:45:00",
        )

        d = original.to_dict()
        restored = JobProgress.from_dict(d)

        assert restored.stage == original.stage
        assert restored.current_step == original.current_step
        assert restored.total_steps == original.total_steps
        assert restored.message == original.message


# =============================================================================
# TestJobConfig
# =============================================================================


class TestJobConfig:
    """Tests for JobConfig dataclass."""

    def test_creation_with_defaults(self):
        """Test JobConfig creation with default values."""
        config = JobConfig()

        assert config.data_path is None
        assert config.n_chains == 4
        assert config.n_draws == 1000
        assert config.n_tune == 1000
        assert config.target_accept == 0.95
        assert config.use_numpyro is False
        assert config.random_seed == 42
        assert config.trend_type == "linear"
        assert config.trend_settings == {}
        assert config.yearly_order == 2
        assert config.pool_geo is True
        assert config.name == ""
        assert config.description == ""
        assert config.tags == []

    def test_creation_with_custom_values(self):
        """Test JobConfig creation with custom values."""
        config = JobConfig(
            name="Custom Job",
            n_chains=2,
            n_draws=500,
            use_numpyro=True,
            trend_type="piecewise",
            trend_settings={"n_changepoints": 5},
            tags=["test"],
        )

        assert config.name == "Custom Job"
        assert config.n_chains == 2
        assert config.n_draws == 500
        assert config.use_numpyro is True
        assert config.trend_type == "piecewise"
        assert config.trend_settings == {"n_changepoints": 5}
        assert config.tags == ["test"]

    def test_trend_settings_default_factory(self):
        """Test trend_settings uses separate dict per instance."""
        config1 = JobConfig()
        config2 = JobConfig()

        config1.trend_settings["key"] = "value"

        assert config2.trend_settings == {}

    def test_tags_default_factory(self):
        """Test tags uses separate list per instance."""
        config1 = JobConfig()
        config2 = JobConfig()

        config1.tags.append("tag1")

        assert config2.tags == []

    def test_to_dict_serialization(self):
        """Test to_dict produces correct dictionary."""
        config = JobConfig(
            name="Test",
            n_chains=2,
            trend_settings={"key": "value"},
        )

        d = config.to_dict()

        assert d["name"] == "Test"
        assert d["n_chains"] == 2
        assert d["trend_settings"] == {"key": "value"}
        assert "data_path" in d
        assert "n_draws" in d

    def test_from_dict_deserialization(self):
        """Test from_dict recreates JobConfig."""
        data = {
            "data_path": None,
            "n_chains": 2,
            "n_draws": 500,
            "n_tune": 250,
            "target_accept": 0.9,
            "use_numpyro": True,
            "random_seed": 123,
            "trend_type": "spline",
            "trend_settings": {"n_knots": 10},
            "yearly_order": 3,
            "pool_geo": False,
            "name": "From Dict",
            "description": "Test",
            "tags": ["a", "b"],
        }

        config = JobConfig.from_dict(data)

        assert config.n_chains == 2
        assert config.use_numpyro is True
        assert config.trend_type == "spline"
        assert config.tags == ["a", "b"]

    def test_round_trip_with_trend_settings(self):
        """Test round-trip with complex trend settings."""
        original = JobConfig(
            trend_type="piecewise",
            trend_settings={
                "n_changepoints": 5,
                "changepoint_range": 0.8,
                "changepoint_prior_scale": 0.05,
            },
        )

        d = original.to_dict()
        restored = JobConfig.from_dict(d)

        assert restored.trend_settings == original.trend_settings


# =============================================================================
# TestJobResult
# =============================================================================


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_creation_with_defaults(self):
        """Test JobResult creation with default values."""
        result = JobResult()

        assert result.divergences == 0
        assert result.rhat_max == 1.0
        assert result.ess_bulk_min == 0.0
        assert result.r_squared == 0.0
        assert result.rmse == 0.0
        assert result.mape == 0.0
        assert result.fit_duration_seconds == 0.0
        assert result.trace_path is None
        assert result.contributions_path is None
        assert result.summary_path is None
        assert result.error_message is None
        assert result.error_traceback is None

    def test_creation_with_error_info(self):
        """Test JobResult with error information."""
        result = JobResult(
            error_message="Model fitting failed",
            error_traceback="Traceback...",
            fit_duration_seconds=10.5,
        )

        assert result.error_message == "Model fitting failed"
        assert result.error_traceback == "Traceback..."
        assert result.fit_duration_seconds == 10.5

    def test_creation_with_paths(self):
        """Test JobResult with result paths."""
        result = JobResult(
            trace_path="/path/to/trace.nc",
            contributions_path="/path/to/contrib.pkl",
            summary_path="/path/to/summary.csv",
        )

        assert result.trace_path == "/path/to/trace.nc"
        assert result.contributions_path == "/path/to/contrib.pkl"
        assert result.summary_path == "/path/to/summary.csv"

    def test_to_dict_serialization(self):
        """Test to_dict produces correct dictionary."""
        result = JobResult(
            divergences=5,
            rhat_max=1.01,
            r_squared=0.85,
        )

        d = result.to_dict()

        assert d["divergences"] == 5
        assert d["rhat_max"] == 1.01
        assert d["r_squared"] == 0.85

    def test_from_dict_deserialization(self):
        """Test from_dict recreates JobResult."""
        data = {
            "divergences": 10,
            "rhat_max": 1.02,
            "ess_bulk_min": 400.0,
            "r_squared": 0.9,
            "rmse": 50.0,
            "mape": 5.0,
            "fit_duration_seconds": 120.0,
            "trace_path": "/trace",
            "contributions_path": "/contrib",
            "summary_path": "/summary",
            "error_message": None,
            "error_traceback": None,
        }

        result = JobResult.from_dict(data)

        assert result.divergences == 10
        assert result.r_squared == 0.9
        assert result.trace_path == "/trace"


# =============================================================================
# TestJob
# =============================================================================


class TestJob:
    """Tests for Job dataclass."""

    def test_creation_sets_created_at(self):
        """Test Job creation sets created_at automatically."""
        config = JobConfig(name="Test")
        progress = JobProgress()

        job = Job(
            id="test-id",
            status=JobStatus.PENDING,
            config=config,
            progress=progress,
        )

        assert job.created_at != ""
        # Should be valid ISO format
        datetime.fromisoformat(job.created_at)

    def test_is_active_property_pending(self):
        """Test is_active returns True for PENDING."""
        job = Job(
            id="test",
            status=JobStatus.PENDING,
            config=JobConfig(),
            progress=JobProgress(),
        )

        assert job.is_active is True

    def test_is_active_property_running(self):
        """Test is_active returns True for RUNNING."""
        job = Job(
            id="test",
            status=JobStatus.RUNNING,
            config=JobConfig(),
            progress=JobProgress(),
        )

        assert job.is_active is True

    def test_is_active_property_completed(self):
        """Test is_active returns False for COMPLETED."""
        job = Job(
            id="test",
            status=JobStatus.COMPLETED,
            config=JobConfig(),
            progress=JobProgress(),
        )

        assert job.is_active is False

    def test_is_active_property_failed(self):
        """Test is_active returns False for FAILED."""
        job = Job(
            id="test",
            status=JobStatus.FAILED,
            config=JobConfig(),
            progress=JobProgress(),
        )

        assert job.is_active is False

    def test_display_name_with_config_name(self):
        """Test display_name uses config name when available."""
        job = Job(
            id="abc123def456",
            status=JobStatus.PENDING,
            config=JobConfig(name="My Named Job"),
            progress=JobProgress(),
        )

        assert job.display_name == "My Named Job"

    def test_display_name_fallback_to_id(self):
        """Test display_name falls back to truncated ID."""
        job = Job(
            id="abc123def456",
            status=JobStatus.PENDING,
            config=JobConfig(),  # No name
            progress=JobProgress(),
        )

        assert job.display_name == "Job abc123de"

    def test_duration_seconds_with_completed_job(self):
        """Test duration_seconds for completed job."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)

        job = Job(
            id="test",
            status=JobStatus.COMPLETED,
            config=JobConfig(),
            progress=JobProgress(),
            started_at=start.isoformat(),
            completed_at=end.isoformat(),
        )

        assert job.duration_seconds == 3600.0  # 1 hour

    def test_duration_seconds_with_running_job(self):
        """Test duration_seconds for running job."""
        start = datetime.now() - timedelta(minutes=30)

        job = Job(
            id="test",
            status=JobStatus.RUNNING,
            config=JobConfig(),
            progress=JobProgress(),
            started_at=start.isoformat(),
        )

        # Should be approximately 30 minutes (1800 seconds)
        assert 1790 <= job.duration_seconds <= 1810

    def test_duration_seconds_with_pending_job(self):
        """Test duration_seconds for pending job (no start time)."""
        job = Job(
            id="test",
            status=JobStatus.PENDING,
            config=JobConfig(),
            progress=JobProgress(),
        )

        assert job.duration_seconds is None

    def test_to_dict_serialization(self):
        """Test to_dict produces correct dictionary."""
        job = Job(
            id="test-id",
            status=JobStatus.RUNNING,
            config=JobConfig(name="Test"),
            progress=JobProgress(stage="fitting"),
            result=JobResult(r_squared=0.9),
            started_at="2024-01-01T00:00:00",
            pid=12345,
        )

        d = job.to_dict()

        assert d["id"] == "test-id"
        assert d["status"] == "running"
        assert d["config"]["name"] == "Test"
        assert d["progress"]["stage"] == "fitting"
        assert d["result"]["r_squared"] == 0.9
        assert d["pid"] == 12345

    def test_to_dict_without_result(self):
        """Test to_dict with no result."""
        job = Job(
            id="test",
            status=JobStatus.PENDING,
            config=JobConfig(),
            progress=JobProgress(),
        )

        d = job.to_dict()

        assert d["result"] is None

    def test_from_dict_deserialization(self):
        """Test from_dict recreates Job."""
        data = {
            "id": "restored-job",
            "status": "completed",
            "config": JobConfig(name="Restored").to_dict(),
            "progress": JobProgress(stage="complete").to_dict(),
            "result": JobResult(r_squared=0.95).to_dict(),
            "created_at": "2024-01-01T00:00:00",
            "started_at": "2024-01-01T00:01:00",
            "completed_at": "2024-01-01T01:00:00",
            "pid": 9999,
        }

        job = Job.from_dict(data)

        assert job.id == "restored-job"
        assert job.status == JobStatus.COMPLETED
        assert job.config.name == "Restored"
        assert job.progress.stage == "complete"
        assert job.result.r_squared == 0.95
        assert job.pid == 9999

    def test_from_dict_without_result(self):
        """Test from_dict with no result."""
        data = {
            "id": "test",
            "status": "pending",
            "config": JobConfig().to_dict(),
            "progress": JobProgress().to_dict(),
            "result": None,
        }

        job = Job.from_dict(data)

        assert job.result is None

    def test_round_trip_serialization(self):
        """Test complete round-trip serialization."""
        original = Job(
            id="round-trip-test",
            status=JobStatus.COMPLETED,
            config=JobConfig(name="Original", tags=["a", "b"]),
            progress=JobProgress(stage="done", current_step=100),
            result=JobResult(r_squared=0.92, divergences=2),
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T02:00:00",
            pid=5555,
        )

        d = original.to_dict()
        restored = Job.from_dict(d)

        assert restored.id == original.id
        assert restored.status == original.status
        assert restored.config.name == original.config.name
        assert restored.config.tags == original.config.tags
        assert restored.progress.stage == original.progress.stage
        assert restored.result.r_squared == original.result.r_squared


# =============================================================================
# TestUpdateProgress
# =============================================================================


class TestUpdateProgress:
    """Tests for _update_progress function."""

    def test_update_progress_creates_file(self, tmp_path):
        """Test _update_progress creates progress.json file."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        _update_progress(job_dir, "loading", 10, 100, "Loading data...")

        progress_path = job_dir / "progress.json"
        assert progress_path.exists()

    def test_update_progress_content_structure(self, tmp_path):
        """Test _update_progress writes correct content."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        _update_progress(job_dir, "fitting", 50, 100, "Fitting model...")

        with open(job_dir / "progress.json") as f:
            data = json.load(f)

        assert data["stage"] == "fitting"
        assert data["current_step"] == 50
        assert data["total_steps"] == 100
        assert data["message"] == "Fitting model..."
        assert "updated_at" in data

    def test_update_progress_atomic_write(self, tmp_path):
        """Test _update_progress uses atomic write (temp file)."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        _update_progress(job_dir, "test", 0, 100, "")

        # Temp file should not exist after completion
        assert not (job_dir / "progress.json.tmp").exists()
        assert (job_dir / "progress.json").exists()


# =============================================================================
# TestUpdateStatus
# =============================================================================


class TestUpdateStatus:
    """Tests for _update_status function."""

    def test_update_status_creates_file(self, tmp_path):
        """Test _update_status creates status.json file."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        _update_status(job_dir, JobStatus.PENDING)

        assert (job_dir / "status.json").exists()

    def test_update_status_sets_started_at_on_running(self, tmp_path):
        """Test _update_status sets started_at when status is RUNNING."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        _update_status(job_dir, JobStatus.RUNNING)

        with open(job_dir / "status.json") as f:
            data = json.load(f)

        assert "started_at" in data
        assert data["status"] == "running"

    def test_update_status_sets_completed_at_on_terminal(self, tmp_path):
        """Test _update_status sets completed_at for terminal states."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        for status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            _update_status(job_dir, status)

            with open(job_dir / "status.json") as f:
                data = json.load(f)

            assert "completed_at" in data

    def test_update_status_handles_empty_file(self, tmp_path):
        """Test _update_status handles empty existing file."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        # Create empty file
        (job_dir / "status.json").write_text("")

        # Should not raise
        _update_status(job_dir, JobStatus.RUNNING)

        with open(job_dir / "status.json") as f:
            data = json.load(f)

        assert data["status"] == "running"

    def test_update_status_handles_corrupted_json(self, tmp_path):
        """Test _update_status handles corrupted JSON file."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        # Create corrupted JSON
        (job_dir / "status.json").write_text("{invalid json")

        # Should not raise
        _update_status(job_dir, JobStatus.COMPLETED)

        with open(job_dir / "status.json") as f:
            data = json.load(f)

        assert data["status"] == "completed"

    def test_update_status_atomic_write(self, tmp_path):
        """Test _update_status uses atomic write."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        _update_status(job_dir, JobStatus.PENDING)

        # Temp file should not exist
        assert not (job_dir / "status.json.tmp").exists()


# =============================================================================
# TestSaveResult
# =============================================================================


class TestSaveResult:
    """Tests for _save_result function."""

    def test_save_result_creates_file(self, tmp_path):
        """Test _save_result creates result.json file."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        result = JobResult(r_squared=0.9)
        _save_result(job_dir, result)

        assert (job_dir / "result.json").exists()

    def test_save_result_content_structure(self, tmp_path):
        """Test _save_result writes correct content."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        result = JobResult(
            divergences=5,
            rhat_max=1.01,
            r_squared=0.85,
            error_message="Test error",
        )
        _save_result(job_dir, result)

        with open(job_dir / "result.json") as f:
            data = json.load(f)

        assert data["divergences"] == 5
        assert data["rhat_max"] == 1.01
        assert data["r_squared"] == 0.85
        assert data["error_message"] == "Test error"

    def test_save_result_atomic_write(self, tmp_path):
        """Test _save_result uses atomic write."""
        job_dir = tmp_path / "job1"
        job_dir.mkdir()

        _save_result(job_dir, JobResult())

        assert not (job_dir / "result.json.tmp").exists()


# =============================================================================
# TestJobManager
# =============================================================================


class TestJobManager:
    """Tests for JobManager class."""

    def test_init_creates_directory(self, tmp_path):
        """Test JobManager creates jobs directory."""
        jobs_dir = tmp_path / "new_jobs"

        manager = JobManager(jobs_dir)

        assert jobs_dir.exists()
        assert manager.jobs_dir == jobs_dir

    def test_init_with_string_path(self, tmp_path):
        """Test JobManager accepts string path."""
        jobs_dir = str(tmp_path / "jobs")

        manager = JobManager(jobs_dir)

        assert manager.jobs_dir == Path(jobs_dir)

    def test_init_with_default_path(self):
        """Test JobManager uses default path when none provided."""
        with patch.object(Path, "mkdir"):
            manager = JobManager()
            assert manager.jobs_dir == DEFAULT_JOBS_DIR

    def test_create_job_creates_directory_structure(
        self, job_manager, mock_panel, sample_job_config
    ):
        """Test create_job creates job directory with files."""
        job = job_manager.create_job(mock_panel, sample_job_config)

        job_dir = job_manager.jobs_dir / job.id
        assert job_dir.exists()
        assert (job_dir / "config.json").exists()
        assert (job_dir / "panel.pkl").exists()
        assert (job_dir / "status.json").exists()
        assert (job_dir / "progress.json").exists()

    def test_create_job_saves_config(
        self, job_manager, mock_panel, sample_job_config
    ):
        """Test create_job saves correct config."""
        job = job_manager.create_job(mock_panel, sample_job_config)

        with open(job_manager.jobs_dir / job.id / "config.json") as f:
            saved_config = json.load(f)

        assert saved_config["name"] == "Test Job"
        assert saved_config["n_chains"] == 2
        assert saved_config["tags"] == ["test", "unit"]

    def test_create_job_saves_panel_data(
        self, job_manager, mock_panel, sample_job_config
    ):
        """Test create_job saves panel data."""
        job = job_manager.create_job(mock_panel, sample_job_config)

        panel_path = job_manager.jobs_dir / job.id / "panel.pkl"
        with open(panel_path, "rb") as f:
            loaded_panel = pickle.load(f)

        # Compare panel data attributes
        pd.testing.assert_series_equal(loaded_panel.y, mock_panel.y)
        pd.testing.assert_frame_equal(loaded_panel.X_media, mock_panel.X_media)

    def test_create_job_returns_pending_job(
        self, job_manager, mock_panel, sample_job_config
    ):
        """Test create_job returns job with PENDING status."""
        job = job_manager.create_job(mock_panel, sample_job_config)

        assert job.status == JobStatus.PENDING
        assert job.config.name == sample_job_config.name
        assert job.progress.stage == "pending"

    def test_get_job_not_found(self, job_manager):
        """Test get_job returns None for nonexistent job."""
        result = job_manager.get_job("nonexistent-id")
        assert result is None

    def test_get_job_missing_config(self, job_manager):
        """Test get_job returns None if config is missing."""
        job_dir = job_manager.jobs_dir / "bad-job"
        job_dir.mkdir()
        # No config.json created

        result = job_manager.get_job("bad-job")
        assert result is None

    def test_get_job_corrupted_files(self, job_manager):
        """Test get_job handles corrupted JSON files gracefully."""
        job_dir = job_manager.jobs_dir / "corrupted-job"
        job_dir.mkdir()

        # Create valid config
        with open(job_dir / "config.json", "w") as f:
            json.dump(JobConfig().to_dict(), f)

        # Create corrupted status
        (job_dir / "status.json").write_text("{invalid")

        # Should still return job with default status
        result = job_manager.get_job("corrupted-job")
        assert result is not None
        assert result.status == JobStatus.PENDING

    def test_get_job_loads_all_components(
        self, job_manager, mock_panel, sample_job_config
    ):
        """Test get_job loads all job components."""
        created_job = job_manager.create_job(mock_panel, sample_job_config)

        # Simulate progress update
        _update_progress(
            job_manager.jobs_dir / created_job.id,
            "fitting",
            50,
            100,
            "Fitting...",
        )

        loaded_job = job_manager.get_job(created_job.id)

        assert loaded_job is not None
        assert loaded_job.id == created_job.id
        assert loaded_job.config.name == sample_job_config.name
        assert loaded_job.progress.stage == "fitting"
        assert loaded_job.progress.current_step == 50

    def test_list_jobs_empty(self, job_manager):
        """Test list_jobs returns empty list when no jobs."""
        jobs = job_manager.list_jobs()
        assert jobs == []

    def test_list_jobs_returns_all(self, job_manager, mock_panel):
        """Test list_jobs returns all jobs."""
        job1 = job_manager.create_job(mock_panel, JobConfig(name="Job1"))
        job2 = job_manager.create_job(mock_panel, JobConfig(name="Job2"))

        jobs = job_manager.list_jobs()

        assert len(jobs) == 2
        job_names = {j.config.name for j in jobs}
        assert job_names == {"Job1", "Job2"}

    def test_list_jobs_with_status_filter(self, job_manager, mock_panel):
        """Test list_jobs filters by status."""
        job1 = job_manager.create_job(mock_panel, JobConfig(name="Job1"))

        # Manually update one to completed
        _update_status(job_manager.jobs_dir / job1.id, JobStatus.COMPLETED)

        job2 = job_manager.create_job(mock_panel, JobConfig(name="Job2"))

        pending_jobs = job_manager.list_jobs(status_filter=[JobStatus.PENDING])
        completed_jobs = job_manager.list_jobs(status_filter=[JobStatus.COMPLETED])

        assert len(pending_jobs) == 1
        assert len(completed_jobs) == 1
        assert pending_jobs[0].config.name == "Job2"
        assert completed_jobs[0].config.name == "Job1"

    def test_list_jobs_with_limit(self, job_manager, mock_panel):
        """Test list_jobs respects limit."""
        for i in range(5):
            job_manager.create_job(mock_panel, JobConfig(name=f"Job{i}"))

        jobs = job_manager.list_jobs(limit=3)

        assert len(jobs) == 3

    def test_list_jobs_ordering(self, job_manager, mock_panel):
        """Test list_jobs respects ordering."""
        job1 = job_manager.create_job(mock_panel, JobConfig(name="First"))
        time.sleep(0.01)  # Ensure different timestamps
        job2 = job_manager.create_job(mock_panel, JobConfig(name="Second"))

        # Default: descending by created_at
        jobs = job_manager.list_jobs()
        assert jobs[0].config.name == "Second"  # Most recent first

        # Ascending
        jobs = job_manager.list_jobs(ascending=True)
        assert jobs[0].config.name == "First"  # Oldest first

    def test_start_job_not_found(self, job_manager):
        """Test start_job returns False for nonexistent job."""
        result = job_manager.start_job("nonexistent")
        assert result is False

    def test_start_job_not_pending(self, job_manager, mock_panel):
        """Test start_job returns False for non-pending job."""
        job = job_manager.create_job(mock_panel, JobConfig())
        _update_status(job_manager.jobs_dir / job.id, JobStatus.COMPLETED)

        result = job_manager.start_job(job.id)
        assert result is False

    @patch("mmm_framework.jobs.mp.Process")
    def test_start_job_success(self, mock_process_class, job_manager, mock_panel):
        """Test start_job starts process successfully."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        job = job_manager.create_job(mock_panel, JobConfig())
        result = job_manager.start_job(job.id)

        assert result is True
        mock_process.start.assert_called_once()

    @patch("mmm_framework.jobs.mp.Process")
    def test_submit_job_creates_and_starts(
        self, mock_process_class, job_manager, mock_panel, sample_job_config
    ):
        """Test submit_job creates and starts job."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process_class.return_value = mock_process

        job = job_manager.submit_job(mock_panel, sample_job_config)

        assert job is not None
        assert job.config.name == sample_job_config.name
        mock_process.start.assert_called_once()

    def test_cancel_job_not_found(self, job_manager):
        """Test cancel_job returns False for nonexistent job."""
        result = job_manager.cancel_job("nonexistent")
        assert result is False

    def test_cancel_job_completed(self, job_manager, mock_panel):
        """Test cancel_job returns False for completed job."""
        job = job_manager.create_job(mock_panel, JobConfig())
        _update_status(job_manager.jobs_dir / job.id, JobStatus.COMPLETED)

        result = job_manager.cancel_job(job.id)
        assert result is False

    def test_cancel_job_pending(self, job_manager, mock_panel):
        """Test cancel_job cancels pending job."""
        job = job_manager.create_job(mock_panel, JobConfig())

        result = job_manager.cancel_job(job.id)

        assert result is True
        updated_job = job_manager.get_job(job.id)
        assert updated_job.status == JobStatus.CANCELLED

    @patch("mmm_framework.jobs.mp.Process")
    def test_cancel_job_running(self, mock_process_class, job_manager, mock_panel):
        """Test cancel_job terminates running process."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        job = job_manager.create_job(mock_panel, JobConfig())
        job_manager.start_job(job.id)

        # Now cancel
        result = job_manager.cancel_job(job.id)

        assert result is True
        mock_process.terminate.assert_called_once()

    def test_delete_job_not_found(self, job_manager):
        """Test delete_job returns False for nonexistent job."""
        result = job_manager.delete_job("nonexistent")
        assert result is False

    def test_delete_job_active_without_force(self, job_manager, mock_panel):
        """Test delete_job fails for active job without force."""
        job = job_manager.create_job(mock_panel, JobConfig())

        result = job_manager.delete_job(job.id, force=False)
        assert result is False

        # Job should still exist
        assert job_manager.get_job(job.id) is not None

    def test_delete_job_active_with_force(self, job_manager, mock_panel):
        """Test delete_job succeeds for active job with force."""
        job = job_manager.create_job(mock_panel, JobConfig())

        result = job_manager.delete_job(job.id, force=True)
        assert result is True

        # Job should be deleted
        assert job_manager.get_job(job.id) is None

    def test_delete_job_completed(self, job_manager, mock_panel):
        """Test delete_job succeeds for completed job."""
        job = job_manager.create_job(mock_panel, JobConfig())
        _update_status(job_manager.jobs_dir / job.id, JobStatus.COMPLETED)

        result = job_manager.delete_job(job.id)
        assert result is True

        # Directory should be removed
        assert not (job_manager.jobs_dir / job.id).exists()

    def test_load_job_results_not_found(self, job_manager):
        """Test load_job_results returns None for nonexistent job."""
        result = job_manager.load_job_results("nonexistent")
        assert result is None

    def test_load_job_results_not_completed(self, job_manager, mock_panel):
        """Test load_job_results returns None for incomplete job."""
        job = job_manager.create_job(mock_panel, JobConfig())

        result = job_manager.load_job_results(job.id)
        assert result is None

    def test_load_job_results_success(self, job_manager, mock_panel):
        """Test load_job_results loads model data."""
        job = job_manager.create_job(mock_panel, JobConfig())
        job_dir = job_manager.jobs_dir / job.id

        # Mark as completed
        _update_status(job_dir, JobStatus.COMPLETED)

        # Create mock model.pkl
        model_data = {"mmm": "mock", "results": "mock", "panel": mock_panel}
        with open(job_dir / "model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        result = job_manager.load_job_results(job.id)

        assert result is not None
        assert result["mmm"] == "mock"
        # Compare panel data by attributes
        pd.testing.assert_series_equal(result["panel"].y, mock_panel.y)

    def test_get_active_jobs(self, job_manager, mock_panel):
        """Test get_active_jobs returns pending and running jobs."""
        job1 = job_manager.create_job(mock_panel, JobConfig(name="Active1"))
        job2 = job_manager.create_job(mock_panel, JobConfig(name="Active2"))
        job3 = job_manager.create_job(mock_panel, JobConfig(name="Completed"))
        _update_status(job_manager.jobs_dir / job3.id, JobStatus.COMPLETED)

        active = job_manager.get_active_jobs()

        assert len(active) == 2
        names = {j.config.name for j in active}
        assert "Active1" in names
        assert "Active2" in names

    def test_get_completed_jobs(self, job_manager, mock_panel):
        """Test get_completed_jobs returns only completed jobs."""
        job1 = job_manager.create_job(mock_panel, JobConfig(name="Active"))
        job2 = job_manager.create_job(mock_panel, JobConfig(name="Completed"))
        _update_status(job_manager.jobs_dir / job2.id, JobStatus.COMPLETED)

        completed = job_manager.get_completed_jobs()

        assert len(completed) == 1
        assert completed[0].config.name == "Completed"

    def test_cleanup_old_jobs_by_age(self, job_manager, mock_panel):
        """Test cleanup_old_jobs removes old completed jobs."""
        job = job_manager.create_job(mock_panel, JobConfig(name="Old"))
        job_dir = job_manager.jobs_dir / job.id

        # Mark completed with old timestamp
        _update_status(job_dir, JobStatus.COMPLETED)
        with open(job_dir / "status.json") as f:
            status = json.load(f)
        status["completed_at"] = (
            datetime.now() - timedelta(days=60)
        ).isoformat()
        with open(job_dir / "status.json", "w") as f:
            json.dump(status, f)

        job_manager.cleanup_old_jobs(max_age_days=30)

        assert job_manager.get_job(job.id) is None

    def test_cleanup_old_jobs_by_count(self, job_manager, mock_panel):
        """Test cleanup_old_jobs respects max_jobs limit."""
        for i in range(10):
            job = job_manager.create_job(mock_panel, JobConfig(name=f"Job{i}"))
            _update_status(job_manager.jobs_dir / job.id, JobStatus.COMPLETED)

        job_manager.cleanup_old_jobs(max_jobs=5, max_age_days=365)

        completed = job_manager.get_completed_jobs()
        assert len(completed) == 5


# =============================================================================
# TestGetJobManager
# =============================================================================


class TestGetJobManager:
    """Tests for get_job_manager function."""

    def test_creates_singleton(self, tmp_path):
        """Test get_job_manager creates singleton."""
        # Reset the global
        import mmm_framework.jobs as jobs_module

        jobs_module._default_manager = None

        with patch.object(Path, "mkdir"):
            manager = get_job_manager()
            assert manager is not None

    def test_returns_existing_instance(self, tmp_path):
        """Test get_job_manager returns existing instance."""
        import mmm_framework.jobs as jobs_module

        jobs_module._default_manager = None

        jobs_dir = tmp_path / "jobs"

        manager1 = get_job_manager(jobs_dir)
        manager2 = get_job_manager()  # No path - should return same

        assert manager1 is manager2

    def test_creates_new_with_different_path(self, tmp_path):
        """Test get_job_manager creates new manager for different path."""
        import mmm_framework.jobs as jobs_module

        jobs_module._default_manager = None

        jobs_dir1 = tmp_path / "jobs1"
        jobs_dir2 = tmp_path / "jobs2"

        manager1 = get_job_manager(jobs_dir1)
        manager2 = get_job_manager(jobs_dir2)

        assert manager1.jobs_dir == jobs_dir1
        assert manager2.jobs_dir == jobs_dir2
        assert manager1 is not manager2


# =============================================================================
# TestSubmitModelJob
# =============================================================================


class TestSubmitModelJob:
    """Tests for submit_model_job convenience function."""

    @patch("mmm_framework.jobs.get_job_manager")
    def test_creates_job_config_from_params(self, mock_get_manager, mock_panel):
        """Test submit_model_job creates JobConfig from parameters."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_manager.submit_job.return_value = MagicMock()

        submit_model_job(
            mock_panel,
            name="Custom Job",
            description="Test description",
            n_chains=2,
            n_draws=500,
            trend_type="piecewise",
            trend_settings={"n_changepoints": 5},
            tags=["a", "b"],
        )

        call_args = mock_manager.submit_job.call_args
        config = call_args[0][1]  # Second positional arg

        assert config.name == "Custom Job"
        assert config.description == "Test description"
        assert config.n_chains == 2
        assert config.n_draws == 500
        assert config.trend_type == "piecewise"
        assert config.trend_settings == {"n_changepoints": 5}
        assert config.tags == ["a", "b"]

    @patch("mmm_framework.jobs.get_job_manager")
    def test_uses_default_job_manager(self, mock_get_manager, mock_panel):
        """Test submit_model_job uses default job manager."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_manager.submit_job.return_value = MagicMock()

        submit_model_job(mock_panel)

        mock_get_manager.assert_called_once()

    @patch("mmm_framework.jobs.get_job_manager")
    def test_returns_submitted_job(self, mock_get_manager, mock_panel):
        """Test submit_model_job returns the job."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        expected_job = MagicMock()
        mock_manager.submit_job.return_value = expected_job

        result = submit_model_job(mock_panel)

        assert result is expected_job
