"""
Main validation orchestrator.

Provides the ModelValidator class that coordinates all validation components.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from .channel_diagnostics import ChannelDiagnostics
from .config import ValidationConfig
from .posterior_predictive import PPCValidator
from .residual_diagnostics import ResidualDiagnostics
from .results import (
    ConvergenceSummary,
    LOOResults,
    ModelComparisonEntry,
    ModelComparisonResults,
    ValidationSummary,
    WAICResults,
)

if TYPE_CHECKING:
    import arviz as az


class ModelValidator:
    """
    Main validation orchestrator.

    Provides unified interface for all validation types.

    Examples
    --------
    >>> from mmm_framework.validation import ModelValidator, ValidationConfig
    >>>
    >>> # Quick validation
    >>> validator = ModelValidator(model, results)
    >>> summary = validator.quick_check()
    >>>
    >>> # Thorough validation with calibration
    >>> config = (ValidationConfigBuilder()
    ...     .thorough()
    ...     .with_calibration(lift_tests)
    ...     .build())
    >>> summary = validator.validate(config)
    >>> summary.to_html_report()
    """

    def __init__(
        self,
        model: Any,
        results: Any | None = None,
    ):
        """
        Initialize model validator.

        Parameters
        ----------
        model : Any
            Fitted model (BayesianMMM, NestedMMM, MultivariateMMM, etc.).
        results : Any, optional
            Model results container. If None, extracted from model.
        """
        self.model = model
        self.results = results

    def validate(
        self,
        config: ValidationConfig | None = None,
    ) -> ValidationSummary:
        """
        Run validation according to config.

        Parameters
        ----------
        config : ValidationConfig, optional
            Validation configuration. Defaults to standard validation.

        Returns
        -------
        ValidationSummary
            Comprehensive validation results.
        """
        config = config or ValidationConfig.standard()

        if config.verbose:
            logger.info(f"Starting {config.level.value} validation...")

        summary = ValidationSummary(
            model_name=type(self.model).__name__,
            validation_date=datetime.now().isoformat(),
        )

        # Always check convergence
        summary.convergence = self._check_convergence()
        if config.verbose:
            status = "Pass" if summary.convergence.converged else "Fail"
            logger.info(f"Convergence check: {status}")

        # Posterior predictive checks
        if config.run_ppc:
            try:
                ppc_validator = PPCValidator(self.model, config.ppc)
                summary.ppc = ppc_validator.run()
                if config.verbose:
                    status = "Pass" if summary.ppc.overall_pass else "Fail"
                    logger.info(f"PPC check: {status}")
            except Exception as e:
                logger.warning(f"PPC check failed: {e}")
                summary.warnings.append(f"PPC check failed: {str(e)}")

        # Residual diagnostics
        if config.run_residuals:
            try:
                residual_diagnostics = ResidualDiagnostics(self.model, config.residuals)
                summary.residuals = residual_diagnostics.run_all()
                if config.verbose:
                    status = "Pass" if summary.residuals.overall_adequate else "Fail"
                    logger.info(f"Residual diagnostics: {status}")
            except Exception as e:
                logger.warning(f"Residual diagnostics failed: {e}")
                summary.warnings.append(f"Residual diagnostics failed: {str(e)}")

        # Channel diagnostics
        if config.run_channel_diagnostics:
            try:
                channel_diagnostics = ChannelDiagnostics(
                    self.model, config.channel_diagnostics
                )
                summary.channel_diagnostics = channel_diagnostics.run_all()
                if config.verbose:
                    mc_status = (
                        "Warning"
                        if summary.channel_diagnostics.multicollinearity_warning
                        else "Pass"
                    )
                    conv_status = (
                        "Warning"
                        if summary.channel_diagnostics.convergence_warning
                        else "Pass"
                    )
                    logger.info(
                        f"Channel diagnostics: Multicollinearity={mc_status}, Convergence={conv_status}"
                    )
            except Exception as e:
                logger.warning(f"Channel diagnostics failed: {e}")
                summary.warnings.append(f"Channel diagnostics failed: {str(e)}")

        # Model comparison (LOO-CV, WAIC)
        if config.run_model_comparison:
            try:
                summary.model_comparison = self._run_model_comparison(config)
                if config.verbose:
                    logger.info(
                        f"Model comparison: LOO ELPD = {summary.model_comparison.models[0].loo.elpd_loo:.2f}"
                    )
            except Exception as e:
                logger.warning(f"Model comparison failed: {e}")
                summary.warnings.append(f"Model comparison failed: {str(e)}")

        # Cross-validation (expensive)
        if config.run_cross_validation:
            try:
                summary.cross_validation = self._run_cross_validation(config)
                if config.verbose:
                    logger.info(
                        f"Cross-validation: Mean R² = {summary.cross_validation.mean_r2:.4f}"
                    )
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                summary.warnings.append(f"Cross-validation failed: {str(e)}")

        # Sensitivity analysis (expensive)
        if config.run_sensitivity:
            try:
                summary.sensitivity = self._run_sensitivity_analysis(config)
                if config.verbose:
                    n_robust = len(summary.sensitivity.robust_parameters)
                    n_sensitive = len(summary.sensitivity.sensitive_parameters)
                    logger.info(
                        f"Sensitivity analysis: {n_robust} robust, {n_sensitive} sensitive parameters"
                    )
            except Exception as e:
                logger.warning(f"Sensitivity analysis failed: {e}")
                summary.warnings.append(f"Sensitivity analysis failed: {str(e)}")

        # Stability analysis (expensive)
        if config.run_stability:
            try:
                summary.stability = self._run_stability_analysis(config)
                if config.verbose:
                    logger.info(
                        f"Stability analysis: Score = {summary.stability.stability_score:.2f}"
                    )
            except Exception as e:
                logger.warning(f"Stability analysis failed: {e}")
                summary.warnings.append(f"Stability analysis failed: {str(e)}")

        # Calibration (requires external data)
        if config.run_calibration and config.lift_tests:
            try:
                summary.calibration = self._run_calibration(config)
                if config.verbose:
                    status = "Pass" if summary.calibration.calibrated else "Fail"
                    logger.info(f"Calibration check: {status}")
            except Exception as e:
                logger.warning(f"Calibration check failed: {e}")
                summary.warnings.append(f"Calibration check failed: {str(e)}")

        # Assess overall quality
        summary.overall_quality = self._assess_quality(summary)
        summary.critical_issues = self._identify_issues(summary)
        summary.recommendations = self._generate_recommendations(summary)

        if config.verbose:
            logger.info(f"Validation complete: {summary.overall_quality}")

        return summary

    def quick_check(self) -> ValidationSummary:
        """
        Run quick validation only.

        Returns
        -------
        ValidationSummary
            Quick validation results.
        """
        return self.validate(ValidationConfig.quick())

    def full_validation(self) -> ValidationSummary:
        """
        Run thorough validation.

        Returns
        -------
        ValidationSummary
            Comprehensive validation results.
        """
        return self.validate(ValidationConfig.thorough())

    def _check_convergence(self) -> ConvergenceSummary:
        """Check MCMC convergence diagnostics."""
        import arviz as az

        trace = self._get_trace()

        try:
            summary = az.summary(trace)
            rhat_max = float(summary["r_hat"].max())
            ess_bulk_min = float(summary["ess_bulk"].min())
            ess_tail_min = float(summary["ess_tail"].min())
        except Exception:
            rhat_max = 1.0
            ess_bulk_min = 1000.0
            ess_tail_min = 1000.0

        # Check for divergences
        divergences = 0
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            divergences = int(trace.sample_stats["diverging"].values.sum())

        converged = divergences == 0 and rhat_max < 1.01 and ess_bulk_min > 400

        return ConvergenceSummary(
            divergences=divergences,
            rhat_max=rhat_max,
            ess_bulk_min=ess_bulk_min,
            ess_tail_min=ess_tail_min,
            converged=converged,
        )

    def _run_model_comparison(
        self,
        config: ValidationConfig,
    ) -> ModelComparisonResults:
        """Run LOO-CV and/or WAIC for model comparison."""
        import arviz as az

        trace = self._get_trace()
        method = config.model_comparison.method

        # Ensure log likelihood is computed (required for LOO-CV and WAIC)
        trace = self._ensure_log_likelihood(trace)

        loo_results = None
        waic_results = None

        if method in ("loo", "both"):
            try:
                loo_data = az.loo(trace, pointwise=config.model_comparison.pointwise)
                loo_results = LOOResults(
                    elpd_loo=float(loo_data.elpd_loo),
                    se_elpd_loo=float(loo_data.se),
                    p_loo=float(loo_data.p_loo),
                    pareto_k=(
                        loo_data.pareto_k.values
                        if hasattr(loo_data, "pareto_k")
                        else None
                    ),
                    n_bad_k=(
                        int((loo_data.pareto_k > 0.7).sum())
                        if hasattr(loo_data, "pareto_k")
                        else 0
                    ),
                    pointwise_elpd=(
                        loo_data.loo_i.values
                        if config.model_comparison.pointwise
                        else None
                    ),
                )
            except Exception as e:
                logger.warning(f"LOO-CV computation failed: {e}")

        if method in ("waic", "both"):
            try:
                waic_data = az.waic(trace, pointwise=config.model_comparison.pointwise)
                waic_results = WAICResults(
                    waic=float(waic_data.waic),
                    se_waic=float(waic_data.se),
                    p_waic=float(waic_data.p_waic),
                    pointwise=(
                        waic_data.waic_i.values
                        if config.model_comparison.pointwise
                        else None
                    ),
                )
            except Exception as e:
                logger.warning(f"WAIC computation failed: {e}")

        entry = ModelComparisonEntry(
            name=type(self.model).__name__,
            loo=loo_results,
            waic=waic_results,
        )

        return ModelComparisonResults(models=[entry])

    def _ensure_log_likelihood(self, trace: Any) -> Any:
        """Ensure log likelihood is computed in the trace."""
        import arviz as az

        # Check if log likelihood already exists
        if hasattr(trace, "log_likelihood") and trace.log_likelihood is not None:
            return trace

        # Try to compute log likelihood using PyMC
        if hasattr(self.model, "model"):
            try:
                import pymc as pm

                pymc_model = self.model.model
                with pymc_model:
                    # Compute log likelihood for the observed variable
                    pm.compute_log_likelihood(trace)
                logger.info("Computed log likelihood for model comparison")
            except Exception as e:
                logger.warning(f"Could not compute log likelihood: {e}")

        return trace

    def _run_cross_validation(self, config: ValidationConfig):
        """Run time-series cross-validation (placeholder for full implementation)."""
        # This would require refitting the model multiple times
        # For now, return None or raise NotImplementedError
        raise NotImplementedError(
            "Time-series cross-validation requires refitting and is not yet implemented"
        )

    def _run_sensitivity_analysis(self, config: ValidationConfig):
        """Run sensitivity analysis (placeholder for full implementation)."""
        raise NotImplementedError(
            "Sensitivity analysis requires refitting and is not yet implemented"
        )

    def _run_stability_analysis(self, config: ValidationConfig):
        """Run stability analysis (placeholder for full implementation)."""
        raise NotImplementedError(
            "Stability analysis requires refitting and is not yet implemented"
        )

    def _run_calibration(self, config: ValidationConfig):
        """Run calibration check against lift tests (placeholder)."""
        raise NotImplementedError("Calibration check is not yet implemented")

    def _get_trace(self) -> Any:
        """Get ArviZ trace from model."""
        if hasattr(self.model, "_trace"):
            return self.model._trace
        elif hasattr(self.model, "trace"):
            return self.model.trace
        elif self.results is not None and hasattr(self.results, "trace"):
            return self.results.trace
        raise ValueError("Could not extract trace from model or results")

    def _assess_quality(
        self,
        summary: ValidationSummary,
    ) -> Literal["excellent", "good", "acceptable", "poor"]:
        """Assess overall model quality based on validation results."""
        issues = 0
        warnings = 0

        # Convergence issues
        if summary.convergence:
            if not summary.convergence.converged:
                issues += 1
            if summary.convergence.divergences > 0:
                issues += 1

        # PPC issues
        if summary.ppc:
            if not summary.ppc.overall_pass:
                warnings += len(summary.ppc.problematic_checks)

        # Residual issues
        if summary.residuals:
            if not summary.residuals.overall_adequate:
                failed_tests = sum(
                    1 for t in summary.residuals.test_results if not t.passed
                )
                warnings += failed_tests

        # Channel issues
        if summary.channel_diagnostics:
            if summary.channel_diagnostics.multicollinearity_warning:
                warnings += 1
            if summary.channel_diagnostics.convergence_warning:
                issues += 1

        # Model comparison issues
        if summary.model_comparison and summary.model_comparison.models:
            loo = summary.model_comparison.models[0].loo
            if loo and loo.n_bad_k > 0:
                if loo.n_bad_k > 5:
                    issues += 1
                else:
                    warnings += 1

        # Determine quality
        if issues == 0 and warnings <= 1:
            return "excellent"
        elif issues == 0 and warnings <= 3:
            return "good"
        elif issues <= 1:
            return "acceptable"
        else:
            return "poor"

    def _identify_issues(self, summary: ValidationSummary) -> list[str]:
        """Identify critical issues from validation results."""
        issues = []

        if summary.convergence:
            if summary.convergence.divergences > 0:
                issues.append(
                    f"MCMC has {summary.convergence.divergences} divergent transitions"
                )
            if summary.convergence.rhat_max >= 1.01:
                issues.append(
                    f"R-hat indicates non-convergence (max={summary.convergence.rhat_max:.3f})"
                )
            if summary.convergence.ess_bulk_min < 100:
                issues.append(
                    f"Very low effective sample size (min ESS={summary.convergence.ess_bulk_min:.0f})"
                )

        if summary.channel_diagnostics:
            if summary.channel_diagnostics.convergence_warning:
                non_converged = [
                    ch
                    for ch, r in summary.channel_diagnostics.convergence_by_channel.items()
                    if not r.converged
                ]
                issues.append(
                    f"Convergence issues for channels: {', '.join(non_converged)}"
                )

        return issues

    def _generate_recommendations(self, summary: ValidationSummary) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if summary.convergence:
            if summary.convergence.divergences > 0:
                recommendations.append(
                    "Consider reparameterizing the model or using stronger priors"
                )
            if summary.convergence.ess_bulk_min < 400:
                recommendations.append("Increase the number of samples or chains")

        if summary.residuals and not summary.residuals.overall_adequate:
            recommendations.extend(summary.residuals.recommendations)

        if summary.channel_diagnostics:
            if summary.channel_diagnostics.multicollinearity_warning:
                recommendations.append(
                    "Consider combining highly correlated channels or using regularization"
                )

        if summary.model_comparison and summary.model_comparison.models:
            loo = summary.model_comparison.models[0].loo
            if loo and loo.n_bad_k > 5:
                recommendations.append(
                    f"LOO-CV has {loo.n_bad_k} bad Pareto k values - consider using K-fold CV"
                )

        return recommendations


__all__ = ["ModelValidator"]
