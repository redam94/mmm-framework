"""Subscription tiers, feature-gating map, usage metering, and plan limits.

The product is **open-core**: a free, self-hosted OSS tier, and paid *managed*
tiers whose differentiators are exactly the multi-tenancy / SSO / hosted-sandbox
capabilities the platform ships (i.e. the things a single-tenant self-host
doesn't need). A tier is recorded on ``organizations.plan``; this module maps
that to entitlements (feature flags + seat/project/fit limits), meters per-org
usage, and provides the limit checks routes enforce.

Dollar prices live in the buyer-facing collateral (``docs/pricing.html``) and are
illustrative — set your own. The *limits and feature gates here* are the
enforceable contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import store

# ----- feature flags (tie to real shipped capabilities) -----------------------

FEATURES = {
    "multi_tenant": "Organization isolation + RBAC across many users",
    "hosted_sandbox": "Managed, sandboxed agent code execution (container kernel)",
    "branding": "Client branding + branded reports/slides",
    "sso": "External IdP / SSO (OIDC/SAML)",
    "audit_export": "Export the tamper-evident audit log",
    "priority_support": "Priority support + SLA",
}


@dataclass(frozen=True)
class Plan:
    key: str
    name: str
    max_seats: int | None  # None = unlimited
    max_projects: int | None
    monthly_fit_quota: int | None
    features: frozenset[str]
    blurb: str

    def has(self, feature: str) -> bool:
        return feature in self.features


# Illustrative limits — tune freely; the gating mechanism is what matters.
PLANS: dict[str, Plan] = {
    # Plan limits apply only in HOSTED multi-tenant mode (auth on). Self-hosting
    # the OSS core with auth OFF runs as the dev principal with no limits at all —
    # that's the "open" in open-core, described in the collateral, not a billed plan.
    "free": Plan(
        key="free",
        name="Free",
        max_seats=2,
        max_projects=3,
        monthly_fit_quota=20,
        features=frozenset(),
        blurb="Get started on the managed platform — or self-host the OSS core with no limits.",
    ),
    "team": Plan(
        key="team",
        name="Team",
        max_seats=5,
        max_projects=10,
        monthly_fit_quota=200,
        features=frozenset({"multi_tenant", "hosted_sandbox", "branding"}),
        blurb="Managed multi-tenant workspace for a single team.",
    ),
    "business": Plan(
        key="business",
        name="Business",
        max_seats=25,
        max_projects=50,
        monthly_fit_quota=1000,
        features=frozenset(
            {
                "multi_tenant",
                "hosted_sandbox",
                "branding",
                "sso",
                "audit_export",
                "priority_support",
            }
        ),
        blurb="SSO, audit export, and headroom for an agency or larger org.",
    ),
    "enterprise": Plan(
        key="enterprise",
        name="Enterprise",
        max_seats=None,
        max_projects=None,
        monthly_fit_quota=None,
        features=frozenset(FEATURES.keys()),
        blurb="Unlimited scale, custom SLAs, on-prem/VPC, dedicated support.",
    ),
}

DEFAULT_PLAN = "free"


class PlanLimitError(Exception):
    """A plan seat/project/quota limit was hit (HTTP 402/409 at the route)."""


def get_plan(plan_key: str | None) -> Plan:
    return PLANS.get(plan_key or DEFAULT_PLAN, PLANS[DEFAULT_PLAN])


def entitlements_for_org(org_id: str, db_path: Path | str | None = None) -> Plan:
    org = store.get_organization(org_id, db_path=db_path)
    return get_plan(org.get("plan") if org else None)


def _month_start_ts() -> float:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


def org_usage(org_id: str, db_path: Path | str | None = None) -> dict:
    """Current usage vs. the org's plan limits (seats, projects, monthly fits)."""
    plan = entitlements_for_org(org_id, db_path=db_path)
    seats = store.count_org_members(org_id, db_path=db_path)
    projects = store.count_org_projects(org_id, db_path=db_path)
    fits = store.count_org_fits_since(org_id, _month_start_ts(), db_path=db_path)

    def _slot(used: int, limit: int | None) -> dict:
        return {
            "used": used,
            "limit": limit,
            "remaining": None if limit is None else max(limit - used, 0),
            "over": limit is not None and used > limit,
        }

    return {
        "plan": plan.key,
        "plan_name": plan.name,
        "features": sorted(plan.features),
        "seats": _slot(seats, plan.max_seats),
        "projects": _slot(projects, plan.max_projects),
        "fits_this_month": _slot(fits, plan.monthly_fit_quota),
    }


# ----- enforceable limit checks (raise PlanLimitError) ------------------------


def assert_within_seat_limit(org_id: str, db_path: Path | str | None = None) -> None:
    plan = entitlements_for_org(org_id, db_path=db_path)
    if plan.max_seats is None:
        return
    if store.count_org_members(org_id, db_path=db_path) >= plan.max_seats:
        raise PlanLimitError(
            f"Seat limit reached for the {plan.name} plan ({plan.max_seats}). "
            "Upgrade to add more members."
        )


def assert_within_project_limit(org_id: str, db_path: Path | str | None = None) -> None:
    plan = entitlements_for_org(org_id, db_path=db_path)
    if plan.max_projects is None:
        return
    if store.count_org_projects(org_id, db_path=db_path) >= plan.max_projects:
        raise PlanLimitError(
            f"Project limit reached for the {plan.name} plan ({plan.max_projects}). "
            "Upgrade to create more projects."
        )


def assert_within_fit_quota(org_id: str, db_path: Path | str | None = None) -> None:
    """Enforce the monthly fit quota. The usage was metered (``org_usage``) but
    never blocked — so a plan's ``monthly_fit_quota`` was advisory only. This
    closes that gap: a fit beyond the quota raises :class:`PlanLimitError`."""
    plan = entitlements_for_org(org_id, db_path=db_path)
    if plan.monthly_fit_quota is None:
        return  # unlimited tier
    used = store.count_org_fits_since(org_id, _month_start_ts(), db_path=db_path)
    if used >= plan.monthly_fit_quota:
        raise PlanLimitError(
            f"Monthly fit quota reached for the {plan.name} plan "
            f"({plan.monthly_fit_quota} fits/month). Upgrade your plan or wait "
            "for the next billing cycle."
        )
