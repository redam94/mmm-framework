"""One-time migration: stamp an org_id onto legacy filesystem records.

Run BEFORE enabling MMM_AUTH on an existing single-tenant install so its
pre-existing data/configs/models/projects/budget-plans stay visible to its
tenant org. Idempotent — only records missing an org_id are stamped.

Usage (from the api/ directory)::

    python backfill_org.py <org_id>

Pass the org id you'll sign the install's owner into (e.g. the bootstrap org,
or 'org_default' which is what mmm_framework.auth.store bootstraps by default).
"""

import sys

from storage import backfill_org_id, get_storage


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 1
    org_id = sys.argv[1]
    counts = backfill_org_id(get_storage(), org_id)
    total = sum(counts.values())
    print(f"Stamped org_id={org_id!r} on {total} record(s): {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
