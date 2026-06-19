# Ad-platform integrations — recommendation & connector contract

**Status:** the framework ships an ad-platform connector *contract* plus three
*stubs* (Google Ads, Meta Ads, TikTok Ads). The stubs document auth + the
recommended ingestion path and raise a guided `AdPlatformNotImplemented` until a
live API path is wired. This doc explains which platforms are easiest to
integrate and why the default recommendation routes through BigQuery rather than
hand-rolled API clients.

## TL;DR recommendation

> **Don't start with a direct API client. Land each platform's spend in
> BigQuery via a managed transfer, then read it with
> `BigQueryDataSource`.** It's less code to maintain, survives API-version
> churn, and reuses the GCS/BigQuery integration this framework already has.

The MMM only needs **periodic spend (and optionally impressions/clicks/
conversions) by channel and date** — not real-time, not row-level. That is
exactly what scheduled warehouse transfers are built for.

## Ranking — easiest first

| Platform | Direct-API ease | First-party Python SDK | Auth friction | Best path |
|---|---|---|---|---|
| **Meta Ads** | Easy | `facebook_business` (mature) | System-User token — no per-user OAuth | BigQuery via Fivetran/Supermetrics, or Insights API direct |
| **Google Ads** | Moderate | `google-ads` (mature) | OAuth2 **+ approved developer token** (the friction) | **BigQuery Data Transfer Service** (first-party, native connector) |
| **TikTok Ads** | Moderate | none first-party | OAuth app + long-lived token, REST | BigQuery via Fivetran/Supermetrics |

Notes:
- **Meta** is the easiest *direct* integration: the SDK is mature and a Business
  Manager System-User token sidesteps interactive OAuth.
- **Google Ads** has a great SDK, but the **developer-token approval** gate makes
  the managed BigQuery Data Transfer route strictly easier — and it's
  first-party Google, so it's the natural fit for a GCP-centric deployment.
- **TikTok** has no first-party Python SDK; a managed connector to BigQuery
  avoids maintaining a bespoke REST client.

Platforms intentionally **not** stubbed yet (higher effort, lower MMM demand):
LinkedIn Ads, Amazon Ads (profile/region sharding), Microsoft (Bing) Ads,
Snapchat, Pinterest, The Trade Desk. Add them by following the contract below.

## Why BigQuery-first

1. **One auth story.** `BigQueryDataSource` already authenticates via ADC — the
   same identity used for Vertex. No new secrets per platform.
2. **No client to maintain.** Ad APIs version aggressively and rate-limit;
   transfers (BigQuery Data Transfer Service for Google Ads; Fivetran /
   Supermetrics / Stitch for Meta/TikTok/others) absorb that churn.
3. **Backfill + scheduling for free.** Transfers handle historical backfill and
   incremental daily syncs; the model just reads the latest table.
4. **Auditable + governed.** Spend lands in your warehouse with lineage, not in
   an app's ephemeral memory.

Once the data is in BigQuery:

```python
from mmm_framework.integrations import BigQueryDataSource, BigQueryConfig

src = BigQueryDataSource(BigQueryConfig(project="acme", dataset="ads"))
df = src.read_dataframe(query="""
  SELECT date AS Period, campaign AS Campaign, 'GoogleAds_Search' AS VariableName,
         cost_micros / 1e6 AS VariableValue
  FROM `acme.ads.google_ads_campaign_stats`
  WHERE date BETWEEN '2023-01-01' AND '2024-12-31'
""")
# df is already MFF-shaped -> load_mff(df, mff_config)
```

## Connector contract (for direct API implementations)

`mmm_framework.integrations.ad_platforms`:

- **`AdPlatformConnector`** — ABC. Implement `fetch_spend(*, start, end,
  granularity="weekly", **kwargs) -> pd.DataFrame` returning **MFF long format**
  (`Period, Geography, Product, Campaign, Outlet, Creative, VariableName,
  VariableValue`). Override `test_connection()` to probe credentials.
- **`PlatformInfo`** — catalog metadata (`platform`, `label`, `status`, `ease`,
  `official_sdk`, `auth`, `recommended_path`, `metrics`). Drives the Settings
  "Data connections" catalog (`list_ad_platforms()`), ranked easiest-first.
- **`spend_to_mff(df, *, date_col, value_cols, geo_col=…, …)`** — reshape a tidy
  platform export into MFF long format. `value_cols` maps a source metric column
  to the `VariableName` it becomes, e.g.
  `{"cost": "GoogleAds_Search", "impressions": "GoogleAds_Search_impr"}`.

To make a stub live: implement `fetch_spend` (call the API, then `spend_to_mff`),
flip `PlatformInfo.status` to `"available"`, and add the SDK to the appropriate
optional-dependency group in `pyproject.toml`.

```python
from mmm_framework.integrations.ad_platforms import list_ad_platforms, build_ad_platform

list_ad_platforms()           # catalog, easiest-first, with sdk_installed flags
conn = build_ad_platform("meta_ads")
conn.fetch_spend(start="2023-01-01", end="2023-03-31")  # raises until implemented
```

## Where this surfaces

- **Catalog API:** `GET /integrations/catalog` returns both the data sources
  (GCS/BigQuery) and these ad platforms with their `installed`/`sdk_installed`
  flags and recommended paths.
- **UI:** the Settings → "Data connections" section renders the catalog so an
  analyst can see what's wired, what's a stub, and the recommended route.
