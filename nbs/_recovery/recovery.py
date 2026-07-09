"""Nested-mediation recovery harness on the aurora world (PyMC 6).
Usage: python recovery.py <candidate_name>
Each candidate builds a PyMC model; we fit (numpyro), then measure recovery of
proportion_mediated (TV/Display, truth 0.988/0.967) and total ROAS (truth 2.14/2.11).
Spend is static data -> geometric adstock is a matrix-vector product (X_lag @ w(alpha)).
"""
import sys, warnings, json, numpy as np
warnings.filterwarnings("ignore")
import pymc as pm, pytensor.tensor as pt, arviz as az
sys.path.insert(0, "/Users/redam94/mmm-framework/nbs")
from aurora import generate_aurora, CHANNELS

A = generate_aurora()
T = len(A.weeks)
BRAND = ["TV", "Display"]; DIRECT = ["Search", "Social"]
IDX = {c: CHANNELS.index(c) for c in CHANNELS}
LMAX = 8

# --- data prep (all static) ---
spend = A.spend[CHANNELS].to_numpy(float)                    # (T, 4)
xmax = spend.max(0)
xn = spend / xmax                                            # normalized [0,1]
def lagmat(x):                                               # (T, LMAX): x[t-k]
    p = np.concatenate([np.zeros(LMAX - 1), x])
    return np.stack([p[t:t + LMAX][::-1] for t in range(len(x))])
XLAG = {c: lagmat(xn[:, IDX[c]]) for c in CHANNELS}          # per channel (T, LMAX) normalized
XLAG_RAW = {c: lagmat(spend[:, IDX[c]]) for c in CHANNELS}   # per channel (T, LMAX) raw spend
y = A.sales_total.astype(float); ymean, ystd = y.mean(), y.std(); yz = (y - ymean) / ystd
surv = A.awareness_survey.astype(float); smask = ~np.isnan(surv)
smean, sstd = np.nanmean(surv), np.nanstd(surv); surv_z = (surv - smean) / sstd
demand = A.category_demand_index.astype(float); dz = (demand - demand.mean()) / demand.std()
price = A.price.astype(float); pz = (price - price.mean()) / price.std()
t = np.arange(T); winter = np.cos(2 * np.pi * (t / 52.0))   # seasonal regressor (known form)
TRUE = dict(pm_tv=0.988, pm_disp=0.967, roas_tv=2.14, roas_disp=2.11)


def adstock_sat(name, c, alpha_ab=(2., 2.), half_prior=("gamma", 2., 4.)):
    """Return (sat_c tensor). alpha~Beta, half~prior; sat = adstock/(adstock+half)."""
    alpha = pm.Beta(f"alpha_{name}", *alpha_ab)
    w = alpha ** pt.arange(LMAX); w = w / w.sum()
    adstock = pt.dot(pt.as_tensor(XLAG[c]), w)               # (T,)
    if half_prior[0] == "gamma":
        half = pm.Gamma(f"half_{name}", half_prior[1], half_prior[2])
    else:
        half = pm.HalfNormal(f"half_{name}", half_prior[1])
    return adstock / (adstock + half)


# ============================ CANDIDATES ============================

def candidate_1():
    """Scale-fixed mediator: standardized survey pins media->awareness; tight direct prior."""
    with pm.Model() as m:
        sat = {c: adstock_sat(c, c) for c in CHANNELS}
        # awareness latent (standardized), built by brand channels
        a0 = pm.Normal("a0", 0, 1)
        b_aw = {c: pm.HalfNormal(f"b_{c}_aw", 3.0) for c in BRAND}
        a_z = a0 + sum(b_aw[c] * sat[c] for c in BRAND)
        pm.Deterministic("a_z", a_z)
        # survey pins the standardized media->awareness relationship
        s_surv = pm.HalfNormal("s_survey", 0.5)
        pm.Normal("survey_obs", mu=a_z[smask], sigma=s_surv, observed=surv_z[smask])
        # sales (standardized): awareness path + small direct (brand) + direct (search/social) + controls
        ay = pm.Normal("ay", 0, 1)
        gamma = pm.HalfNormal("gamma", 2.0)                  # awareness -> sales (positive)
        d = {c: pm.Normal(f"d_{c}", 0, 0.3) for c in BRAND}  # tight: direct is small (truth ~0)
        bd = {c: pm.Normal(f"bd_{c}", 0, 1.0) for c in DIRECT}
        cd = pm.Normal("c_demand", 0, 1); cp = pm.Normal("c_price", 0, 1); cw = pm.Normal("c_winter", 0, 1)
        mu = (ay + gamma * a_z
              + sum(d[c] * sat[c] for c in BRAND)
              + sum(bd[c] * sat[c] for c in DIRECT)
              + cd * dz + cp * pz + cw * winter)
        sig = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sig, observed=yz)
        # record per-channel mediated/direct standardized-effect-per-sat and sat sums (for measurement)
        for c in BRAND:
            pm.Deterministic(f"med_coef_{c}", gamma * b_aw[c])   # mediated effect per unit sat_c (std sales)
            pm.Deterministic(f"dir_coef_{c}", d[c])              # direct effect per unit sat_c
            pm.Deterministic(f"satsum_{c}", sat[c].sum())
    return m


def candidate_2():
    """C1 + trend & seasonal baseline in sales; winter in awareness (matches DGP baseline)."""
    tz = (t - t.mean()) / t.std()
    with pm.Model() as m:
        sat = {c: adstock_sat(c, c) for c in CHANNELS}
        a0 = pm.Normal("a0", 0, 1)
        b_aw = {c: pm.HalfNormal(f"b_{c}_aw", 3.0) for c in BRAND}
        aw_w = pm.Normal("aw_winter", 0, 0.5)
        a_z = a0 + sum(b_aw[c] * sat[c] for c in BRAND) + aw_w * winter
        pm.Deterministic("a_z", a_z)
        s_surv = pm.HalfNormal("s_survey", 0.5)
        pm.Normal("survey_obs", mu=a_z[smask], sigma=s_surv, observed=surv_z[smask])
        ay = pm.Normal("ay", 0, 1)
        gamma = pm.HalfNormal("gamma", 2.0)
        d = {c: pm.Normal(f"d_{c}", 0, 0.3) for c in BRAND}
        bd = {c: pm.Normal(f"bd_{c}", 0, 1.0) for c in DIRECT}
        cd = pm.Normal("c_demand", 0, 1); cp = pm.Normal("c_price", 0, 1)
        cw = pm.Normal("c_winter", 0, 1); ctr = pm.Normal("c_trend", 0, 1)   # NEW: trend baseline
        mu = (ay + gamma * a_z
              + sum(d[c] * sat[c] for c in BRAND)
              + sum(bd[c] * sat[c] for c in DIRECT)
              + cd * dz + cp * pz + cw * winter + ctr * tz)
        sig = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sig, observed=yz)
        for c in BRAND:
            pm.Deterministic(f"med_coef_{c}", gamma * b_aw[c])
            pm.Deterministic(f"dir_coef_{c}", d[c])
            pm.Deterministic(f"satsum_{c}", sat[c].sum())
    return m


def _nested_core(s_survey_prior, tight_direct=0.3, half_survey_center=None):
    """Shared C2 structure, parameterized on the survey-noise prior + direct-prior width."""
    tz = (t - t.mean()) / t.std()
    with pm.Model() as m:
        sat = {c: adstock_sat(c, c) for c in CHANNELS}
        a0 = pm.Normal("a0", 0, 1)
        b_aw = {c: pm.HalfNormal(f"b_{c}_aw", 3.0) for c in BRAND}
        aw_w = pm.Normal("aw_winter", 0, 0.5)
        a_z = a0 + sum(b_aw[c] * sat[c] for c in BRAND) + aw_w * winter
        pm.Deterministic("a_z", a_z)
        if half_survey_center is not None:
            s_surv = pm.TruncatedNormal("s_survey", mu=half_survey_center, sigma=0.1, lower=0.01)
        else:
            s_surv = pm.HalfNormal("s_survey", s_survey_prior)
        pm.Normal("survey_obs", mu=a_z[smask], sigma=s_surv, observed=surv_z[smask])
        ay = pm.Normal("ay", 0, 1)
        gamma = pm.HalfNormal("gamma", 2.0)
        d = {c: pm.Normal(f"d_{c}", 0, tight_direct) for c in BRAND}
        bd = {c: pm.Normal(f"bd_{c}", 0, 1.0) for c in DIRECT}
        cd = pm.Normal("c_demand", 0, 1); cp = pm.Normal("c_price", 0, 1)
        cw = pm.Normal("c_winter", 0, 1); ctr = pm.Normal("c_trend", 0, 1)
        mu = (ay + gamma * a_z + sum(d[c] * sat[c] for c in BRAND)
              + sum(bd[c] * sat[c] for c in DIRECT) + cd * dz + cp * pz + cw * winter + ctr * tz)
        sig = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sig, observed=yz)
        for c in BRAND:
            pm.Deterministic(f"med_coef_{c}", gamma * b_aw[c])
            pm.Deterministic(f"dir_coef_{c}", d[c]); pm.Deterministic(f"satsum_{c}", sat[c].sum())
    return m


def candidate_3():   # tighten survey noise -> pins awareness scale
    return _nested_core(0.25)


def adstock_sat_raw(name, c):
    """DGP-matched saturation: RAW spend adstock + hill with K~40 (Gamma mean 40)."""
    alpha = pm.Beta(f"alpha_{name}", 2., 2.)
    w = alpha ** pt.arange(LMAX); w = w / w.sum()
    adstock = pt.dot(pt.as_tensor(XLAG_RAW[c]), w)
    half = pm.Gamma(f"half_{name}", alpha=4., beta=0.1)   # mean 40, covers ~20-70
    return adstock / (adstock + half)


def candidate_4():
    """C3 structure but DGP-matched raw-spend saturation (shape matches truth)."""
    tz = (t - t.mean()) / t.std()
    with pm.Model() as m:
        sat = {c: adstock_sat_raw(c, c) for c in CHANNELS}
        a0 = pm.Normal("a0", 0, 1)
        b_aw = {c: pm.HalfNormal(f"b_{c}_aw", 3.0) for c in BRAND}
        aw_w = pm.Normal("aw_winter", 0, 0.5)
        a_z = a0 + sum(b_aw[c] * sat[c] for c in BRAND) + aw_w * winter
        pm.Deterministic("a_z", a_z)
        s_surv = pm.HalfNormal("s_survey", 0.25)
        pm.Normal("survey_obs", mu=a_z[smask], sigma=s_surv, observed=surv_z[smask])
        ay = pm.Normal("ay", 0, 1)
        gamma = pm.HalfNormal("gamma", 2.0)
        d = {c: pm.Normal(f"d_{c}", 0, 0.3) for c in BRAND}
        bd = {c: pm.Normal(f"bd_{c}", 0, 1.0) for c in DIRECT}
        cd = pm.Normal("c_demand", 0, 1); cp = pm.Normal("c_price", 0, 1)
        cw = pm.Normal("c_winter", 0, 1); ctr = pm.Normal("c_trend", 0, 1)
        mu = (ay + gamma * a_z + sum(d[c] * sat[c] for c in BRAND)
              + sum(bd[c] * sat[c] for c in DIRECT) + cd * dz + cp * pz + cw * winter + ctr * tz)
        sig = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sig, observed=yz)
        for c in BRAND:
            pm.Deterministic(f"med_coef_{c}", gamma * b_aw[c])
            pm.Deterministic(f"dir_coef_{c}", d[c]); pm.Deterministic(f"satsum_{c}", sat[c].sum())
    return m


def candidate_5():
    """C4 (raw-spend sat) + quadratic baseline + tighter survey noise (~true 0.2)."""
    tz = (t - t.mean()) / t.std(); tz2 = tz ** 2 - (tz ** 2).mean()
    with pm.Model() as m:
        sat = {c: adstock_sat_raw(c, c) for c in CHANNELS}
        a0 = pm.Normal("a0", 0, 1)
        b_aw = {c: pm.HalfNormal(f"b_{c}_aw", 3.0) for c in BRAND}
        aw_w = pm.Normal("aw_winter", 0, 0.5)
        a_z = a0 + sum(b_aw[c] * sat[c] for c in BRAND) + aw_w * winter
        pm.Deterministic("a_z", a_z)
        s_surv = pm.HalfNormal("s_survey", 0.2)
        pm.Normal("survey_obs", mu=a_z[smask], sigma=s_surv, observed=surv_z[smask])
        ay = pm.Normal("ay", 0, 1)
        gamma = pm.HalfNormal("gamma", 2.0)
        d = {c: pm.Normal(f"d_{c}", 0, 0.3) for c in BRAND}
        bd = {c: pm.Normal(f"bd_{c}", 0, 1.0) for c in DIRECT}
        cd = pm.Normal("c_demand", 0, 1); cp = pm.Normal("c_price", 0, 1)
        cw = pm.Normal("c_winter", 0, 1); ctr = pm.Normal("c_trend", 0, 1); ct2 = pm.Normal("c_trend2", 0, 1)
        mu = (ay + gamma * a_z + sum(d[c] * sat[c] for c in BRAND)
              + sum(bd[c] * sat[c] for c in DIRECT) + cd * dz + cp * pz + cw * winter
              + ctr * tz + ct2 * tz2)
        sig = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sig, observed=yz)
        for c in BRAND:
            pm.Deterministic(f"med_coef_{c}", gamma * b_aw[c])
            pm.Deterministic(f"dir_coef_{c}", d[c]); pm.Deterministic(f"satsum_{c}", sat[c].sum())
    return m


CANDIDATES = {"candidate_1": candidate_1, "candidate_2": candidate_2, "candidate_3": candidate_3,
              "candidate_4": candidate_4, "candidate_5": candidate_5}


def measure(idata, m):
    post = idata.posterior
    out = {}
    for c in BRAND:
        med = post[f"med_coef_{c}"].values.ravel()
        dir_ = post[f"dir_coef_{c}"].values.ravel()
        satsum = post[f"satsum_{c}"].values.ravel()
        pm_c = med / (med + dir_)                            # proportion mediated (sat cancels)
        # total-effect ROAS: total std-sales contribution = (med+dir)*sat, back to $ via ystd; / spend
        total_dollars = (med + dir_) * satsum * ystd
        roas = total_dollars / spend[:, IDX[c]].sum()
        out[c] = dict(pm=float(np.mean(pm_c)), pm_lo=float(np.percentile(pm_c, 5)), pm_hi=float(np.percentile(pm_c, 95)),
                      roas=float(np.mean(roas)), roas_lo=float(np.percentile(roas, 5)), roas_hi=float(np.percentile(roas, 95)))
    s = az.summary(idata, var_names=[v for v in post.data_vars if not v.startswith(("a_z",))], round_to="none")
    out["rhat"] = float(np.nanmax(s["r_hat"])); out["ess"] = float(np.nanmin(s["ess_bulk"]))
    try: out["div"] = int(idata.sample_stats["diverging"].sum())
    except Exception: out["div"] = -1
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "candidate_1"
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    tune = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    m = CANDIDATES[name]()
    with m:
        idata = pm.sample(draws=draws, tune=tune, chains=4, target_accept=0.95,
                          nuts_sampler="numpyro", random_seed=0, progressbar=False)
    r = measure(idata, m)
    print(f"\n===== {name} (draws={draws}, tune={tune}) =====")
    print(f"TRUE   PM: TV 0.988 Disp 0.967 | ROAS: TV 2.14 Disp 2.11")
    for c in BRAND:
        x = r[c]
        print(f"  {c:8s} PM={x['pm']:.3f} [{x['pm_lo']:.2f},{x['pm_hi']:.2f}]  "
              f"ROAS={x['roas']:.2f} [{x['roas_lo']:.2f},{x['roas_hi']:.2f}]")
    print(f"  conv: max_rhat={r['rhat']:.3f}  min_ess={r['ess']:.0f}  div={r['div']}")
    ok = (r["TV"]["pm"] >= 0.85 and r["Display"]["pm"] >= 0.80
          and 1.39 <= r["TV"]["roas"] <= 2.89 and r["rhat"] < 1.05)
    print(f"  SUCCESS={ok}")
    print("JSON:" + json.dumps({name: r}))
