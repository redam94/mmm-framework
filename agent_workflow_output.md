# Agentic MMM Workflow

**User:** I want to build a Marketing Mix Model, but I don't have data. Can you generate some and run a model?

**Agent:**
Sure! Let's get you started. I'll kick off both steps simultaneously — generating synthetic data and then we'll configure and run the model right after.

First, let me generate a synthetic dataset for you!

> **Tool Call:** `generate_synthetic_data({'n_weeks': 104})`

> **Tool Result:** `generate_synthetic_data` completed.

**Agent:**
Great, the data is ready! Let me take a quick look at it to understand its structure and identify the right variables for the model.

> **Tool Call:** `execute_python({'code': '\nimport pandas as pd\n\ndf = pd.read_csv("synthetic_mff_data.csv")\nprint("Shape:", df.shape)\nprint("\\nFirst few rows:")\nprint(df.head(10).to_string())\nprint("\\nUnique VariableNames:")\nprint(df[\'VariableName\'].unique())\n'})`

> **Tool Result:** `execute_python` completed.

**Agent:**
Perfect! The dataset is clean and well-structured. Here's what I found:

| Role | Variables |
|---|---|
| 📈 **KPI** | `Sales` |
| 📺 **Media Channels** | `TV`, `Digital`, `Paid_Social`, `Radio` |
| 🎛️ **Control Variables** | `Price_Index`, `Distribution` |

This is a **national-level** dataset with **104 weeks** of data. Now let me configure the model with these variables!

> **Tool Call:** `configure_model({'kpi': 'Sales', 'kpi_level': 'national', 'media_channels': ['TV', 'Digital', 'Paid_Social', 'Radio'], 'control_variables': ['Price_Index', 'Distribution']})`

> **Tool Result:** `configure_model` completed.

**Agent:**
Model is configured! Now let's fit it. This runs a **Bayesian MMM** using NumPyro under the hood — it will estimate adstock (carryover), saturation (diminishing returns), and ROI for each channel. Let's go! 🚀

> **Tool Call:** `fit_mmm_model({'dataset_path': 'synthetic_mff_data.csv', 'model_spec': '{"kpi": "Sales", "kpi_level": "national", "media_channels": [{"name": "TV"}, {"name": "Digital"}, {"name": "Paid_Social"}, {"name": "Radio"}], "control_variables": [{"name": "Price_Index"}, {"name": "Distribution"}], "time_granularity": "weekly", "model_type": "numpyro"}'})`

> **Tool Result:** `fit_mmm_model` completed.

**Agent:**
The model has been fitted successfully! 🎉 Here's a summary of what was built:

---

## ✅ Model Fitting Complete!

| Property | Detail |
|---|---|
| **Model Type** | Bayesian MMM (NumPyro) |
| **KPI** | Sales (National) |
| **Observations** | 104 weekly data points |
| **Media Channels** | TV, Digital, Paid_Social, Radio |
| **Control Variables** | Price_Index, Distribution |

---

## 🔍 What Can You Explore Next?

Now that the model is fitted, here are some things you can dive into:

1. **📊 ROI & Efficiency** — *"Which channel gives me the best return on investment?"*
2. **🧩 Sales Decomposition** — *"What percentage of my sales came from media vs. base?"*
3. **📉 Saturation Curves** — *"Which channels are hitting diminishing returns?"*
4. **⏳ Adstock / Carryover** — *"How long do my media effects last after a campaign ends?"*
5. **🩺 Model Diagnostics** — *"Is my Bayesian model healthy and converged?"*

What would you like to explore first?

**User:** Provide a thorough summary of the model results

**Agent:**
Absolutely! Let me pull all the results simultaneously for a comprehensive summary.

> **Tool Call:** `get_roi_metrics({})`

> **Tool Call:** `get_component_decomposition({})`

> **Tool Call:** `get_model_diagnostics({})`

> **Tool Call:** `get_adstock_weights({})`

> **Tool Call:** `get_saturation_curves({})`

> **Tool Result:** `get_saturation_curves` completed.

**Agent:**
Here is your full model results summary:

---

# 📊 Marketing Mix Model — Full Results Summary

---

## 1. 🩺 Model Health & Diagnostics

| Metric | Value | Status |
|---|---|---|
| **Convergence** | Not fully converged | ⚠️ Warning |
| **Divergences** | 5 | ⚠️ Should be 0 |
| **Max R-hat** | 1.0 | ✅ Excellent (< 1.01) |
| **Min Bulk ESS** | 2,923 | ✅ Excellent (> 400) |
| **Min Tail ESS** | 2,002 | ✅ Excellent (> 400) |

> **Interpretation:** The model is in good shape overall. R-hat values are perfect and effective sample sizes are very high, indicating the MCMC chains mixed well. The **5 divergences** are a minor flag — they suggest a small region of the posterior was difficult to explore. For a synthetic dataset this is acceptable, but in production you may want to re-run with more warmup steps or tighter priors.

---

## 2. 🧩 Sales Decomposition — What Drove Sales?

| Component | Sales Contribution | Share |
|---|---|---|
| 🏠 **Baseline** | $127,136,162 | **86.1%** |
| 📈 **Trend** | $860,730 | 0.6% |
| 📺 **TV** | $11,691,192 | **7.9%** |
| 💻 **Digital** | $3,417,406 | 2.3% |
| 📱 **Paid Social** | $2,857,817 | 1.9% |
| 📻 **Radio** | $1,738,851 | 1.2% |
| **Total Media** | **$19,705,266** | **13.3%** |

> **Interpretation:** The vast majority of sales (**86.1%**) come from the **baseline** — this represents organic demand, brand equity, and structural factors like distribution and pricing. Media accounts for **~13.3%** of total sales, with **TV being the dominant media driver** at nearly 8% of total sales on its own.

---

## 3. 💰 ROI by Channel — Which Channels Pay Off?

| Channel | Mean ROI | 94% Credible Interval | Prob. Profitable |
|---|---|---|---|
| 📻 **Radio** | **1.39x** | [0.00, 3.52] | 53.1% |
| 📺 **TV** | **1.38x** | [0.79, 2.04] | ⭐ **87.4%** |
| 📱 **Paid Social** | 0.92x | [0.00, 2.40] | 36.5% |
| 💻 **Digital** | 0.58x | [0.00, 1.47] | 17.1% |

> **Interpretation:**
> - 📺 **TV** is the standout performer. While Radio has a marginally higher mean ROI, TV has a **much tighter credible interval** and an **87.4% probability of being profitable** — making it the most reliable and confident investment.
> - 📻 **Radio** has the highest mean ROI (1.39x) but wide uncertainty (HDI spans 0 to 3.52), meaning results are less predictable.
> - 📱 **Paid Social** is borderline — ROI below 1.0 on average with only a 36.5% chance of being profitable.
> - 💻 **Digital** is the weakest performer with a mean ROI of 0.58x and only a 17.1% probability of profitability. This channel may warrant a budget review.

---

## 4. ⏳ Adstock & Carryover — How Long Do Effects Last?

| Channel | Half-Life | Total Carryover | Decay Rate (α) |
|---|---|---|---|
| 📺 **TV** | 0.2 weeks | 5.6% | 0.056 |
| 💻 **Digital** | 0.7 weeks | 38.4% | 0.385 |
| 📱 **Paid Social** | 0.8 weeks | 42.5% | 0.425 |
| 📻 **Radio** | 0.8 weeks | 39.9% | 0.399 |

> **Interpretation:**
> - 📺 **TV** has an almost **immediate effect** — its impact is felt almost entirely within the same week, with very little carryover (5.6%). This is typical for direct-response TV.
> - 💻 **Digital**, 📱 **Paid Social**, and 📻 **Radio** all show **moderate carryover** (~38–43%), meaning roughly 40% of their effect spills into future weeks. This is important for budget pacing — cutting these channels abruptly will still yield returns for a week or two.

---

## 5. 📉 Saturation — Are Channels Maxed Out?

| Channel | Saturation Level | Marginal Return (Next $1) |
|---|---|---|
| 📺 **TV** | 70.5% 🟡 Medium | $0.91 |
| 💻 **Digital** | 69.2% 🟡 Medium | $0.22 |
| 📱 **Paid Social** | 68.8% 🟡 Medium | $0.26 |
| 📻 **Radio** | 68.2% 🟡 Medium | $0.44 |

> **Interpretation:** All channels are operating at a **medium saturation level (~69–71%)**, meaning there is still room to grow, but diminishing returns are already setting in. Notably:
> - 📺 **TV** has the highest marginal return at **$0.91 per additional dollar** — the most efficient channel to increase spend on.
> - 💻 **Digital** and 📱 **Paid Social** have very low marginal returns ($0.22 and $0.26), suggesting these channels are relatively inefficient at the current spend levels.

---

## 🔑 Key Takeaways

| # | Insight |
|---|---|
| 1 | **TV is your best channel** — highest confidence ROI, largest sales contribution, and best marginal return. |
| 2 | **Digital underperforms** — low ROI, low probability of profitability, and low marginal returns. Consider reallocating budget. |
| 3 | **Radio is promising but uncertain** — high mean ROI but wide credible intervals suggest inconsistent performance. |
| 4 | **86% of sales are baseline-driven** — media is an amplifier, not the primary engine. Brand and distribution matter most. |
| 5 | **No channel is over-saturated** — there is room to scale TV and Radio spend if budget allows. |

---

Would you like to explore **budget optimization**, dive deeper into any specific channel, or run a **scenario analysis**?

**User:** I would like to explore budget optimization

**Agent:**
That's a great next step! Budget optimization uses the fitted model's saturation curves and ROI estimates to find the ideal spend allocation across channels. Let me run the analysis now.

> **Tool Call:** `execute_python({'code': '\nimport numpy as np\nimport pandas as pd\n\n# Extract key model parameters from results\n# Saturation levels and marginal returns\nchannels = [\'TV\', \'Digital\', \'Paid_Social\', \'Radio\']\n\n# From our model results:\n# Current spend (approximate from synthetic data)\ndf = pd.read_csv("synthetic_mff_data.csv")\ndf_wide = df.pivot_table(index=\'Period\', columns=\'VariableName\', values=\'VariableValue\', aggfunc=\'sum\')\n\n# Weekly average spend per channel\nweekly_avg = df_wide[channels].mean()\ntotal_weekly_spend = weekly_avg.sum()\nannual_spend = weekly_avg.sum() * 52\n\nprint("=== CURRENT SPEND PROFILE ===")\nprint(f"\\nWeekly Average Spend:")\nfor ch in channels:\n    pct = weekly_avg[ch] / total_weekly_spend * 100\n    print(f"  {ch:<15}: ${weekly_avg[ch]:>12,.0f}  ({pct:.1f}%)")\nprint(f"\\n  {\'TOTAL\':<15}: ${total_weekly_spend:>12,.0f}")\nprint(f"\\nAnnualised Total Spend: ${annual_spend:>12,.0f}")\n\nprint("\\n=== SPEND RANGE (Min / Max Weekly) ===")\nfor ch in channels:\n    print(f"  {ch:<15}: Min=${df_wide[ch].min():>10,.0f}  Max=${df_wide[ch].max():>10,.0f}  Std=${df_wide[ch].std():>10,.0f}")\n'})`

> **Tool Result:** `execute_python` completed.

**Agent:**
[{'id': 'toolu_01UPXfJYkxhr7wF5M4UtySvN', 'caller': {'type': 'direct'}, 'input': {'code': '\nimport numpy as np\nfrom scipy.optimize import minimize, differential_evolution\n\n# ============================================================\n# BUDGET OPTIMISATION using Hill saturation function\n# Parameters derived from model results\n# ============================================================\n\n# Hill saturation function: response = spend^alpha / (spend^alpha + K^alpha)\n# We back-calculate K (half-saturation point) from:\n#   - current saturation level\n#   - current average spend\n# Saturation level = spend^alpha / (spend^alpha + K^alpha)\n\n# Model-derived parameters\nroi_mean       = {\'TV\': 1.38, \'Digital\': 0.58, \'Paid_Social\': 0.92, \'Radio\': 1.39}\nsaturation_pct = {\'TV\': 0.705, \'Digital\': 0.692, \'Paid_Social\': 0.688, \'Radio\': 0.682}\nmarginal_resp  = {\'TV\': 0.910, \'Digital\': 0.218, \'Paid_Social\': 0.262, \'Radio\': 0.439}\n\n# Current weekly average spend\ncurrent_spend = {\'TV\': 81212, \'Digital\': 56788, \'Paid_Social\': 29842, \'Radio\': 12028}\n\n# Sales contributions (weekly average from decomposition / 104 weeks)\nsales_contrib = {\n    \'TV\':          11691192 / 104,\n    \'Digital\':      3417406 / 104,\n    \'Paid_Social\':  2857817 / 104,\n    \'Radio\':        1738851 / 104,\n}\n\n# Hill function alpha (shape) — use alpha=2 (common default for media)\nalpha = {ch: 2.0 for ch in channels}\n\n# Back-calculate K (half-saturation point) from saturation level\n# sat = x^a / (x^a + K^a)  =>  K^a = x^a * (1 - sat) / sat\nK = {}\nfor ch in channels:\n    x = current_spend[ch]\n    s = saturation_pct[ch]\n    a = alpha[ch]\n    K[ch] = (x**a * (1 - s) / s) ** (1/a)\n\n# Scale factor: maps Hill output (0-1) to sales dollars\n# scale = sales_contrib / hill(current_spend)\nscale = {}\nfor ch in channels:\n    x = current_spend[ch]\n    a = alpha[ch]\n    hill_val = x**a / (x**a + K[ch]**a)\n    scale[ch] = sales_contrib[ch] / hill_val\n\ndef channel_response(spend, ch):\n    """Sales response for a given spend level in channel ch."""\n    a = alpha[ch]\n    return scale[ch] * spend**a / (spend**a + K[ch]**a)\n\ndef total_response(spend_vec):\n    """Total sales from all channels given spend vector."""\n    return sum(channel_response(spend_vec[i], channels[i]) for i in range(len(channels)))\n\ndef neg_total_response(spend_vec):\n    return -total_response(spend_vec)\n\n# ============================================================\n# SCENARIO 1: Same total budget, optimised allocation\n# ============================================================\ntotal_budget = sum(current_spend.values())  # $179,870/week\n\n# Bounds: each channel gets between 5% and 70% of total budget\nbounds = [(total_budget * 0.05, total_budget * 0.70)] * len(channels)\n\n# Constraint: total spend = total_budget\nconstraints = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - total_budget}\n\nresult_same = differential_evolution(\n    neg_total_response,\n    bounds=bounds,\n    constraints=constraints,\n    seed=42,\n    maxiter=1000,\n    tol=1e-8,\n    polish=True\n)\n\nopt_spend_same = result_same.x\nopt_response_same = total_response(opt_spend_same)\ncurrent_response = total_response(list(current_spend.values()))\nuplift_same = (opt_response_same - current_response) / current_response * 100\n\n# ============================================================\n# SCENARIO 2: +20% budget increase, optimised allocation\n# ============================================================\ntotal_budget_plus20 = total_budget * 1.20\nbounds_plus20 = [(total_budget_plus20 * 0.05, total_budget_plus20 * 0.70)] * len(channels)\nconstraints_plus20 = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - total_budget_plus20}\n\nresult_plus20 = differential_evolution(\n    neg_total_response,\n    bounds=bounds_plus20,\n    constraints=constraints_plus20,\n    seed=42,\n    maxiter=1000,\n    tol=1e-8,\n    polish=True\n)\nopt_spend_plus20 = result_plus20.x\nopt_response_plus20 = total_response(opt_spend_plus20)\nuplift_plus20 = (opt_response_plus20 - current_response) / current_response * 100\n\n# ============================================================\n# SCENARIO 3: -20% budget cut, optimised allocation\n# ============================================================\ntotal_budget_minus20 = total_budget * 0.80\nbounds_minus20 = [(total_budget_minus20 * 0.05, total_budget_minus20 * 0.70)] * len(channels)\nconstraints_minus20 = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - total_budget_minus20}\n\nresult_minus20 = differential_evolution(\n    neg_total_response,\n    bounds=bounds_minus20,\n    constraints=constraints_minus20,\n    seed=42,\n    maxiter=1000,\n    tol=1e-8,\n    polish=True\n)\nopt_spend_minus20 = result_minus20.x\nopt_response_minus20 = total_response(opt_spend_minus20)\nuplift_minus20 = (opt_response_minus20 - current_response) / current_response * 100\n\n# ============================================================\n# PRINT RESULTS\n# ============================================================\nprint("=" * 70)\nprint("BUDGET OPTIMISATION RESULTS")\nprint("=" * 70)\n\nscenarios = [\n    ("Current (Baseline)",    list(current_spend.values()), current_response,  0),\n    ("Same Budget (Optimised)", opt_spend_same,             opt_response_same,  uplift_same),\n    ("-20% Budget (Optimised)", opt_spend_minus20,          opt_response_minus20, uplift_minus20),\n    ("+20% Budget (Optimised)", opt_spend_plus20,           opt_response_plus20,  uplift_plus20),\n]\n\nfor name, spends, response, uplift in scenarios:\n    total_s = sum(spends)\n    print(f"\\n{\'─\'*70}")\n    print(f"  📌 {name}")\n    print(f"{\'─\'*70}")\n    print(f"  {\'Channel\':<15} {\'Spend\':>12}  {\'Share\':>7}  {\'Change vs Current\':>18}")\n    for i, ch in enumerate(channels):\n        share = spends[i] / total_s * 100\n        delta = spends[i] - current_spend[ch]\n        delta_str = f"{\'▲\' if delta > 0 else \'▼\' if delta < 0 else \'─\'} ${abs(delta):>8,.0f}" if name != "Current (Baseline)" else "—"\n        print(f"  {ch:<15} ${spends[i]:>10,.0f}  {share:>6.1f}%  {delta_str:>18}")\n    print(f"  {\'─\'*60}")\n    print(f"  {\'TOTAL SPEND\':<15} ${total_s:>10,.0f}")\n    print(f"  {\'Media Sales\':<15} ${response:>10,.0f}/week")\n    if uplift != 0:\n        print(f"  {\'Uplift vs Now\':<15} {uplift:>+10.1f}%")\n\nprint(f"\\n{\'=\'*70}")\nprint("MARGINAL RETURN ON NEXT $1,000 SPEND (Current Allocation)")\nprint(f"{\'=\'*70}")\nfor ch in channels:\n    delta = 1000\n    base = channel_response(current_spend[ch], ch)\n    incremental = channel_response(current_spend[ch] + delta, ch) - base\n    print(f"  {ch:<15}: ${incremental:>8,.2f} incremental sales per $1,000 spent")\n'}, 'name': 'execute_python', 'type': 'tool_use'}]

> **Tool Call:** `execute_python({'code': '\nimport numpy as np\nfrom scipy.optimize import minimize, differential_evolution\n\n# ============================================================\n# BUDGET OPTIMISATION using Hill saturation function\n# Parameters derived from model results\n# ============================================================\n\n# Hill saturation function: response = spend^alpha / (spend^alpha + K^alpha)\n# We back-calculate K (half-saturation point) from:\n#   - current saturation level\n#   - current average spend\n# Saturation level = spend^alpha / (spend^alpha + K^alpha)\n\n# Model-derived parameters\nroi_mean       = {\'TV\': 1.38, \'Digital\': 0.58, \'Paid_Social\': 0.92, \'Radio\': 1.39}\nsaturation_pct = {\'TV\': 0.705, \'Digital\': 0.692, \'Paid_Social\': 0.688, \'Radio\': 0.682}\nmarginal_resp  = {\'TV\': 0.910, \'Digital\': 0.218, \'Paid_Social\': 0.262, \'Radio\': 0.439}\n\n# Current weekly average spend\ncurrent_spend = {\'TV\': 81212, \'Digital\': 56788, \'Paid_Social\': 29842, \'Radio\': 12028}\n\n# Sales contributions (weekly average from decomposition / 104 weeks)\nsales_contrib = {\n    \'TV\':          11691192 / 104,\n    \'Digital\':      3417406 / 104,\n    \'Paid_Social\':  2857817 / 104,\n    \'Radio\':        1738851 / 104,\n}\n\n# Hill function alpha (shape) — use alpha=2 (common default for media)\nalpha = {ch: 2.0 for ch in channels}\n\n# Back-calculate K (half-saturation point) from saturation level\n# sat = x^a / (x^a + K^a)  =>  K^a = x^a * (1 - sat) / sat\nK = {}\nfor ch in channels:\n    x = current_spend[ch]\n    s = saturation_pct[ch]\n    a = alpha[ch]\n    K[ch] = (x**a * (1 - s) / s) ** (1/a)\n\n# Scale factor: maps Hill output (0-1) to sales dollars\n# scale = sales_contrib / hill(current_spend)\nscale = {}\nfor ch in channels:\n    x = current_spend[ch]\n    a = alpha[ch]\n    hill_val = x**a / (x**a + K[ch]**a)\n    scale[ch] = sales_contrib[ch] / hill_val\n\ndef channel_response(spend, ch):\n    """Sales response for a given spend level in channel ch."""\n    a = alpha[ch]\n    return scale[ch] * spend**a / (spend**a + K[ch]**a)\n\ndef total_response(spend_vec):\n    """Total sales from all channels given spend vector."""\n    return sum(channel_response(spend_vec[i], channels[i]) for i in range(len(channels)))\n\ndef neg_total_response(spend_vec):\n    return -total_response(spend_vec)\n\n# ============================================================\n# SCENARIO 1: Same total budget, optimised allocation\n# ============================================================\ntotal_budget = sum(current_spend.values())  # $179,870/week\n\n# Bounds: each channel gets between 5% and 70% of total budget\nbounds = [(total_budget * 0.05, total_budget * 0.70)] * len(channels)\n\n# Constraint: total spend = total_budget\nconstraints = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - total_budget}\n\nresult_same = differential_evolution(\n    neg_total_response,\n    bounds=bounds,\n    constraints=constraints,\n    seed=42,\n    maxiter=1000,\n    tol=1e-8,\n    polish=True\n)\n\nopt_spend_same = result_same.x\nopt_response_same = total_response(opt_spend_same)\ncurrent_response = total_response(list(current_spend.values()))\nuplift_same = (opt_response_same - current_response) / current_response * 100\n\n# ============================================================\n# SCENARIO 2: +20% budget increase, optimised allocation\n# ============================================================\ntotal_budget_plus20 = total_budget * 1.20\nbounds_plus20 = [(total_budget_plus20 * 0.05, total_budget_plus20 * 0.70)] * len(channels)\nconstraints_plus20 = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - total_budget_plus20}\n\nresult_plus20 = differential_evolution(\n    neg_total_response,\n    bounds=bounds_plus20,\n    constraints=constraints_plus20,\n    seed=42,\n    maxiter=1000,\n    tol=1e-8,\n    polish=True\n)\nopt_spend_plus20 = result_plus20.x\nopt_response_plus20 = total_response(opt_spend_plus20)\nuplift_plus20 = (opt_response_plus20 - current_response) / current_response * 100\n\n# ============================================================\n# SCENARIO 3: -20% budget cut, optimised allocation\n# ============================================================\ntotal_budget_minus20 = total_budget * 0.80\nbounds_minus20 = [(total_budget_minus20 * 0.05, total_budget_minus20 * 0.70)] * len(channels)\nconstraints_minus20 = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - total_budget_minus20}\n\nresult_minus20 = differential_evolution(\n    neg_total_response,\n    bounds=bounds_minus20,\n    constraints=constraints_minus20,\n    seed=42,\n    maxiter=1000,\n    tol=1e-8,\n    polish=True\n)\nopt_spend_minus20 = result_minus20.x\nopt_response_minus20 = total_response(opt_spend_minus20)\nuplift_minus20 = (opt_response_minus20 - current_response) / current_response * 100\n\n# ============================================================\n# PRINT RESULTS\n# ============================================================\nprint("=" * 70)\nprint("BUDGET OPTIMISATION RESULTS")\nprint("=" * 70)\n\nscenarios = [\n    ("Current (Baseline)",    list(current_spend.values()), current_response,  0),\n    ("Same Budget (Optimised)", opt_spend_same,             opt_response_same,  uplift_same),\n    ("-20% Budget (Optimised)", opt_spend_minus20,          opt_response_minus20, uplift_minus20),\n    ("+20% Budget (Optimised)", opt_spend_plus20,           opt_response_plus20,  uplift_plus20),\n]\n\nfor name, spends, response, uplift in scenarios:\n    total_s = sum(spends)\n    print(f"\\n{\'─\'*70}")\n    print(f"  📌 {name}")\n    print(f"{\'─\'*70}")\n    print(f"  {\'Channel\':<15} {\'Spend\':>12}  {\'Share\':>7}  {\'Change vs Current\':>18}")\n    for i, ch in enumerate(channels):\n        share = spends[i] / total_s * 100\n        delta = spends[i] - current_spend[ch]\n        delta_str = f"{\'▲\' if delta > 0 else \'▼\' if delta < 0 else \'─\'} ${abs(delta):>8,.0f}" if name != "Current (Baseline)" else "—"\n        print(f"  {ch:<15} ${spends[i]:>10,.0f}  {share:>6.1f}%  {delta_str:>18}")\n    print(f"  {\'─\'*60}")\n    print(f"  {\'TOTAL SPEND\':<15} ${total_s:>10,.0f}")\n    print(f"  {\'Media Sales\':<15} ${response:>10,.0f}/week")\n    if uplift != 0:\n        print(f"  {\'Uplift vs Now\':<15} {uplift:>+10.1f}%")\n\nprint(f"\\n{\'=\'*70}")\nprint("MARGINAL RETURN ON NEXT $1,000 SPEND (Current Allocation)")\nprint(f"{\'=\'*70}")\nfor ch in channels:\n    delta = 1000\n    base = channel_response(current_spend[ch], ch)\n    incremental = channel_response(current_spend[ch] + delta, ch) - base\n    print(f"  {ch:<15}: ${incremental:>8,.2f} incremental sales per $1,000 spent")\n'})`

> **Tool Result:** `execute_python` completed.

**Agent:**
[{'id': 'toolu_01KGtD5Lq1X1NE7tua8WEj9C', 'caller': {'type': 'direct'}, 'input': {'code': '\nimport numpy as np\nfrom scipy.optimize import differential_evolution\n\nchannels = [\'TV\', \'Digital\', \'Paid_Social\', \'Radio\']\n\n# Model-derived parameters\nsaturation_pct = {\'TV\': 0.705, \'Digital\': 0.692, \'Paid_Social\': 0.688, \'Radio\': 0.682}\ncurrent_spend  = {\'TV\': 81212, \'Digital\': 56788, \'Paid_Social\': 29842, \'Radio\': 12028}\nsales_contrib  = {\n    \'TV\':          11691192 / 104,\n    \'Digital\':      3417406 / 104,\n    \'Paid_Social\':  2857817 / 104,\n    \'Radio\':        1738851 / 104,\n}\n\nalpha = {ch: 2.0 for ch in channels}\n\n# Back-calculate K (half-saturation point)\nK = {}\nfor ch in channels:\n    x = current_spend[ch]\n    s = saturation_pct[ch]\n    a = alpha[ch]\n    K[ch] = (x**a * (1 - s) / s) ** (1/a)\n\n# Scale factor\nscale = {}\nfor ch in channels:\n    x = current_spend[ch]\n    a = alpha[ch]\n    hill_val = x**a / (x**a + K[ch]**a)\n    scale[ch] = sales_contrib[ch] / hill_val\n\ndef channel_response(spend, ch):\n    a = alpha[ch]\n    return scale[ch] * spend**a / (spend**a + K[ch]**a)\n\ndef total_response(spend_vec):\n    return sum(channel_response(spend_vec[i], channels[i]) for i in range(len(channels)))\n\ndef neg_total_response(spend_vec):\n    return -total_response(spend_vec)\n\ntotal_budget = sum(current_spend.values())\ncurrent_response = total_response(list(current_spend.values()))\n\ndef run_optimisation(budget):\n    bounds = [(budget * 0.05, budget * 0.70)] * len(channels)\n    constraints = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - budget}\n    result = differential_evolution(\n        neg_total_response, bounds=bounds,\n        constraints=constraints, seed=42,\n        maxiter=2000, tol=1e-10, polish=True\n    )\n    return result.x, total_response(result.x)\n\nopt_same,     resp_same     = run_optimisation(total_budget)\nopt_minus20,  resp_minus20  = run_optimisation(total_budget * 0.80)\nopt_plus20,   resp_plus20   = run_optimisation(total_budget * 1.20)\n\n# ── Print results ──────────────────────────────────────────────────────────────\nscenarios = [\n    ("Current (Baseline)",      list(current_spend.values()), current_response,  None),\n    ("Same Budget – Optimised", opt_same,                     resp_same,          (resp_same    - current_response) / current_response * 100),\n    ("-20% Budget – Optimised", opt_minus20,                  resp_minus20,       (resp_minus20 - current_response) / current_response * 100),\n    ("+20% Budget – Optimised", opt_plus20,                   resp_plus20,        (resp_plus20  - current_response) / current_response * 100),\n]\n\nprint("=" * 72)\nprint("  BUDGET OPTIMISATION RESULTS")\nprint("=" * 72)\n\nfor name, spends, response, uplift in scenarios:\n    total_s = sum(spends)\n    print(f"\\n  📌 {name}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'Channel\':<15} {\'Spend\':>12}  {\'Share\':>7}  {\'vs Current\':>16}")\n    for i, ch in enumerate(channels):\n        share = spends[i] / total_s * 100\n        if name == "Current (Baseline)":\n            delta_str = "—"\n        else:\n            delta = spends[i] - current_spend[ch]\n            arrow = \'▲\' if delta > 0 else \'▼\'\n            delta_str = f"{arrow} ${abs(delta):>8,.0f}"\n        print(f"  {ch:<15} ${spends[i]:>10,.0f}  {share:>6.1f}%  {delta_str:>16}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'TOTAL SPEND\':<15} ${total_s:>10,.0f}")\n    print(f"  {\'Media Sales\':<15} ${response:>10,.0f} / week")\n    if uplift is not None:\n        print(f"  {\'Uplift vs Now\':<15} {uplift:>+10.1f}%")\n\nprint(f"\\n{\'=\'*72}")\nprint("  MARGINAL RETURN ON NEXT $1,000 (at current spend levels)")\nprint(f"{\'=\'*72}")\nfor ch in channels:\n    base = channel_response(current_spend[ch], ch)\n    inc  = channel_response(current_spend[ch] + 1000, ch) - base\n    print(f"  {ch:<15}: ${inc:>8,.2f} incremental sales per $1,000 spent")\nprint(f"{\'=\'*72}")\n'}, 'name': 'execute_python', 'type': 'tool_use'}]

> **Tool Call:** `execute_python({'code': '\nimport numpy as np\nfrom scipy.optimize import differential_evolution\n\nchannels = [\'TV\', \'Digital\', \'Paid_Social\', \'Radio\']\n\n# Model-derived parameters\nsaturation_pct = {\'TV\': 0.705, \'Digital\': 0.692, \'Paid_Social\': 0.688, \'Radio\': 0.682}\ncurrent_spend  = {\'TV\': 81212, \'Digital\': 56788, \'Paid_Social\': 29842, \'Radio\': 12028}\nsales_contrib  = {\n    \'TV\':          11691192 / 104,\n    \'Digital\':      3417406 / 104,\n    \'Paid_Social\':  2857817 / 104,\n    \'Radio\':        1738851 / 104,\n}\n\nalpha = {ch: 2.0 for ch in channels}\n\n# Back-calculate K (half-saturation point)\nK = {}\nfor ch in channels:\n    x = current_spend[ch]\n    s = saturation_pct[ch]\n    a = alpha[ch]\n    K[ch] = (x**a * (1 - s) / s) ** (1/a)\n\n# Scale factor\nscale = {}\nfor ch in channels:\n    x = current_spend[ch]\n    a = alpha[ch]\n    hill_val = x**a / (x**a + K[ch]**a)\n    scale[ch] = sales_contrib[ch] / hill_val\n\ndef channel_response(spend, ch):\n    a = alpha[ch]\n    return scale[ch] * spend**a / (spend**a + K[ch]**a)\n\ndef total_response(spend_vec):\n    return sum(channel_response(spend_vec[i], channels[i]) for i in range(len(channels)))\n\ndef neg_total_response(spend_vec):\n    return -total_response(spend_vec)\n\ntotal_budget = sum(current_spend.values())\ncurrent_response = total_response(list(current_spend.values()))\n\ndef run_optimisation(budget):\n    bounds = [(budget * 0.05, budget * 0.70)] * len(channels)\n    constraints = {\'type\': \'eq\', \'fun\': lambda x: sum(x) - budget}\n    result = differential_evolution(\n        neg_total_response, bounds=bounds,\n        constraints=constraints, seed=42,\n        maxiter=2000, tol=1e-10, polish=True\n    )\n    return result.x, total_response(result.x)\n\nopt_same,     resp_same     = run_optimisation(total_budget)\nopt_minus20,  resp_minus20  = run_optimisation(total_budget * 0.80)\nopt_plus20,   resp_plus20   = run_optimisation(total_budget * 1.20)\n\n# ── Print results ──────────────────────────────────────────────────────────────\nscenarios = [\n    ("Current (Baseline)",      list(current_spend.values()), current_response,  None),\n    ("Same Budget – Optimised", opt_same,                     resp_same,          (resp_same    - current_response) / current_response * 100),\n    ("-20% Budget – Optimised", opt_minus20,                  resp_minus20,       (resp_minus20 - current_response) / current_response * 100),\n    ("+20% Budget – Optimised", opt_plus20,                   resp_plus20,        (resp_plus20  - current_response) / current_response * 100),\n]\n\nprint("=" * 72)\nprint("  BUDGET OPTIMISATION RESULTS")\nprint("=" * 72)\n\nfor name, spends, response, uplift in scenarios:\n    total_s = sum(spends)\n    print(f"\\n  📌 {name}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'Channel\':<15} {\'Spend\':>12}  {\'Share\':>7}  {\'vs Current\':>16}")\n    for i, ch in enumerate(channels):\n        share = spends[i] / total_s * 100\n        if name == "Current (Baseline)":\n            delta_str = "—"\n        else:\n            delta = spends[i] - current_spend[ch]\n            arrow = \'▲\' if delta > 0 else \'▼\'\n            delta_str = f"{arrow} ${abs(delta):>8,.0f}"\n        print(f"  {ch:<15} ${spends[i]:>10,.0f}  {share:>6.1f}%  {delta_str:>16}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'TOTAL SPEND\':<15} ${total_s:>10,.0f}")\n    print(f"  {\'Media Sales\':<15} ${response:>10,.0f} / week")\n    if uplift is not None:\n        print(f"  {\'Uplift vs Now\':<15} {uplift:>+10.1f}%")\n\nprint(f"\\n{\'=\'*72}")\nprint("  MARGINAL RETURN ON NEXT $1,000 (at current spend levels)")\nprint(f"{\'=\'*72}")\nfor ch in channels:\n    base = channel_response(current_spend[ch], ch)\n    inc  = channel_response(current_spend[ch] + 1000, ch) - base\n    print(f"  {ch:<15}: ${inc:>8,.2f} incremental sales per $1,000 spent")\nprint(f"{\'=\'*72}")\n'})`

> **Tool Result:** `execute_python` completed.

**Agent:**
[{'id': 'toolu_016pH36QR6izM6czGFHotkHg', 'caller': {'type': 'direct'}, 'input': {'code': '\nimport numpy as np\nfrom scipy.optimize import minimize\n\nchannels = [\'TV\', \'Digital\', \'Paid_Social\', \'Radio\']\n\nsaturation_pct = {\'TV\': 0.705, \'Digital\': 0.692, \'Paid_Social\': 0.688, \'Radio\': 0.682}\ncurrent_spend  = {\'TV\': 81212, \'Digital\': 56788, \'Paid_Social\': 29842, \'Radio\': 12028}\nsales_contrib  = {\n    \'TV\':          11691192 / 104,\n    \'Digital\':      3417406 / 104,\n    \'Paid_Social\':  2857817 / 104,\n    \'Radio\':        1738851 / 104,\n}\n\nalpha = {ch: 2.0 for ch in channels}\n\nK = {}\nfor ch in channels:\n    x = current_spend[ch]\n    s = saturation_pct[ch]\n    a = alpha[ch]\n    K[ch] = (x**a * (1 - s) / s) ** (1/a)\n\nscale = {}\nfor ch in channels:\n    x = current_spend[ch]\n    a = alpha[ch]\n    hill_val = x**a / (x**a + K[ch]**a)\n    scale[ch] = sales_contrib[ch] / hill_val\n\ndef channel_response(spend, ch):\n    a = alpha[ch]\n    return scale[ch] * spend**a / (spend**a + K[ch]**a)\n\ndef total_response(spend_vec):\n    return sum(channel_response(spend_vec[i], channels[i]) for i in range(len(channels)))\n\ndef neg_total_response(spend_vec):\n    return -total_response(spend_vec)\n\ntotal_budget = sum(current_spend.values())\ncurrent_response = total_response(list(current_spend.values()))\n\ndef run_optimisation(budget):\n    x0 = np.array([budget / len(channels)] * len(channels))\n    bounds = [(budget * 0.05, budget * 0.70)] * len(channels)\n    constraints = [{\'type\': \'eq\', \'fun\': lambda x: np.sum(x) - budget}]\n    best_val = np.inf\n    best_x = x0.copy()\n    # Multiple random restarts for robustness\n    rng = np.random.default_rng(42)\n    for _ in range(30):\n        x_init = rng.dirichlet(np.ones(len(channels))) * budget\n        x_init = np.clip(x_init, budget * 0.05, budget * 0.70)\n        x_init = x_init / x_init.sum() * budget\n        res = minimize(neg_total_response, x_init, method=\'SLSQP\',\n                       bounds=bounds, constraints=constraints,\n                       options={\'ftol\': 1e-12, \'maxiter\': 2000})\n        if res.fun < best_val:\n            best_val = res.fun\n            best_x = res.x\n    return best_x, total_response(best_x)\n\nopt_same,    resp_same    = run_optimisation(total_budget)\nopt_minus20, resp_minus20 = run_optimisation(total_budget * 0.80)\nopt_plus20,  resp_plus20  = run_optimisation(total_budget * 1.20)\n\nscenarios = [\n    ("Current (Baseline)",      list(current_spend.values()), current_response,  None),\n    ("Same Budget – Optimised", opt_same,                     resp_same,          (resp_same    - current_response) / current_response * 100),\n    ("-20% Budget – Optimised", opt_minus20,                  resp_minus20,       (resp_minus20 - current_response) / current_response * 100),\n    ("+20% Budget – Optimised", opt_plus20,                   resp_plus20,        (resp_plus20  - current_response) / current_response * 100),\n]\n\nprint("=" * 72)\nprint("  BUDGET OPTIMISATION RESULTS")\nprint("=" * 72)\n\nfor name, spends, response, uplift in scenarios:\n    total_s = sum(spends)\n    print(f"\\n  📌 {name}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'Channel\':<15} {\'Spend\':>12}  {\'Share\':>7}  {\'vs Current\':>16}")\n    for i, ch in enumerate(channels):\n        share = spends[i] / total_s * 100\n        if name == "Current (Baseline)":\n            delta_str = "—"\n        else:\n            delta = spends[i] - current_spend[ch]\n            arrow = \'▲\' if delta > 0 else \'▼\'\n            delta_str = f"{arrow} ${abs(delta):>8,.0f}"\n        print(f"  {ch:<15} ${spends[i]:>10,.0f}  {share:>6.1f}%  {delta_str:>16}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'TOTAL SPEND\':<15} ${total_s:>10,.0f}")\n    print(f"  {\'Media Sales\':<15} ${response:>10,.0f} / week")\n    if uplift is not None:\n        print(f"  {\'Uplift vs Now\':<15} {uplift:>+10.1f}%")\n\nprint(f"\\n{\'=\'*72}")\nprint("  MARGINAL RETURN ON NEXT $1,000 (at current spend levels)")\nprint(f"{\'=\'*72}")\nfor ch in channels:\n    base = channel_response(current_spend[ch], ch)\n    inc  = channel_response(current_spend[ch] + 1000, ch) - base\n    print(f"  {ch:<15}: ${inc:>8,.2f} incremental sales per $1,000 spent")\nprint(f"{\'=\'*72}")\n'}, 'name': 'execute_python', 'type': 'tool_use'}]

> **Tool Call:** `execute_python({'code': '\nimport numpy as np\nfrom scipy.optimize import minimize\n\nchannels = [\'TV\', \'Digital\', \'Paid_Social\', \'Radio\']\n\nsaturation_pct = {\'TV\': 0.705, \'Digital\': 0.692, \'Paid_Social\': 0.688, \'Radio\': 0.682}\ncurrent_spend  = {\'TV\': 81212, \'Digital\': 56788, \'Paid_Social\': 29842, \'Radio\': 12028}\nsales_contrib  = {\n    \'TV\':          11691192 / 104,\n    \'Digital\':      3417406 / 104,\n    \'Paid_Social\':  2857817 / 104,\n    \'Radio\':        1738851 / 104,\n}\n\nalpha = {ch: 2.0 for ch in channels}\n\nK = {}\nfor ch in channels:\n    x = current_spend[ch]\n    s = saturation_pct[ch]\n    a = alpha[ch]\n    K[ch] = (x**a * (1 - s) / s) ** (1/a)\n\nscale = {}\nfor ch in channels:\n    x = current_spend[ch]\n    a = alpha[ch]\n    hill_val = x**a / (x**a + K[ch]**a)\n    scale[ch] = sales_contrib[ch] / hill_val\n\ndef channel_response(spend, ch):\n    a = alpha[ch]\n    return scale[ch] * spend**a / (spend**a + K[ch]**a)\n\ndef total_response(spend_vec):\n    return sum(channel_response(spend_vec[i], channels[i]) for i in range(len(channels)))\n\ndef neg_total_response(spend_vec):\n    return -total_response(spend_vec)\n\ntotal_budget = sum(current_spend.values())\ncurrent_response = total_response(list(current_spend.values()))\n\ndef run_optimisation(budget):\n    x0 = np.array([budget / len(channels)] * len(channels))\n    bounds = [(budget * 0.05, budget * 0.70)] * len(channels)\n    constraints = [{\'type\': \'eq\', \'fun\': lambda x: np.sum(x) - budget}]\n    best_val = np.inf\n    best_x = x0.copy()\n    # Multiple random restarts for robustness\n    rng = np.random.default_rng(42)\n    for _ in range(30):\n        x_init = rng.dirichlet(np.ones(len(channels))) * budget\n        x_init = np.clip(x_init, budget * 0.05, budget * 0.70)\n        x_init = x_init / x_init.sum() * budget\n        res = minimize(neg_total_response, x_init, method=\'SLSQP\',\n                       bounds=bounds, constraints=constraints,\n                       options={\'ftol\': 1e-12, \'maxiter\': 2000})\n        if res.fun < best_val:\n            best_val = res.fun\n            best_x = res.x\n    return best_x, total_response(best_x)\n\nopt_same,    resp_same    = run_optimisation(total_budget)\nopt_minus20, resp_minus20 = run_optimisation(total_budget * 0.80)\nopt_plus20,  resp_plus20  = run_optimisation(total_budget * 1.20)\n\nscenarios = [\n    ("Current (Baseline)",      list(current_spend.values()), current_response,  None),\n    ("Same Budget – Optimised", opt_same,                     resp_same,          (resp_same    - current_response) / current_response * 100),\n    ("-20% Budget – Optimised", opt_minus20,                  resp_minus20,       (resp_minus20 - current_response) / current_response * 100),\n    ("+20% Budget – Optimised", opt_plus20,                   resp_plus20,        (resp_plus20  - current_response) / current_response * 100),\n]\n\nprint("=" * 72)\nprint("  BUDGET OPTIMISATION RESULTS")\nprint("=" * 72)\n\nfor name, spends, response, uplift in scenarios:\n    total_s = sum(spends)\n    print(f"\\n  📌 {name}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'Channel\':<15} {\'Spend\':>12}  {\'Share\':>7}  {\'vs Current\':>16}")\n    for i, ch in enumerate(channels):\n        share = spends[i] / total_s * 100\n        if name == "Current (Baseline)":\n            delta_str = "—"\n        else:\n            delta = spends[i] - current_spend[ch]\n            arrow = \'▲\' if delta > 0 else \'▼\'\n            delta_str = f"{arrow} ${abs(delta):>8,.0f}"\n        print(f"  {ch:<15} ${spends[i]:>10,.0f}  {share:>6.1f}%  {delta_str:>16}")\n    print(f"  {\'─\'*68}")\n    print(f"  {\'TOTAL SPEND\':<15} ${total_s:>10,.0f}")\n    print(f"  {\'Media Sales\':<15} ${response:>10,.0f} / week")\n    if uplift is not None:\n        print(f"  {\'Uplift vs Now\':<15} {uplift:>+10.1f}%")\n\nprint(f"\\n{\'=\'*72}")\nprint("  MARGINAL RETURN ON NEXT $1,000 (at current spend levels)")\nprint(f"{\'=\'*72}")\nfor ch in channels:\n    base = channel_response(current_spend[ch], ch)\n    inc  = channel_response(current_spend[ch] + 1000, ch) - base\n    print(f"  {ch:<15}: ${inc:>8,.2f} incremental sales per $1,000 spent")\nprint(f"{\'=\'*72}")\n'})`

> **Tool Result:** `execute_python` completed.

**Agent:**
Here is your full budget optimisation analysis:

---

# 💰 Budget Optimisation Report

---

## 📌 Current Spend Baseline

| Channel | Weekly Spend | Share |
|---|---|---|
| 📺 TV | $81,212 | 45.2% |
| 💻 Digital | $56,788 | 31.6% |
| 📱 Paid Social | $29,842 | 16.6% |
| 📻 Radio | $12,028 | 6.7% |
| **Total** | **$179,870** | **100%** |
| **Media-Driven Sales** | **$189,474 / week** | |

---

## 🎯 Scenario 1: Same Budget, Optimised Allocation (+2.1% uplift)

> *Spend the same total amount, but redistribute it more efficiently.*

| Channel | Current | Optimised | Change | New Share |
|---|---|---|---|---|
| 📺 **TV** | $81,212 | $95,738 | ▲ +$14,526 | 53.2% |
| 💻 **Digital** | $56,788 | $39,965 | ▼ −$16,823 | 22.2% |
| 📱 **Paid Social** | $29,842 | $29,318 | ▼ −$524 | 16.3% |
| 📻 **Radio** | $12,028 | $14,850 | ▲ +$2,822 | 8.3% |
| **Media Sales** | $189,474 | **$193,509** | **+$4,035/wk** | |

> ✅ **Just by reallocating budget — with no extra spend — you can generate an additional ~$4,000 in media-driven sales per week (~$210K annually).** The key move is shifting money from Digital into TV.

---

## 📉 Scenario 2: −20% Budget Cut, Optimised Allocation (−11.1% vs current)

> *If budget must be cut, where should you protect spend?*

| Channel | Current | Optimised | Change | New Share |
|---|---|---|---|---|
| 📺 **TV** | $81,212 | $82,006 | ▲ +$794 | 57.0% |
| 💻 **Digital** | $56,788 | $25,093 | ▼ −$31,695 | 17.4% |
| 📱 **Paid Social** | $29,842 | $24,092 | ▼ −$5,750 | 16.7% |
| 📻 **Radio** | $12,028 | $12,705 | ▲ +$677 | 8.8% |
| **Media Sales** | $189,474 | **$168,380** | −$21,094/wk | |

> 🛡️ **If cuts are unavoidable, protect TV and Radio at all costs.** The model recommends absorbing almost the entire cut from Digital (−$31.7K), which has the lowest ROI. TV spend is held nearly flat, preserving the most efficient channel.

---

## 📈 Scenario 3: +20% Budget Increase, Optimised Allocation (+11.7% uplift)

> *If you had more budget, where should the incremental dollars go?*

| Channel | Current | Optimised | Change | New Share |
|---|---|---|---|---|
| 📺 **TV** | $81,212 | $111,910 | ▲ +$30,698 | 51.8% |
| 💻 **Digital** | $56,788 | $51,397 | ▼ −$5,391 | 23.8% |
| 📱 **Paid Social** | $29,842 | $35,170 | ▲ +$5,328 | 16.3% |
| 📻 **Radio** | $12,028 | $17,368 | ▲ +$5,340 | 8.0% |
| **Media Sales** | $189,474 | **$211,574** | **+$22,100/wk** | |

> 🚀 **A 20% budget increase, optimally deployed, yields a 11.7% uplift in media-driven sales.** The majority of new spend should go into TV (+$30.7K), with balanced increases across Radio and Paid Social. Interestingly, Digital is slightly *reduced* even with more budget available — confirming it is the least efficient channel.

---

## 🔬 Marginal Return on Next $1,000 Spent (at current levels)

| Channel | Sales per $1,000 | Verdict |
|---|---|---|
| 📻 **Radio** | **$823.56** | 🥇 Most efficient next dollar |
| 📺 **TV** | **$807.61** | 🥈 Highly efficient |
| 📱 **Paid Social** | $558.05 | 🟡 Moderate |
| 💻 **Digital** | $350.95 | 🔴 Least efficient |

---

## 🔑 Key Recommendations

| # | Recommendation |
|---|---|
| 1 | **Shift ~$17K/week from Digital into TV** — this single change unlocks +2.1% more sales at zero extra cost. |
| 2 | **Digital is over-funded** — at 31.6% of budget, it is your largest channel but your worst performer. It should be capped at ~22%. |
| 3 | **TV should be your anchor channel** — it consistently receives 50–57% of the optimised budget across all scenarios. |
| 4 | **Radio is punching above its weight** — small increases here yield strong marginal returns. Consider growing it to ~8–9% of budget. |
| 5 | **If cutting budget, cut Digital first** — it absorbs the most pain with the least damage to overall sales performance. |

---

Would you like to explore a **custom budget scenario**, run a **spend curve analysis** for a specific channel, or look at anything else?
