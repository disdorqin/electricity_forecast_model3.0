"""P99: Realtime two-model 30D backtest tests."""
from __future__ import annotations
import json, os, tempfile
import numpy as np, pandas as pd, pytest
from fusion.unified_weight_learner import train_pooled_30d_bgew, LEARNER_POLICY
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _gen_data(days=35, target="2026-07-03", seed=42):
    np.random.seed(seed); preds, acts = [], []
    for d in range(days):
        day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=d)
        ds = day.strftime("%Y-%m-%d")
        for h in range(1,25):
            ts = day + pd.Timedelta(hours=h-1)
            y = 300.0 + np.random.uniform(-50,50)
            preds.append({"model_name":"rt_da_anchor","business_day":ds,"ds":ts,"hour_business":h,"y_pred":y+np.random.uniform(-15,15)})
            preds.append({"model_name":"sgdfnet_rt_assist","business_day":ds,"ds":ts,"hour_business":h,"y_pred":y+np.random.uniform(-20,20)})
            acts.append({"business_day":ds,"ds":ts,"hour_business":h,"y_true":y})
    return pd.DataFrame(preds), pd.DataFrame(acts)

class TestPooledLearnerBacktest:
    def test_learner_runs(self):
        p,a = _gen_data()
        r = train_pooled_30d_bgew(p,a,"2026-07-03","realtime")
        assert r["status"] in ("REALTIME_LEARNER_POOLED_TRAINED","REALTIME_LEARNER_SINGLE_MODEL")
    def test_weights_sum_to_one(self):
        p,a = _gen_data()
        r = train_pooled_30d_bgew(p,a,"2026-07-03","realtime")
        if r["weights_df"] is not None:
            assert abs(r["weights_df"]["weight"].sum() - 1.0) < 1e-6
    def test_no_lookahead(self):
        p,a = _gen_data()
        r = train_pooled_30d_bgew(p,a,"2026-07-01","realtime")
        assert r["training_days"] < 35
    def test_single_model(self):
        p,a = _gen_data()
        p = p[p["model_name"]=="rt_da_anchor"]
        r = train_pooled_30d_bgew(p,a,"2026-07-03","realtime")
        assert r["status"] == "REALTIME_LEARNER_SINGLE_MODEL"
    def test_learner_policy(self):
        assert LEARNER_POLICY["realtime"] == "pooled_30d_bgew"
    def test_period_all(self):
        p,a = _gen_data()
        r = train_pooled_30d_bgew(p,a,"2026-07-03","realtime")
        if r["weights_df"] is not None:
            assert (r["weights_df"]["period"] == "all").all()
    def test_regime_all(self):
        p,a = _gen_data()
        r = train_pooled_30d_bgew(p,a,"2026-07-03","realtime")
        if r["weights_df"] is not None:
            assert (r["weights_df"]["regime"] == "all").all()
    def test_training_rows_count(self):
        p,a = _gen_data()
        r = train_pooled_30d_bgew(p,a,"2026-07-03","realtime")
        assert r["training_rows"] > 0
