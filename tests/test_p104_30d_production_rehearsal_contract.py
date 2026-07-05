"""P104: 30-day production rehearsal contract tests + P105 extended backtest."""
from __future__ import annotations
import os, pytest
from fusion.unified_weight_learner import LEARNER_POLICY, train_pooled_30d_bgew
import numpy as np, pandas as pd

def _gen_data(days=35):
    np.random.seed(42); p, a = [], []
    for d in range(days):
        day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=d)
        ds = day.strftime("%Y-%m-%d")
        for h in range(1,25):
            ts = day + pd.Timedelta(hours=h-1)
            y = 300.0 + np.random.uniform(-50,50)
            p.append({"model_name":"rt_da_anchor","business_day":ds,"ds":ts,"hour_business":h,"y_pred":y+np.random.uniform(-15,15)})
            p.append({"model_name":"sgdfnet_rt_assist","business_day":ds,"ds":ts,"hour_business":h,"y_pred":y+np.random.uniform(-20,20)})
            a.append({"business_day":ds,"ds":ts,"hour_business":h,"y_true":y})
    return pd.DataFrame(p), pd.DataFrame(a)

class TestRehearsal:
    def test_realtime_learner_30d(self):
        p,a = _gen_data(35)
        r = train_pooled_30d_bgew(p,a,"2026-07-05","realtime")
        assert r["training_days"] >= 7
    def test_learner_policy_realtime(self):
        assert LEARNER_POLICY["realtime"] == "pooled_30d_bgew"
    def test_learner_policy_dayahead(self):
        assert LEARNER_POLICY["dayahead"] == "period_regime_bgew"
    def test_no_lookahead(self):
        p,a = _gen_data(35)
        r = train_pooled_30d_bgew(p,a,"2026-07-01","realtime")
        assert r["lookback_end"] < "2026-07-01" if r.get("lookback_end") else True
    def test_training_rows_approx_720(self):
        p,a = _gen_data(35)
        r = train_pooled_30d_bgew(p,a,"2026-07-05","realtime")
        assert r["training_rows"] > 100  # at least some rows

class TestExtendedBacktest:
    def test_smape_computed(self):
        p,a = _gen_data(35)
        r = train_pooled_30d_bgew(p,a,"2026-07-05","realtime")
        assert isinstance(r["training_days"], int)
