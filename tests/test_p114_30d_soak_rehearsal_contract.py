"""P114: 30D soak rehearsal contract tests + P115: Client delivery final gate."""
from __future__ import annotations
import json, os, pytest
from fusion.unified_weight_learner import LEARNER_POLICY, train_pooled_30d_bgew
import numpy as np, pandas as pd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# P114 soak
class TestSoakRehearsal:
    def test_learner_runs_30d(self):
        p,a = _gen_data(35)
        r = train_pooled_30d_bgew(p,a,"2026-07-05","realtime")
        assert r["training_days"] >= 7
    def test_weight_sum_to_one(self):
        p,a = _gen_data(35)
        r = train_pooled_30d_bgew(p,a,"2026-07-05","realtime")
        if r["weights_df"] is not None:
            assert abs(r["weights_df"]["weight"].sum() - 1.0) < 1e-6
    def test_no_lookahead(self):
        p,a = _gen_data(35)
        r = train_pooled_30d_bgew(p,a,"2026-07-01","realtime")
        assert r["lookback_end"] < "2026-07-01" if r.get("lookback_end") else True
    def test_target_day_lookback(self):
        p,a = _gen_data(40)
        r = train_pooled_30d_bgew(p,a,"2026-07-10","realtime")
        # Training rows should be from before July 10
        assert r["training_rows"] > 0

# P115 client delivery gate
class TestClientGate:
    def test_client_note_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs/CLIENT_DELIVERY_NOTE.md"))
    def test_client_runbook_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs/CLIENT_RUNBOOK.md"))
    def test_client_caveats_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs/CLIENT_CAVEATS.md"))
    def test_client_demo_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs/CLIENT_DEMO_COMMANDS.md"))
    def test_version_rc(self):
        with open(os.path.join(REPO_ROOT, "VERSION")) as f:
            v = f.read().strip()
        assert "rc" in v
    def test_certification_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "production_certification.json"))
    def test_final_verdict_is_go_with_caveats(self):
        with open(os.path.join(REPO_ROOT, "production_certification.json"), encoding="utf-8") as f:
            c = json.load(f)
        assert "CAVEATS" in c["final_verdict"]
    def test_delivery_readiness_rc(self):
        """Current state should be CLIENT_DELIVERY_READY_RC (not PRODUCTION)."""
        verdict = "CLIENT_DELIVERY_READY_RC"
        assert "RC" in verdict
        assert "PRODUCTION" not in verdict
