"""
models/realtime_state.py — P91: Realtime design state constants.

Replaces the old fallback-centric naming (REALTIME_DA_ANCHOR_FALLBACK,
REALTIME_DEEP_READY_FAST_DEV, FAST_DEV_ONLY) with DA-Safe Baseline
centric naming.

State machine rules:

    REALTIME_DA_SAFE_BASELINE
        rt_da_anchor is the official primary prediction, NOT a fallback.
        Always available as long as day-ahead predictions exist.

    REALTIME_ASSIST_SGDFNET_AVAILABLE
        SGDFNet (from 2.0 exp repo) is available as an optional assist /
        sidecar candidate for realtime fusion.

    REALTIME_ASSIST_DISABLED
        SGDFNet is NOT available. This is NOT a NO_GO state. The system
        runs on rt_da_anchor alone with full delivery capability.

    REALTIME_HYBRID_READY
        Both rt_da_anchor AND SGDFNet assist are available and have
        passed ledger/learner validation. Two-candidate realtime fusion
        is active.

Final output status rules:

    If rt_da_anchor is available AND final_output realtime_price
    has no NaN AND safety checks pass:
        realtime core = READY

    If SGDFNet is unavailable:
        status = REALTIME_READY_DA_SAFE_ONLY

    If SGDFNet is available AND passes ledger + learner:
        status = REALTIME_HYBRID_READY

    If realtime_price is NaN:
        NO_GO
"""

from __future__ import annotations

# ── New DA-Safe states (replacing old fallback naming) ────────────────
REALTIME_DA_SAFE_BASELINE = "REALTIME_DA_SAFE_BASELINE"
REALTIME_ASSIST_SGDFNET_AVAILABLE = "REALTIME_ASSIST_SGDFNET_AVAILABLE"
REALTIME_ASSIST_DISABLED = "REALTIME_ASSIST_DISABLED"
REALTIME_HYBRID_READY = "REALTIME_HYBRID_READY"

# ── Final output status aliases ───────────────────────────────────────
REALTIME_READY_DA_SAFE_ONLY = "REALTIME_READY_DA_SAFE_ONLY"
REALTIME_HYBRID_READY_FINAL = "REALTIME_HYBRID_READY"
REALTIME_NO_GO = "REALTIME_NO_GO"

# ── SGDFNet assist status codes ──────────────────────────────────────
SGDFNET_ASSIST_READY = "SGDFNET_ASSIST_READY"
SGDFNET_ASSIST_CODE_ONLY = "SGDFNET_ASSIST_CODE_ONLY"
SGDFNET_ASSIST_BLOCKED = "SGDFNET_ASSIST_BLOCKED"

# ── Learner policy keys ──────────────────────────────────────────────
LEARNER_POLICY_DAYAHEAD = "period_regime_bgew"
LEARNER_POLICY_REALTIME = "pooled_30d_bgew"
LEARNER_POLICY_REALTIME_SINGLE = "realtime_single_model_safe_baseline"

# ── Learner status constants ─────────────────────────────────────────
REALTIME_LEARNER_POOLED_TRAINED = "REALTIME_LEARNER_POOLED_TRAINED"
REALTIME_LEARNER_SINGLE_MODEL = "REALTIME_LEARNER_SINGLE_MODEL"
REALTIME_LEARNER_BLOCKED = "REALTIME_LEARNER_BLOCKED"

# ── Reason codes ─────────────────────────────────────────────────────
SGDFNET_ASSIST_DISABLED = "SGDFNET_ASSIST_DISABLED"
SGDFNET_ASSIST_ACTIVE = "SGDFNET_ASSIST_ACTIVE"
DA_SAFE_BASELINE_ACTIVE = "DA_SAFE_BASELINE_ACTIVE"
REALTIME_HYBRID_FUSION_ACTIVE = "REALTIME_HYBRID_FUSION_ACTIVE"
REALTIME_DA_ANCHOR_NAN = "REALTIME_DA_ANCHOR_NAN"
SGDFNET_WEIGHT_SUPPRESSED = "SGDFNET_WEIGHT_SUPPRESSED"
HARD_REJECT_BAD_ASSIST = "HARD_REJECT_BAD_ASSIST"
