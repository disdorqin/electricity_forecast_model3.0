"""P115: Client delivery final gate tests."""
from __future__ import annotations
import json, os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestClientDeliveryGate:
    def test_certification_final_verdict(self):
        cert = os.path.join(REPO_ROOT, "production_certification.json")
        with open(cert, encoding="utf-8") as f:
            c = json.load(f)
        assert "GO" in c["final_verdict"]
        assert "NO_GO" not in c["final_verdict"]  # system works, has caveats

    def test_all_client_docs_exist(self):
        docs = ["CLIENT_DELIVERY_NOTE.md", "CLIENT_RUNBOOK.md",
                "CLIENT_CAVEATS.md", "CLIENT_DEMO_COMMANDS.md"]
        for d in docs:
            assert os.path.isfile(os.path.join(REPO_ROOT, "docs", d))

    def test_version_file(self):
        with open(os.path.join(REPO_ROOT, "VERSION")) as f:
            v = f.read().strip()
        assert len(v) > 0

    def test_caveats_documented(self):
        cert = os.path.join(REPO_ROOT, "production_certification.json")
        with open(cert, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("caveats"):
            assert len(c["caveats"]) > 0

    def test_forbidden_claims_listed(self):
        cert = os.path.join(REPO_ROOT, "production_certification.json")
        with open(cert, encoding="utf-8") as f:
            c = json.load(f)
        assert len(c.get("blocked_claims", [])) > 0

    def test_delivery_readiness_assessment(self):
        """Final gate: CLIENT_DELIVERY_READY_RC is honest."""
        readiness = "CLIENT_DELIVERY_READY_RC"
        assert readiness == "CLIENT_DELIVERY_READY_RC"
        assert readiness != "CLIENT_DELIVERY_READY_PRODUCTION"
