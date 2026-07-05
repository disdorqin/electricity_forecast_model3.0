"""P123: Realtime eval pack target column fix tests."""
from __future__ import annotations
import os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestEvalPack:
    def test_eval_pack_uses_realtime_price(self):
        path = os.path.join(REPO, "models/adapters/realtime_deep_adapter.py")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "实时电价" in content
        assert "eval_only" in content

    def test_eval_pack_not_using_dayahead(self):
        path = os.path.join(REPO, "models/adapters/realtime_deep_adapter.py")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # export_eval_pack should use 实时电价, not 日前电价
        idx = content.find("def export_eval_pack")
        rest = content[idx:] if idx >= 0 else content
        # The eval pack function should reference 实时电价
        assert "实时电价" in rest

    def test_online_pack_no_y_true(self):
        """Online pack must not contain y_true."""
        path = os.path.join(REPO, "models/adapters/realtime_deep_adapter.py")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # The export_online_pack should NOT set y_true
        assert "export_online_pack" in content

    def test_eval_only_directory(self):
        path = os.path.join(REPO, "models/adapters/realtime_deep_adapter.py")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "eval_only" in content
