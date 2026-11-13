"""#172: without the ml-despike extra, building the stub detector skips the
test; merely requesting the fixture does not."""
import sys

import pytest


def test_fixture_request_does_not_skip_and_build_skips(stub_detector_factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # extra not installed
    with pytest.raises(pytest.skip.Exception, match="onnxruntime"):
        stub_detector_factory()
