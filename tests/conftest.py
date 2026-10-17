import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def profile():
    return yaml.safe_load((ROOT / "examples" / "profile.example.yaml").read_text())


@pytest.fixture
def postings():
    return ROOT / "examples" / "postings"


@pytest.fixture(autouse=True)
def _no_aws(monkeypatch, tmp_path):
    # Tests never touch AWS: wipe credentials and point runs at a temp dir.
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AOS_RUNS_DIR", str(tmp_path / "runs"))
