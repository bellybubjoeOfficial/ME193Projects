"""Stage 7 -- the parameters page saves values without wrecking config.yaml."""

import pytest
import yaml

from laptop.config import set_values

TEXT = """# header comment
train:
  threshold: 0.5            # strike probability needed
  refractory_s: 0.3
live:
  flash_s: 0.35            # flash length
  nested:
    deep: 1
"""


def test_updates_values_keeps_comments(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(TEXT)
    set_values({("train", "threshold"): 0.625, ("train", "refractory_s"): 0.25,
                ("live", "flash_s"): 0.5}, p)
    out = p.read_text()
    cfg = yaml.safe_load(out)
    assert cfg["train"] == {"threshold": 0.625, "refractory_s": 0.25}
    assert cfg["live"]["flash_s"] == 0.5 and cfg["live"]["nested"]["deep"] == 1
    assert "# strike probability needed" in out and "# header comment" in out
    assert out.index("# strike") == TEXT.index("# strike")  # comment column unchanged


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(TEXT)
    with pytest.raises(KeyError):
        set_values({("train", "nope"): 1}, p)
    assert p.read_text() == TEXT


def test_real_config_round_trip(tmp_path):
    from laptop.config import DEFAULT_PATH
    p = tmp_path / "config.yaml"
    p.write_text(DEFAULT_PATH.read_text())
    before = yaml.safe_load(p.read_text())
    set_values({("train", "threshold"): before["train"]["threshold"]}, p)
    assert yaml.safe_load(p.read_text()) == before
