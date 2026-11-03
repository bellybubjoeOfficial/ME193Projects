"""Load config.yaml (the single home for every tunable value)."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.yaml"


def load_config(path=None):
    with open(path or DEFAULT_PATH) as f:
        return yaml.safe_load(f)


def resolve(path_str):
    """Paths in config.yaml are relative to the project folder."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def set_values(updates, path=None):
    """Rewrite only the given values in config.yaml, keeping comments and layout.

    updates: {("section", "key"): value} for keys one level deep.
    """
    path = path or DEFAULT_PATH
    lines = Path(path).read_text().split("\n")
    section, done = None, set()
    for i, line in enumerate(lines):
        top = re.match(r"^([A-Za-z_]\w*):\s*(#.*)?$", line)
        if top:
            section = top.group(1)
            continue
        m = re.match(r"^  ([A-Za-z_]\w*):(\s*)([^#]*?)(\s*)(#.*)?$", line)
        if m and (section, m.group(1)) in updates:
            key, sp, old, gap, comment = m.groups()
            val = updates[(section, key)]
            new = f"{val:g}" if isinstance(val, float) else str(val)
            if comment:  # keep the comment column where it was
                gap = " " * max(1, len(old) + len(gap) - len(new))
                lines[i] = f"  {key}:{sp}{new}{gap}{comment}"
            else:
                lines[i] = f"  {key}:{sp}{new}"
            done.add((section, key))
    missing = set(updates) - done
    if missing:
        raise KeyError(f"not in {path}: {sorted(missing)}")
    Path(path).write_text("\n".join(lines))
