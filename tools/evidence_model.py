"""Versioned local model artifacts for the public-engineer policy."""
import hashlib
import json
import math
from pathlib import Path

from learn_short import EXE, RULES

ENCODING = "public-engineer-v1"
WIDTHS = {"public-engineer-v1": 6, "short-v1": 5, "position-v1": 6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, weights, parent=None, executable=EXE, rules=RULES, encoding=ENCODING):
    model = dict(schema=1, encoding=encoding, rules_sha256=sha(rules),
                 engine_sha256=sha(executable), parent_sha256=parent, weights=weights)
    validate(model, executable, rules, encoding)
    with Path(path).open("x", encoding="utf8") as output:
        json.dump(model, output, indent=2, allow_nan=False)
        output.write(chr(10))
    return load(path, executable, rules, encoding)


def validate(model, executable=EXE, rules=RULES, encoding=ENCODING):
    if (encoding not in WIDTHS or model.get("schema") != 1 or model.get("encoding") != encoding or
            model.get("rules_sha256") != sha(rules) or model.get("engine_sha256") != sha(executable)):
        raise ValueError("model schema, encoding, rules or executable mismatch")
    weights = model.get("weights")
    if (not isinstance(weights, list) or len(weights) != WIDTHS[encoding] or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in weights)):
        raise ValueError("model must have the declared number of finite numeric weights")
    parent = model.get("parent_sha256")
    if parent is not None and (not isinstance(parent, str) or len(parent) != 64 or
                               any(c not in "0123456789abcdef" for c in parent)):
        raise ValueError("invalid parent digest")
    return weights


def load(path, executable=EXE, rules=RULES, encoding=ENCODING):
    model = json.loads(Path(path).read_text(encoding="utf8"))
    validate(model, executable, rules, encoding)
    return model
