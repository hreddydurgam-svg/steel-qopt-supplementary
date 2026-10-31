"""Shared helpers: profiles, result folders, environment record, JSON/CSV writing."""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def results_dir(name):
    d = ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "unknown", "cpu_count": os.cpu_count(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for mod in ["numpy", "scipy", "pandas", "matplotlib", "pulp", "qiskit", "qiskit_aer", "qbraid"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = "not installed"
    try:
        with open("/proc/meminfo") as fh:
            info["mem_total"] = fh.readline().split(":")[1].strip()
    except Exception:
        pass
    return info


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_Enc)


def profile(cfg, name):
    if name not in cfg["profiles"]:
        raise ValueError(f"profile {name!r} not in {list(cfg['profiles'])}")
    return cfg["profiles"][name]


def parse_profile(argv=None, default="quick", alpha=False, methods=False):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=default, choices=["quick", "paper"])
    if methods:
        ap.add_argument("--methods", default=None, help="comma list, e.g. M1,M2 (default: the profile's list)")
    if alpha:
        ap.add_argument("--alpha", type=float, default=None,
                        help="CVaR alpha override; must be methods.common.cvar_alpha or listed in cvar_alpha_sensitivity")
    args = ap.parse_args(argv)
    return args


def apply_alpha(cfg, args):
    """Apply a CVaR alpha declared in the parameter file (primary or sensitivity value). Returns (alpha, folder suffix).
    The parameter file is not changed on disk and its hash stays the same; the alpha used is recorded in every output."""
    c = cfg["methods"]["common"]
    primary = c["cvar_alpha"]
    if args.alpha is None or args.alpha == primary:
        return primary, ""
    allowed = list(c.get("cvar_alpha_sensitivity", []))
    if args.alpha not in allowed:
        raise SystemExit(f"--alpha {args.alpha} is not declared in the parameter file (primary {primary}, sensitivity {allowed})")
    c["cvar_alpha"] = args.alpha
    return args.alpha, f"_alpha{args.alpha:g}"


def banner(title, p):
    print("=" * 78)
    print(title)
    print(f"params spec v{p.raw['_spec']['version']}  sha256 {p.sha256}")
    print("=" * 78)


# ------------------------------------------------------------------ checkpoints (stop/restart at any time)
_CKPT = {"dir": None}


def set_checkpoint(out, sha):
    """Every expensive unit of work (one training chain, one tuning candidate, one gradient-variance estimate, ...)
    is stored in <out>/_checkpoint as soon as it finishes. Re-running the script reuses finished units, so a
    stopped run continues where it stopped. Results are identical to an uninterrupted run (each unit has its own
    fixed seed). A checkpoint folder written with a different parameter file is refused."""
    d = Path(out) / "_checkpoint"
    d.mkdir(parents=True, exist_ok=True)
    tag = d / "params_sha256.txt"
    if tag.exists() and tag.read_text().strip() != sha:
        raise SystemExit(f"{d} belongs to a different parameter file; delete that folder and re-run.")
    tag.write_text(sha)
    _CKPT["dir"] = d
    n = sum(1 for _ in d.glob("*.json"))
    if n:
        print(f"[checkpoint] {n} finished units found in {d} - they are reused")
    return d


def fingerprint(*parts):
    import hashlib
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            h.update(np.ascontiguousarray(part).tobytes()); h.update(str(part.shape).encode())
        else:
            h.update(json.dumps(part, sort_keys=True, cls=_Enc, default=str).encode())
        h.update(b"|")
    return h.hexdigest()[:32]


def cached(kind, key, fn):
    """Return fn() and store it; if a stored result for (kind, key) exists, return it instead."""
    d = _CKPT["dir"]
    if d is None:
        return fn()
    p = d / f"{kind}_{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass                                       # incomplete file from an interruption: recompute
    r = fn()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, cls=_Enc))
    os.replace(tmp, p)
    return r
