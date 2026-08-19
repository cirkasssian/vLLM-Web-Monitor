"""Minimal Prometheus exposition-format parser (stdlib only)."""

import re
from typing import Callable, Dict, Optional

METRIC_LINE_RE = re.compile(
    r'^(\w+:\w+)\{([^}]*)\}\s+([-+\d.Ee]+)$|^(\w+:\w+)\s+([-+\d.Ee]+)$'
)
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> Dict[str, dict]:
    """Parse Prometheus exposition format into {metric_name: {labels: float}}."""
    out: Dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        if m.group(1):
            name, labels_str, val = m.group(1), m.group(2), m.group(3)
        else:
            name, labels_str, val = m.group(4), '', m.group(5)
        labels = dict(LABEL_RE.findall(labels_str))
        try:
            out.setdefault(name, {})[tuple(sorted(labels.items()))] = float(val)
        except ValueError:
            pass
    return out


def pick(metrics: Dict[str, dict], name: str, pred: Optional[Callable[[dict], bool]] = None) -> Optional[float]:
    """Pick a single value from a metric, optionally filtered by label predicate."""
    buckets = metrics.get(name)
    if not buckets:
        return None
    if pred is None:
        if len(buckets) == 1:
            return next(iter(buckets.values()))
        # Sum across engines (multi-GPU deployments)
        return sum(buckets.values())
    for labels, val in buckets.items():
        if pred(dict(labels)):
            return val
    return None
