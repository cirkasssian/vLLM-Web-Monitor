#!/usr/bin/env python3
"""
Web dashboard for vLLM metrics - stdlib only.

Polls /metrics from a vLLM server, computes derived metrics (rates, latencies),
and serves an HTML page that updates in real-time via long-polling.

Usage:
    python3 vllm_web_dashboard.py --vllm-url http://localhost:8000 --port 8501

No third-party dependencies. Python 3.9+.
"""

import argparse
import json
import os
import re
import signal
import threading
import time
import urllib.request
import urllib.error
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

# Settings persist alongside the app file.
APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / 'config.json'
MIN_INTERVAL, MAX_INTERVAL = 1.0, 300.0


def load_settings() -> Dict:
    """Read persisted settings; return {} when the file is absent or invalid."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save_settings(settings: Dict) -> None:
    """Atomically write settings to CONFIG_PATH."""
    tmp = CONFIG_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(settings, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, CONFIG_PATH)


def save_setting(key: str, value: float) -> None:
    """Persist a single setting, preserving other keys."""
    settings = load_settings()
    settings[key] = value
    save_settings(settings)

# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

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


def pick(metrics: Dict[str, dict], name: str, pred=None) -> Optional[float]:
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


# ---------------------------------------------------------------------------
# Snapshot computation
# ---------------------------------------------------------------------------

class Sampler:
    """Keeps the last few parsed snapshots to compute rates & recent means."""

    def __init__(self, maxlen: int = 60):
        self.history: deque = deque(maxlen=maxlen)

    def ingest(self, raw: Dict[str, dict], ts: float):
        snap = {
            'ts': ts,
            'raw': raw,
            # counters
            'prompt_tokens': pick(raw, 'vllm:prompt_tokens_total'),
            'gen_tokens': pick(raw, 'vllm:generation_tokens_total'),
            'prefix_hits': pick(raw, 'vllm:prefix_cache_hits_total'),
            'prefix_queries': pick(raw, 'vllm:prefix_cache_queries_total'),
            'spec_accepted': pick(raw, 'vllm:spec_decode_num_accepted_tokens_total'),
            'spec_draft_tokens': pick(raw, 'vllm:spec_decode_num_draft_tokens_total'),
            'spec_drafts': pick(raw, 'vllm:spec_decode_num_drafts_total'),
            'success_total': pick(raw, 'vllm:request_success_total'),
            'success_stop': pick(raw, 'vllm:request_success_total', lambda l: l.get('finished_reason') == 'stop'),
            # gauges
            'running': pick(raw, 'vllm:num_requests_running'),
            'waiting': pick(raw, 'vllm:num_requests_waiting'),
            'preempted': pick(raw, 'vllm:num_preemptions_total'),
            'kv_usage': (lambda a, b: a if a is not None else b)(
                pick(raw, 'vllm:kv_cache_usage_perc'),
                pick(raw, 'vllm:gpu_cache_usage_perc')),
            # model info — extract from a label-bearing gauge
            'model_name': None,
            'cache_config': pick(raw, 'vllm:cache_config_info'),
            'num_blocks': None,
            'mem_util': None,
            'kv_dtype': None,
        }
        cc = raw.get('vllm:cache_config_info')
        if cc:
            for labels in cc.keys():
                d = dict(labels)
                snap['kv_dtype'] = d.get('cache_dtype') or d.get('kv_cache_dtype')
                snap['num_blocks'] = d.get('num_gpu_blocks')
                snap['mem_util'] = d.get('gpu_memory_utilization')
                if snap['model_name'] is None:
                    snap['model_name'] = d.get('model_name')
        if snap['model_name'] is None:
            for mname in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
                buckets = raw.get(mname)
                if buckets:
                    for lbls in buckets.keys():
                        md = dict(lbls)
                        if 'model_name' in md:
                            snap['model_name'] = md['model_name']
                            break
                    if snap['model_name']:
                        break
        self.history.append(snap)

    def current(self) -> Optional[dict]:
        if not self.history:
            return None
        cur = self.history[-1]
        prev = self.history[-2] if len(self.history) >= 2 else None

        dt = (cur['ts'] - prev['ts']) if prev else 0.0

        def rate(key: str) -> Optional[float]:
            if not prev or dt <= 0:
                return None
            new, old = cur.get(key), prev.get(key)
            if new is None or old is None:
                return None
            return max(0.0, (new - old)) / dt

        data = {
            'ts': cur['ts'],
            'online': True,
            'model_name': cur.get('model_name'),
            'kv_dtype': cur.get('kv_dtype'),
            'num_blocks': cur.get('num_blocks'),
            'mem_util': cur.get('mem_util'),
            # gauges
            'running': cur.get('running'),
            'waiting': cur.get('waiting'),
            'preempted': cur.get('preempted'),
            'kv_usage': cur.get('kv_usage'),
            # rates
            'prompt_tok_s': rate('prompt_tokens'),
            'gen_tok_s': rate('gen_tokens'),
            # prefix cache hit ratio
            'prefix_hit_ratio': self._ratio(cur, 'prefix_hits', 'prefix_queries'),
            # spec decode (lifetime cumulative, mirrors terminal)
            'spec_accept_rate': self._spec_accept_lifetime(cur),
            'spec_accept_len': self._spec_accept_len_lifetime(cur),
            # completed totals
            'completed_reqs': cur.get('success_total'),
            'completed_stop': cur.get('success_stop'),
        }

        # Recent-mean latency from histograms (delta of _sum/_count)
        data['latency_e2e'] = self._hist_mean_delta('vllm:e2e_request_latency_seconds')
        data['latency_ttft'] = self._hist_mean_delta('vllm:time_to_first_token_seconds')
        data['latency_tpot'] = self._hist_mean_delta('vllm:request_time_per_output_token_seconds')
        data['queue_time'] = self._hist_mean_delta('vllm:request_queue_time_seconds')

        # Avg request shape (mean of histogram over lifetime)
        data['avg_prompt'] = self._lifetime_mean('vllm:request_prompt_tokens')
        data['avg_gen'] = self._lifetime_mean('vllm:request_generation_tokens')
        # Per-request mean generation rate = avg tokens / avg e2e latency
        # (matches the terminal's Average Request tok/s figure).
        _ag, _ae = data['avg_gen'], data['latency_e2e']
        data['avg_gen_tps'] = (_ag / _ae) if (_ag and _ae and _ae > 0) else None

        # History series (last N points)
        n = min(60, len(self.history))
        data['history'] = {
            'active': [round(h.get('running') or 0, 1) for h in list(self.history)[-n:]],
            'gen_tok_s': [round(rate_i(i) or 0, 1) if False else _safe_rate(self, i) for i in range(len(self.history) - n, len(self.history))],
            'kv_pct': [round((h.get('kv_usage') or 0) * 100, 1) for h in list(self.history)[-n:]],
        }
        return data

    def _ratio(self, cur: dict, num_key: str, den_key: str) -> Optional[float]:
        num, den = cur.get(num_key), cur.get(den_key)
        if num is None or den is None or den == 0:
            return None
        return min(1.0, num / den)

    def _spec_accept_lifetime(self, cur: dict) -> Optional[float]:
        a, d = cur.get('spec_accepted'), cur.get('spec_draft_tokens')
        if a is None or d is None or d <= 0:
            return None
        return max(0.0, min(1.0, a / d))

    def _spec_accept_len_lifetime(self, cur: dict) -> Optional[float]:
        a, dr = cur.get('spec_accepted'), cur.get('spec_drafts')
        if a is None or dr is None or dr <= 0:
            return None
        return max(0.0, a / dr)

    def _hist_mean_delta(self, name: str) -> Optional[float]:
        if len(self.history) < 2:
            return None
        cur_raw = self.history[-1]['raw']
        prev_raw = self.history[-2]['raw']
        s_cur = pick(cur_raw, f'{name}_sum') or 0.0
        c_cur = pick(cur_raw, f'{name}_count') or 0.0
        s_prev = pick(prev_raw, f'{name}_sum') or 0.0
        c_prev = pick(prev_raw, f'{name}_count') or 0.0
        ds, dc = s_cur - s_prev, c_cur - c_prev
        if dc <= 0:
            # fall back to lifetime mean
            if c_cur <= 0:
                return None
            return s_cur / c_cur
        return ds / dc

    def _lifetime_mean(self, name: str) -> Optional[float]:
        raw = self.history[-1]['raw'] if self.history else {}
        s = pick(raw, f'{name}_sum')
        c = pick(raw, f'{name}_count')
        if s is None or c is None or c <= 0:
            return None
        return s / c


def _safe_rate(sampler: Sampler, idx: int) -> Optional[float]:
    hist = sampler.history
    if idx <= 0:
        return None
    a, b = hist[idx - 1], hist[idx]
    gt_a, gt_b = a.get('gen_tokens'), b.get('gen_tokens')
    if gt_a is None or gt_b is None:
        return None
    dt = b['ts'] - a['ts']
    if dt <= 0:
        return None
    return max(0.0, (gt_b - gt_a) / dt)


# ---------------------------------------------------------------------------
# HTTP server with long-poll
# ---------------------------------------------------------------------------

PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title data-i18n-title-tag="title_tag">vllm-monitor · vLLM Health Dashboard</title>
<style>
:root{
  --bg:#0b0e14; --panel:#0b0e14; --fg:#d8dee9;
  --dim:#5e6a7e; --white:#ffffff;
  --cyan:#56d4dd; --warn:#ffd166; --err:#ff6b6b;
  --shadow:rgba(0,0,0,.5);
  --mono:'JetBrains Mono','SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace;
}
/* accent: default green; overridden at runtime via inline --accent on <html> */
:root{ --accent:#3fb950; }
[data-theme="light"]{
  --bg:#f4f6f8; --panel:#ffffff; --fg:#1b2733;
  --dim:#6b7785; --white:#0b1220;
  --cyan:#0e7490; --warn:#b7791f; --err:#d64545;
  --shadow:rgba(0,0,0,.15);
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:var(--mono);font-size:14px;line-height:1.35;padding:18px}
.wrap{max-width:1500px;margin:0 auto}
/* ---- Header (two rows, rounded box, title top-left) ---- */
.header{border:1px solid var(--accent);border-radius:14px;padding:10px 16px;margin-bottom:18px;display:flex;align-items:stretch;justify-content:space-between;gap:16px}
.hdr-left{flex:1 1 auto;min-width:0}
.header .row1{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.header .row2{display:flex;align-items:center;gap:12px;flex-wrap:wrap;color:var(--dim);margin-top:3px}
.hdr-right{display:flex;flex-direction:column;justify-content:flex-start;align-items:flex-end;line-height:1.3;flex:none}
.hdr-right a{color:var(--accent);cursor:pointer;text-decoration:none;font-weight:700;font-size:14px}
.hdr-right a:hover{text-decoration:underline}
.theme-toggle{cursor:pointer;display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;padding:2px;border-radius:5px;user-select:none;margin-top:4px;margin-right:-4px;color:var(--accent)}
.theme-toggle svg{display:block;width:16px;height:16px}
.theme-toggle:hover{background:var(--bg)}
.theme-toggle:focus{outline:none}
.theme-toggle:focus-visible{box-shadow:0 0 0 2px var(--bg),0 0 0 4px var(--accent)}
.dot{width:9px;height:9px;border-radius:50%;background:#3fb950;box-shadow:0 0 8px #3fb950;flex:none;position:relative;top:-1px}
.dot.off{background:#ff6b6b;box-shadow:0 0 8px #ff6b6b}
.st-online{color:var(--accent);font-weight:700}
.st-offline{color:var(--err);font-weight:700}
.pause-btn{display:inline-flex;align-items:center;justify-content:center;min-width:12px;height:18px;padding:0 1px;margin-left:-2px;position:relative;top:-1px;border-radius:4px;color:var(--accent);cursor:pointer;user-select:none;font-size:10px;letter-spacing:-1px;line-height:1}
.pause-btn::before{content:'';position:absolute;left:-8px;right:-8px;top:-6px;bottom:-6px}
.pause-btn:hover{background:var(--border)}
.pause-btn:focus{outline:none}
.pause-btn:focus-visible{box-shadow:0 0 0 2px var(--bg),0 0 0 4px var(--accent)}
.pause-btn.paused{color:var(--warn)}
.pb-glyph{display:inline-block;line-height:1}
.dim{color:var(--dim)}
.mono-model{color:var(--cyan);font-weight:700}
.sep{color:var(--dim)}
/* ---- Sections ---- */
.sec{margin-top:16px}
.sec-label{color:var(--accent);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.14em;margin-bottom:8px}
.tiles{display:grid;gap:12px}
.t3{grid-template-columns:repeat(3,1fr)}
.t4{grid-template-columns:repeat(4,1fr)}
/* ---- Tile: title sits on the top border, value centered in body ---- */
.tile{position:relative;border:1px solid var(--accent);border-radius:8px;padding:16px 14px 12px;min-height:62px;display:flex;align-items:center;justify-content:center}
.tile .lbl{position:absolute;top:-8px;left:12px;background:var(--bg);padding:0 6px;color:var(--dim);font-size:12px}
.tile .val{text-align:center;font-weight:700;font-size:20px;white-space:nowrap}
.tile .val.sm{font-size:16px}
.tile .val .l1{display:block}
.tile .val .l2{display:block;color:var(--dim);font-weight:400;font-size:12px;margin-top:3px}
.c-white{color:var(--white)} .c-green{color:var(--accent)} .c-yellow{color:var(--warn)}
.c-red{color:var(--err)} .c-dim{color:var(--dim);font-weight:400}
/* ---- History: y-axis bar chart, axis on left, bars right, caption below ---- */
/* override base .tile flex-centering so the plot can stretch full width */
.tile.spark{display:block}
.spark{padding:14px 14px 12px 8px}
.spark .plot{display:flex;gap:4px;align-items:stretch;width:100%}
.spark .yax{display:flex;flex-direction:column;justify-content:space-between;align-items:flex-end;color:var(--dim);font-size:11px;width:auto;min-width:16px;padding-bottom:1px;flex:none}
.spark .bars{flex:1 1 auto;min-width:0;display:flex;align-items:flex-end;gap:1px;height:120px;border-left:1px solid var(--accent);padding-left:1px}
.spark .bar{flex:1 1 0;background:var(--accent);opacity:.85;min-height:0;transition:height .25s}
.spark .cap{color:var(--dim);font-size:12px;margin-top:10px;text-align:center}
.footer{margin-top:18px;color:var(--dim);font-size:11px;text-align:center}
.footer a{color:var(--accent);cursor:pointer;text-decoration:none}
.footer a:hover{text-decoration:underline}
.modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:50}
.modal-backdrop[hidden]{display:none}
.modal{background:var(--panel,#11151f);border:1px solid var(--accent);border-radius:8px;padding:18px 20px 20px 20px;width:min(360px,90vw);box-shadow:0 8px 30px rgba(0,0,0,.5)}
.modal-head{font-weight:700;color:var(--accent);margin-bottom:14px;font-size:14px}
.fld{display:block;margin-bottom:12px}
.fld-lbl{display:block;color:var(--fg);font-size:12px;margin-bottom:6px}
.fld input[type=number]{width:100%;background:var(--bg);border:1px solid var(--accent);border-radius:6px;color:var(--fg);font-family:var(--mono);font-size:14px;padding:8px 10px}
.fld input:focus{outline:none;border-color:var(--accent)}
.fld select{width:100%;background:var(--bg);border:1px solid var(--accent);border-radius:6px;color:var(--fg);font-family:var(--mono);font-size:14px;padding:8px 10px;cursor:pointer}
.fld select:focus{outline:none;border-color:var(--accent)}
.fld-hint{display:block;color:var(--dim);font-size:11px;margin-top:5px}
.seg{display:flex;gap:4px;background:var(--bg);border:1px solid var(--accent);border-radius:8px;padding:3px}
.seg-opt{flex:1;position:relative}
.seg-opt input{position:absolute;opacity:0;pointer-events:none}
.seg-opt span{display:block;text-align:center;padding:6px 4px;border-radius:6px;color:var(--dim);font-size:12px;cursor:pointer;user-select:none}
.seg-opt input:checked+span{background:var(--accent);color:var(--bg);font-weight:700}
[data-theme="light"] .seg-opt input:checked+span{color:#fff}
.seg-opt span:hover{color:var(--fg)}
.seg-opt input:checked+span:hover{color:inherit}
.swatches{display:flex;gap:10px;flex-wrap:wrap}
.sw{position:relative;cursor:pointer}
.sw input{position:absolute;opacity:0;pointer-events:none}
.sw-dot{display:block;width:26px;height:26px;border-radius:50%;background:var(--sw);border:2px solid transparent;box-shadow:0 0 0 2px var(--bg) inset}
.sw input:checked+.sw-dot{border-color:var(--fg);transform:scale(1.08)}
.sw:hover .sw-dot{filter:brightness(1.12)}
.sw-rc{position:relative;width:26px;height:26px;border-radius:50%;border:none;padding:0;cursor:pointer;background:transparent}
.sw-rc-ring{display:block;width:26px;height:26px;border-radius:50%;background:conic-gradient(#f00,#ff0,#0f0,#0ff,#00f,#f0f,#f00);box-shadow:0 0 0 2px var(--bg) inset}
.sw-rc:hover .sw-rc-ring{filter:brightness(1.12)}
.sw-rc.active-cust .sw-rc-ring{border:2px solid var(--fg);transform:scale(1.08)}
.sw-rc-center{position:absolute;inset:5px;border-radius:50%;background:var(--accent);border:2px solid var(--bg)}
.cp-pop{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;z-index:60}
.cp-pop[hidden]{display:none}
.cp-pop-inner{background:var(--panel);border:1px solid var(--accent);border-radius:10px;padding:14px 16px;width:min(280px,90vw);box-shadow:0 10px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;gap:10px}
.cp-pop-head{display:flex;align-items:center;justify-content:space-between;color:var(--fg);font-size:13px;font-weight:700}
.cp-x{background:transparent;border:none;color:var(--dim);font-size:20px;line-height:1;cursor:pointer;padding:0 2px}
.cp-x:hover{color:var(--fg)}
.sv-canvas{position:relative;height:120px;border-radius:8px;cursor:crosshair;background:linear-gradient(to top,#000,transparent),linear-gradient(to right,#fff,var(--sv-hue,#f00));touch-action:none}
.sv-cursor{position:absolute;width:14px;height:14px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 3px rgba(0,0,0,.5);transform:translate(-50%,-50%);pointer-events:none}
.cp-row{display:flex;align-items:center;gap:8px}
.cp-hex{flex:1;min-width:0;background:var(--bg);border:1px solid var(--accent);border-radius:6px;color:var(--fg);font-family:var(--mono);font-size:13px;padding:6px 8px;text-transform:lowercase}
.cp-hex:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 2px var(--accent)}
.cp-swatch{width:28px;height:28px;border-radius:6px;border:1px solid var(--accent);flex:none;background:var(--accent)}
.cp-chan{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:11px;font-weight:700}
.cp-chan>span:first-child{width:10px}
.cp-range{-webkit-appearance:none;appearance:none;flex:1;height:10px;border-radius:5px;outline:none;cursor:pointer;border:1px solid var(--accent)}
.ch-r{background:linear-gradient(to right,#000,#f00)}
.ch-g{background:linear-gradient(to right,#000,#0f0)}
.ch-b{background:linear-gradient(to right,#000,#00f)}
.cp-range::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;background:var(--fg);border:2px solid var(--bg);cursor:pointer}
.cp-range::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--fg);border:2px solid var(--bg);cursor:pointer}
.modal-actions{display:flex;gap:10px;justify-content:flex-end;margin-top:22px}
.btn{background:var(--accent);color:#0b0e14;border:none;border-radius:6px;padding:8px 16px;font-family:var(--mono);font-size:13px;font-weight:700;cursor:pointer}
.btn:hover{filter:brightness(1.1)}
.btn.ghost{background:transparent;color:var(--dim);border:1px solid var(--accent)}
.btn.ghost:hover{color:var(--fg);filter:none}
.modal-msg{margin-top:2px;font-size:12px}
.modal-msg.ok{color:var(--accent)} .modal-msg.err{color:var(--err)}
@media (max-width:900px){.t3,.t4{grid-template-columns:1fr 1fr}}
@media (max-width:560px){.t3,.t4{grid-template-columns:1fr}}
</style></head><body>
<div class="wrap">

<div class="header">
  <div class="hdr-left">
    <div class="row1">
      <div class="dot" id="dot" data-i18n-title="tip_dot"></div>
      <span class="st-online" id="hdr-status" data-i18n-title="tip_status">ONLINE</span>
      <span class="pause-btn" id="pause-btn" tabindex="0"><span class="pb-glyph">&#10074;&#10074;</span></span>
      <span class="sep dim">·</span>
      <span class="dim" id="url" data-i18n-title="tip_url">—</span>
      <span class="sep dim">·</span>
      <span class="dim" data-i18n-title="tip_interval_hdr">refresh <span id="iv">?</span>s</span>
    </div>
    <div class="row2" id="modelbar">
      <span class="mono-model" id="model" data-i18n-title="tip_model">—</span>
    </div>
  </div>
  <div class="hdr-right">
    <a href="#" id="settings-btn" data-i18n-title="tip_settings" data-settings-label>⚙ <span data-i18n="set_btn_label">настройки</span></a>
    <span class="theme-toggle" id="theme-toggle" data-i18n-title="tip_theme_toggle" role="button" tabindex="0"><svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg></span>
  </div>
</div>

<div class="sec"><div class="sec-label">Load</div>
  <div class="tiles t3">
    <div class="tile" data-i18n-title="tip_running"><div class="lbl">Running</div><div class="val c-white" id="running">—</div></div>
    <div class="tile" data-i18n-title="tip_queued"><div class="lbl">Queued</div><div class="val c-white" id="waiting">—</div></div>
    <div class="tile" data-i18n-title="tip_preempted"><div class="lbl">Preemptions</div><div class="val c-white" id="preempted">—</div></div>
  </div>
</div>

<div class="sec"><div class="sec-label">Latency</div>
  <div class="tiles t4">
    <div class="tile" data-i18n-title="tip_e2e"><div class="lbl">E2E Latency</div><div class="val" id="e2e">—</div></div>
    <div class="tile" data-i18n-title="tip_ttft"><div class="lbl">TTFT</div><div class="val" id="ttft">—</div></div>
    <div class="tile" data-i18n-title="tip_tpot"><div class="lbl">TPOT</div><div class="val" id="tpot">—</div></div>
    <div class="tile" data-i18n-title="tip_queue"><div class="lbl">Queue Time</div><div class="val" id="queue">—</div></div>
  </div>
</div>

<div class="sec"><div class="sec-label">Throughput &amp; Cache</div>
  <div class="tiles t4">
    <div class="tile" data-i18n-title="tip_ptok"><div class="lbl">Prompt Tokens/s</div><div class="val c-white" id="ptok">—</div></div>
    <div class="tile" data-i18n-title="tip_gtok"><div class="lbl">Gen Tokens/s</div><div class="val c-white" id="gtok">—</div></div>
    <div class="tile" data-i18n-title="tip_kv"><div class="lbl">GPU KV Cache</div><div class="val c-white" id="kv">—</div></div>
    <div class="tile" data-i18n-title="tip_pcache"><div class="lbl">Prefix Cache Hit</div><div class="val c-white" id="pcache">—</div></div>
  </div>
</div>

<div class="sec"><div class="sec-label">Stats</div>
  <div class="tiles t3">
    <div class="tile" data-i18n-title="tip_spec"><div class="lbl">Spec Accept (MTP)</div>
      <div class="val sm" id="spec"><span class="l1">—</span><span class="l2" id="spec-sub"></span></div></div>
    <div class="tile" data-i18n-title="tip_done"><div class="lbl">Completed</div>
      <div class="val sm" id="done"><span class="l1">—</span><span class="l2" id="done-sub"></span></div></div>
    <div class="tile" data-i18n-title="tip_shape"><div class="lbl">Average Request</div>
      <div class="val sm" id="shape"><span class="l1">—</span><span class="l2" id="shape-sub"></span></div></div>
  </div>
</div>

<div class="sec"><div class="sec-label">History</div>
  <div class="tiles t3">
    <div class="tile spark" data-i18n-title="tip_hist_active"><div class="lbl">Active Requests</div>
      <div class="plot"><div class="yax"><span id="ax-active-top">0</span><span>0</span></div><div class="bars" id="ch-active"></div></div>
      <div class="cap" id="cap-active">current: 0</div></div>
    <div class="tile spark" data-i18n-title="tip_hist_gen"><div class="lbl">Gen Tokens/s</div>
      <div class="plot"><div class="yax"><span id="ax-gen-top">0</span><span>0</span></div><div class="bars" id="ch-gen"></div></div>
      <div class="cap" id="cap-gen">current: 0.0 tok/s</div></div>
    <div class="tile spark" data-i18n-title="tip_hist_kv"><div class="lbl">GPU Cache %</div>
      <div class="plot"><div class="yax"><span id="ax-kv-top">0%</span><span>0</span></div><div class="bars" id="ch-kv"></div></div>
      <div class="cap" id="cap-kv">current: 0.0%</div></div>
  </div>
</div>


<div class="modal-backdrop" id="settings-modal" hidden>
  <div class="modal">
    <div class="modal-head" data-i18n="set_head">Настройки</div>
    <div class="fld">
      <span class="fld-lbl" data-i18n="set_lang_lbl">Язык интерфейса</span>
      <select id="set-lang"></select>
    </div>
    <label class="fld">
      <span class="fld-lbl" data-i18n="set_interval_lbl">Интервал обновления, сек</span>
      <input type="number" id="set-interval" min="1" max="300" step="0.5" />
      <span class="fld-hint" data-i18n="set_interval_hint">Как часто дашборд опрашивает vLLM /metrics. Диапазон 1–300 с.</span>
    </label>
    <div class="fld">
      <span class="fld-lbl" data-i18n="set_theme_lbl">Тема оформления</span>
      <div class="seg" id="set-theme-seg">
        <label class="seg-opt"><input type="radio" name="set-theme" value="system" /><span data-i18n="theme_system">Системная</span></label>
        <label class="seg-opt"><input type="radio" name="set-theme" value="dark" /><span data-i18n="theme_dark">Тёмная</span></label>
        <label class="seg-opt"><input type="radio" name="set-theme" value="light" /><span data-i18n="theme_light">Светлая</span></label>
      </div>
      <span class="fld-hint" data-i18n="set_theme_hint">«Системная» следует за prefers-color-scheme ОС.</span>
    </div>
    <div class="fld">
      <span class="fld-lbl" data-i18n="set_accent_lbl">Акцентный цвет</span>
      <div class="swatches" id="set-accent-swatches">
        <label class="sw" data-i18n-title="sw_green"><input type="radio" name="set-accent-presets" value="#3fb950"/><span class="sw-dot" style="--sw:#3fb950"></span></label>
        <label class="sw" data-i18n-title="sw_blue"><input type="radio" name="set-accent-presets" value="#58a6ff"/><span class="sw-dot" style="--sw:#58a6ff"></span></label>
        <label class="sw" data-i18n-title="sw_purple"><input type="radio" name="set-accent-presets" value="#bc8cff"/><span class="sw-dot" style="--sw:#bc8cff"></span></label>
        <label class="sw" data-i18n-title="sw_orange"><input type="radio" name="set-accent-presets" value="#ffa657"/><span class="sw-dot" style="--sw:#ffa657"></span></label>
        <label class="sw" data-i18n-title="sw_red"><input type="radio" name="set-accent-presets" value="#ff7b72"/><span class="sw-dot" style="--sw:#ff7b72"></span></label>
        <label class="sw" data-i18n-title="sw_pink"><input type="radio" name="set-accent-presets" value="#ff7eb6"/><span class="sw-dot" style="--sw:#ff7eb6"></span></label>
        <button type="button" class="sw-rc" id="set-accent-open" data-i18n-title="sw_custom">
          <span class="sw-rc-ring"></span>
          <span class="sw-rc-center" id="sw-rc-center"></span>
        </button>
      </div>
      <span class="fld-hint" data-i18n="set_accent_hint">Цвет рамок, графиков и кнопок.</span>
    </div>
    <div class="cp-pop" id="cp-pop" hidden>
      <div class="cp-pop-inner">
        <div class="cp-pop-head"><span data-i18n="pop_head">Произвольный цвет</span><button type="button" class="cp-x" id="cp-close" data-i18n-title="pop_close">&times;</button></div>
        <div class="sv-canvas" id="sv-canvas" data-i18n-title="sv_canvas_tip">
          <div class="sv-cursor" id="sv-cursor"></div>
        </div>
        <div class="cp-row">
          <input type="text" id="set-accent-hex" class="cp-hex" maxlength="7" spellcheck="false" data-i18n-aria="hex_aria" aria-label="HEX-код цвета" value="#3fb950"/>
          <span class="cp-swatch" id="cp-swatch"></span>
        </div>
        <label class="cp-chan">R<input type="range" id="cp-r" class="cp-range ch-r" min="0" max="255" value="63"/></label>
        <label class="cp-chan">G<input type="range" id="cp-g" class="cp-range ch-g" min="0" max="255" value="185"/></label>
        <label class="cp-chan">B<input type="range" id="cp-b" class="cp-range ch-b" min="0" max="255" value="80"/></label>
        <button type="button" class="btn" id="set-accent-ok" data-i18n-title="ok_apply">OK</button>
      </div>
    </div>
    <div class="modal-actions">
      <button class="btn" id="set-cancel" data-i18n="btn_cancel">Закрыть</button>
    </div>
    <div class="modal-msg" id="set-msg"></div>
  </div>
</div>

</div>
<script>
const $=id=>document.getElementById(id);
let auto=true,last=null;
let curInterval=__INTERVAL__;

/* ---- i18n: RU / EN translations ---- */
const I18N={
ru:{
tip_dot:'Состояние связи с vLLM-сервером. Зелёный = /metrics доступен, красный = сервер недоступен или ошибка опроса.',
tip_status:'ONLINE — vLLM-сервер отвечает на запросы /metrics. OFFLINE — связь потеряна.',
tip_pause_on:'Пауза авто-обновления',
tip_pause_off:'Возобновить авто-обновление',
tip_url:'Базовый URL vLLM-сервера, с которого дашборд опрашивает Prometheus-метрики (endpoint /metrics).',
tip_interval_hdr:'Интервал опроса vLLM-сервера, в секундах. Каждые N секунд дашборд запрашивает свежие метрики.',
tip_model:'Имя загруженной в vLLM модели. Читается из метрик (лейб model_name), поэтому обновляется автоматически после смены модели.',
tip_settings:'Открыть настройки (интервал, тема, язык, акцентный цвет)',
tip_theme_toggle:'Быстрое переключение темы (светлая/тёмная)',
tip_running:'Количество запросов, которые в данный момент активно обрабатываются GPU (prefill + decode). 0 = сервер свободен.',
tip_queued:'Запросы, ожидающие своей очереди — не хватает VRAM/KV-блоков или GPU занят другими запросами. 0 = свободно, >0 = есть очередь (перегрузка).',
tip_preempted:'Число запросов, выгруженных из GPU-памяти (preemption) из-за нехватки KV-кэша. 0 = нормально. >0 = KV-кэш переполнен, возможны повторы работы.',
tip_e2e:'End-to-end: общее время обработки запроса от поступления до последнего токена. Среднее за последний интервал опроса. — = нет активных запросов.',
tip_ttft:'Time-To-First-Token: время от запроса до появления первого сгенерированного токена. Зависит от длины prompt и загрузки.',
tip_tpot:'Time-Per-Output-Token: среднее время генерации одного токена после первого. Характеристика скорости decode.',
tip_queue:'Среднее время, которое запрос проводит в очереди до начала обработки. 0 = не ждал.',
tip_ptok:'Скорость обработки входных токенов (prefill). Высокое значение при большом prompt — норма.',
tip_gtok:'Скорость генерации выходных токенов (decode). Основная метрика производительности ответа. 0 = нет активной генерации.',
tip_kv:'Доля занятых блоков KV-кэша. Близко к 100% = мало места для новых/больших запросов, возможен риск preemptions.',
tip_pcache:'Доля запросов, попавших в префиксный кэш (повторные одинаковые начальные части prompt). Чем выше, тем быстрее обработка похожих запросов.',
tip_spec:"Speculative Decoding (MTP): доля принятых draft-токенов. Показывает, насколько эффективно спекулятивная генерация ускоряет вывод. '—' если выключено. Внизу — средняя длина принятого фрагмента (tok/step).",
tip_done:'Всего успешно завершённых запросов с момента старта vLLM. Внизу — средняя длина сгенерированного ответа и число ошибок.',
tip_shape:'Средний размер запроса: сколько входных (prompt) и выходных (generation) токенов в типичном запросе, а также средняя скорость генерации и E2E.',
tip_hist_active:'Динамика числа активных (running) запросов за последние ~2 минуты. Пики = всплески нагрузки. Слева y-ось: пик окна сверху, 0 снизу. Внизу текущее значение.',
tip_hist_gen:'Скорость генерации токенов во времени. Плоская линия на 0 = сервер в простое. Пики = активная генерация. Внизу текущее значение.',
tip_hist_kv:'Использование KV-кэша (%) во времени. Близко к 100% = риск preemptions. Внизу текущее значение.',
tip_kv_dtype:'Тип данных KV-кэша (precision). Например fp8_e5m2, bf16, fp16. Ниже точность — меньше занимаемой видеопамяти.',
tip_blks:'Число доступных KV-блоков на GPU. Определяет максимальное количество одновременных запросов / длину контекста.',
tip_util:'Целевое использование видеопамяти (gpu-memory-utilization). Доля VRAM, которую vLLM резервирует под KV-кэш.',
mdl_title:'Имя загруженной в vLLM модели (из лейба model_name в метриках).',
set_btn_label:'настройки',
set_lang_lbl:'Язык интерфейса',
sw_green:'Зелёный',
sw_blue:'Синий',
sw_purple:'Фиолетовый',
sw_orange:'Оранжевый',
sw_red:'Красный',
sw_pink:'Розовый',
sw_custom:'Произвольный цвет (открыть пикер)',
pop_head:'Произвольный цвет',
pop_close:'Закрыть',
sv_canvas_tip:'Высота/насыщенность',
hex_aria:'HEX-код цвета',
ok_apply:'Применить выбранный цвет',
set_head:'Настройки',
set_interval_lbl:'Интервал обновления, сек',
set_interval_hint:'Как часто дашборд опрашивает vLLM /metrics. Диапазон 1–300 с.',
set_theme_lbl:'Тема оформления',
theme_system:'Системная',
theme_dark:'Тёмная',
theme_light:'Светлая',
set_theme_hint:'«Системная» следует за prefers-color-scheme ОС.',
set_accent_lbl:'Акцентный цвет',
set_accent_hint:'Цвет рамок, графиков и кнопок.',
 btn_cancel:'Закрыть',

 msg_error_prefix:'Ошибка:',
msg_net_prefix:'Сеть:',
msg_interval_invalid:'Интервал должен быть 1–300 сек.',
title_tag:'vllm-monitor · vLLM Health Dashboard'
},
en:{
tip_dot:'Connection status to the vLLM server. Green = /metrics reachable, red = server unreachable or polling error.',
tip_status:'ONLINE — the vLLM server responds to /metrics requests. OFFLINE — connection lost.',
tip_pause_on:'Pause auto-refresh',
tip_pause_off:'Resume auto-refresh',
tip_url:'Base URL of the vLLM server whose Prometheus metrics (the /metrics endpoint) the dashboard polls.',
tip_interval_hdr:'Polling interval for the vLLM server, in seconds. The dashboard fetches fresh metrics every N seconds.',
tip_model:'Name of the model loaded in vLLM. Read from metrics (model_name label), so it updates automatically after a model swap.',
tip_settings:'Open settings (interval, theme, language, accent color)',
tip_theme_toggle:'Quick theme toggle (light/dark)',
tip_running:'Requests actively being processed on the GPU right now (prefill + decode). 0 = idle server.',
tip_queued:'Requests waiting in queue — not enough VRAM/KV blocks or the GPU is busy with other requests. 0 = free, >0 = backlog (overload).',
tip_preempted:'Requests evicted from GPU memory (preemption) due to KV-cache shortage. 0 = healthy. >0 = KV cache is oversubscribed, possible recomputation.',
tip_e2e:'End-to-end: total time to process a request from arrival to the last token. Mean over the last polling interval. — = no active requests.',
tip_ttft:'Time-To-First-Token: time from the request until the first generated token appears. Depends on prompt length and load.',
tip_tpot:'Time-Per-Output-Token: average time to generate one token after the first. Characterizes decode speed.',
tip_queue:'Average time a request spends in the queue before processing starts. 0 = no wait.',
tip_ptok:'Rate of input-token processing (prefill). High values with large prompts are normal.',
tip_gtok:'Rate of output-token generation (decode). The main response-performance metric. 0 = no active generation.',
tip_kv:'Share of occupied KV-cache blocks. Near 100% = little room for new/large requests, risk of preemptions.',
tip_pcache:'Share of requests hitting the prefix cache (repeated identical prompt beginnings). Higher = faster handling of similar requests.',
tip_spec:"Speculative Decoding (MTP): share of accepted draft tokens. Shows how effectively speculative generation speeds up output. '—' when disabled. Below — average accepted fragment length (tok/step).",
tip_done:'Total successfully completed requests since vLLM started. Below — average generated-response length and error count.',
tip_shape:'Average request size: typical prompt (input) and generation (output) token counts, plus average generation speed and E2E.',
tip_hist_active:'Active (running) request count over the last ~2 minutes. Peaks = load spikes. Left y-axis: window peak on top, 0 at bottom. Current value shown below.',
tip_hist_gen:'Token generation rate over time. Flat line at 0 = server idle. Peaks = active generation. Current value shown below.',
tip_hist_kv:'KV-cache usage (%) over time. Near 100% = risk of preemptions. Current value shown below.',
tip_kv_dtype:'KV-cache data type (precision). E.g. fp8_e5m2, bf16, fp16. Lower precision = less video memory used.',
tip_blks:'Number of available KV blocks on the GPU. Determines the max concurrent requests / context length.',
tip_util:'Target video-memory utilization (gpu-memory-utilization). The fraction of VRAM vLLM reserves for the KV cache.',
mdl_title:'Name of the model loaded in vLLM (from the model_name label in metrics).',
set_btn_label:'settings',
set_lang_lbl:'Interface language',
sw_green:'Green',
sw_blue:'Blue',
sw_purple:'Purple',
sw_orange:'Orange',
sw_red:'Red',
sw_pink:'Pink',
sw_custom:'Custom color (open picker)',
pop_head:'Custom color',
pop_close:'Close',
sv_canvas_tip:'Value/saturation',
hex_aria:'Color HEX code',
ok_apply:'Apply the selected color',
set_head:'Settings',
set_interval_lbl:'Refresh interval, sec',
set_interval_hint:'How often the dashboard polls vLLM /metrics. Range 1–300 s.',
set_theme_lbl:'Theme',
theme_system:'System',
theme_dark:'Dark',
theme_light:'Light',
set_theme_hint:'“System” follows the OS prefers-color-scheme.',
set_accent_lbl:'Accent color',
set_accent_hint:'Color of borders, charts and buttons.',
 btn_cancel:'Close',

 msg_error_prefix:'Error:',
msg_net_prefix:'Network:',
msg_interval_invalid:'Interval must be 1–300 seconds.',
title_tag:'vllm-monitor · vLLM Health Dashboard'
}};
let curLang='en';
const LANG_NAMES={ru:'Русский',en:'English'};
function t(key){const d=I18N[curLang]||{};return (key in d)?d[key]:(I18N.en[key]??key);}
function updateModelBarTips(){
  const bar=$('modelbar');if(!bar)return;
  const spans=bar.querySelectorAll('span[data-mtip]');
  spans.forEach(sp=>{const k=sp.getAttribute('data-mtip');if(k)sp.title=t(k);});
}
function applyLang(lang){
  curLang=(lang in I18N)?lang:'en';
  document.querySelectorAll('[data-i18n]').forEach(el=>{const k=el.dataset.i18n;if(k in (I18N[curLang]||{}))el.textContent=t(k);});
  document.querySelectorAll('[data-i18n-title]').forEach(el=>{const k=el.dataset.i18nTitle;el.title=t(k);});
  document.querySelectorAll('[data-i18n-aria]').forEach(el=>{const k=el.dataset.i18nAria;el.setAttribute('aria-label',t(k));});
  const tt=document.querySelector('[data-i18n-title-tag]');if(tt)document.title=t('title_tag');
  updatePauseTip();
  updateModelBarTips();
}
function updatePauseTip(){const b=$('pause-btn');if(b)b.title=t(auto?'tip_pause_on':'tip_pause_off');}

/* ---- formatters mirroring the terminal vllm-monitor ---- */
function fmtCount(n){n=Math.abs(n);
  if(n<1000)return Math.round(n)+'';
  if(n<1e6)return (n/1e3).toFixed(1)+'K';
  if(n<1e9)return (n/1e6).toFixed(1)+'M';
  return (n/1e9).toFixed(1)+'B';}
function fmtDur(s){
  if(Math.abs(s)<1){const ms=s*1000;return (ms<10)?ms.toFixed(1)+'ms':Math.round(ms)+'ms';}
  if(Math.abs(s)<60)return s.toFixed(1)+'s';
  if(Math.abs(s)<3600){const m=Math.floor(Math.round(s)/60),ss=Math.round(s)%60;return m+'m '+ss+'s';}
  const h=Math.floor(Math.round(s)/3600),rem=Math.round(s)%3600;return h+'h '+(rem/60|0)+'m';}
function fmtInt(v){return v==null||isNaN(v)?'—':Math.round(v).toLocaleString('en-US');}
function setVal(id,cls,text){const el=$(id);el.className='val'+(cls?' '+cls:'');el.textContent=text;}

function paintAxis(el,arr,fmtFn,topId){
  el.innerHTML='';
  const finite=arr.filter(x=>x!=null&&!isNaN(x)&&isFinite(x));
  const peak=finite.length?Math.max.apply(null,finite):0;
  $(topId).textContent=fmtFn(peak);
  // one thin column per sample (terminal draws one bar per data point)
  const frag=document.createDocumentFragment();
  for(const v of arr){
    const colEl=document.createElement('div');colEl.className='bar';
    const hv=(v==null||isNaN(v))?0:v;
    colEl.style.height=(peak>0?Math.max(0,hv/peak*100):0)+'%';
    frag.appendChild(colEl);}
  el.appendChild(frag);
  return peak;}

async function tick(){
  try{
    const r=await fetch('/api/status');const d=await r.json();
    if(!d.online){
      $('dot').className='dot off';
      $('hdr-status').textContent='OFFLINE';$('hdr-status').className='st-offline';
      $('url').textContent=d.url||'';return;}
    $('dot').className='dot';
    $('hdr-status').textContent='ONLINE';$('hdr-status').className='st-online';
    $('url').textContent=d.url||'';
    /* model bar */
    const mb=[];
    mb.push('<span class="mono-model" data-mtip="mdl_title" title="'+t('mdl_title')+'">'+(d.model_name||'unknown')+'</span>');
    if(d.kv_dtype)mb.push('<span class="dim" data-mtip="tip_kv_dtype" title="'+t('tip_kv_dtype')+'">kv '+d.kv_dtype+'</span>');
    if(d.num_blocks)mb.push('<span class="dim" data-mtip="tip_blks" title="'+t('tip_blks')+'">'+d.num_blocks+' blks</span>');
    if(d.mem_util!=null)mb.push('<span class="dim" data-mtip="tip_util" title="'+t('tip_util')+'">util '+Math.round(d.mem_util*100)+'%</span>');
    $('modelbar').innerHTML=mb.join('<span class="sep dim">·</span>');
    /* load */
    setVal('running','c-white',Math.round(d.running||0)+'');
    setVal('waiting','c-white',Math.round(d.waiting||0)+'');
    setVal('preempted','c-white',Math.round(d.preempted||0)+'');
    /* latency */
    [['e2e','latency_e2e'],['ttft','latency_ttft'],['tpot','latency_tpot'],['queue','queue_time']].forEach(([id,k])=>{
      const v=d[k];setVal(id,v>0?'c-white':'c-dim',v>0?fmtDur(v):'—');});
    /* throughput & cache */
    setVal('ptok','c-white',(d.prompt_tok_s||0).toFixed(1));
    setVal('gtok','c-white',(d.gen_tok_s||0).toFixed(1));
    setVal('kv','c-white',((d.kv_usage||0)*100).toFixed(1)+'%');
    setVal('pcache','c-white',((d.prefix_hit_ratio||0)*100).toFixed(1)+'%');
    /* stats */
    if(d.spec_accept_rate!=null){
      $('spec').querySelector('.l1').className='l1 c-white';
      $('spec').querySelector('.l1').textContent=(d.spec_accept_rate*100).toFixed(1)+'%';
      $('spec-sub').textContent=d.spec_accept_len!=null?d.spec_accept_len.toFixed(2)+' tok/step':'';
    }else{$('spec').innerHTML='<span class="l1 c-dim">—</span><span class="l2"></span>';}
    const tot=d.completed_reqs||0,stop=d.completed_stop||0,err=Math.max(0,Math.round(tot)-Math.round(stop));
    if(tot>0){
      $('done').querySelector('.l1').className='l1 c-white';
      $('done').querySelector('.l1').textContent=Math.round(tot).toLocaleString('en-US')+' req · '+fmtCount(tot*d.avg_gen||0)+' tok';
      $('done-sub').textContent='len '+Math.round(d.avg_gen||0)+' · err '+err;
    }else{$('done').innerHTML='<span class="l1 c-dim">—</span><span class="l2"></span>';}
    if((d.avg_prompt||0)>0||(d.avg_gen||0)>0){
      $('shape').querySelector('.l1').className='l1 c-white';
      $('shape').querySelector('.l1').textContent=fmtCount(d.avg_prompt||0)+' in · '+fmtCount(d.avg_gen||0)+' out';
      $('shape-sub').textContent=((d.avg_gen_tps!=null)?d.avg_gen_tps.toFixed(1):'0.0')+' tok/s · '+fmtDur(d.latency_e2e||0)+' E2E';
    }else{$('shape').innerHTML='<span class="l1 c-dim">—</span><span class="l2"></span>';}
    /* history */
    const h=d.history||{};
    paintAxis($('ch-active'),h.active||[],x=>Math.round(x)+'','ax-active-top');
    $('cap-active').textContent='current: '+Math.round(d.running||0);
    paintAxis($('ch-gen'),h.gen_tok_s||[],x=>Math.round(x)+'','ax-gen-top');
    $('cap-gen').textContent='current: '+(d.gen_tok_s||0).toFixed(1)+' tok/s';
    paintAxis($('ch-kv'),h.kv_pct||[],x=>Math.round(x)+'%','ax-kv-top');
    $('cap-kv').textContent='current: '+((d.kv_usage||0)*100).toFixed(1)+'%';
    last=d;
  }catch(e){$('hdr-status').textContent='OFFLINE';$('hdr-status').className='st-offline';$('dot').className='dot off';}
}
function setAuto(run){
  auto=run;
  const btn=$('pause-btn');
if(btn){
    btn.innerHTML='<span class="pb-glyph'+(run?'':' play')+'">'+(run?'&#10074;&#10074;':'&#9654;')+'</span>';
    btn.classList.toggle('paused',!run);
    btn.title=t(run?'tip_pause_on':'tip_pause_off');
  }
  if(run)loop();
}
$('pause-btn').onclick=()=>setAuto(!auto);
$('pause-btn').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setAuto(!auto);}};
$('iv').textContent=curInterval;
function loop(){if(!auto)return;tick().then(()=>setTimeout(loop,curInterval*1000));}

/* ---- appearance: theme + accent (accent is any hex) ---- */
let curTheme='system', curAccent='#3fb950';
const PRESET_HEXES=['#3fb950','#58a6ff','#bc8cff','#ffa657','#ff7b72','#ff7eb6'];
const LEGACY_ACCENT={'green':'#3fb950','blue':'#58a6ff','purple':'#bc8cff','orange':'#ffa657','red':'#ff7b72','pink':'#ff7eb6'};
function normAccent(v){
  if(typeof v!=='string')return null;
  v=v.trim().toLowerCase();
  if(LEGACY_ACCENT[v])return LEGACY_ACCENT[v];
  if(/^#[0-9a-f]{6}$/.test(v))return v;
  return null;
}
const mqDark=window.matchMedia('(prefers-color-scheme: dark)');
function applyTheme(mode){
  curTheme=mode;
  const eff=(mode==='light')?'light':(mode==='dark')?'dark':(mqDark.matches?'dark':'light');
  document.documentElement.setAttribute('data-theme',eff);
}
function applyAccent(raw){
  const hex=normAccent(raw);
  if(!hex)return;
  curAccent=hex;
  document.documentElement.style.setProperty('--accent',hex);
}
function applyAppearance(theme,accent){
  if(theme)applyTheme(theme);
  if(accent)applyAccent(accent);
  updateThemeIcon();
}
function effectiveTheme(){
  if(curTheme==='light')return 'light';
  if(curTheme==='dark')return 'dark';
  return mqDark.matches?'dark':'light';
}
const ICON_MOON='<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
const ICON_SUN='<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
function updateThemeIcon(){
  const el=$('theme-toggle');
  if(!el)return;
  // icon shows the theme you'll switch TO (opposite of current effective)
  el.innerHTML=effectiveTheme()==='light'?ICON_MOON:ICON_SUN;
}
function quickToggleTheme(){
  const next=effectiveTheme()==='light'?'dark':'light';
  applyTheme(next);
  updateThemeIcon();
  fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({theme:next})})
    .then(r=>r.json()).then(j=>{if(j&&j.ok){applyTheme(next);updateThemeIcon();}}).catch(()=>{});
}
const themeToggleEl=$('theme-toggle');
if(themeToggleEl){
  themeToggleEl.onclick=quickToggleTheme;
  themeToggleEl.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();quickToggleTheme();}};
}
// hex <-> rgb helpers
function hexToRgb(hex){
  const m=/^#([0-9a-f]{6})$/i.exec(hex||'');
  if(!m)return null;
  const n=parseInt(m[1],16);
  return [(n>>16)&255,(n>>8)&255,n&255];
}
function rgbToHex(r,g,b){
  return '#'+[r,g,b].map(x=>x.toString(16).padStart(2,'0')).join('');
}
// push a hex into the custom picker controls (hex field + R/G/B sliders + swatch)
function pickerSetHex(hex,updatePresets){
  const rgb=hexToRgb(hex);
  if(!rgb)return;
  const [r,g,b]=rgb;
  $('set-accent-hex').value=hex.toLowerCase();
  $('cp-r').value=r;$('cp-g').value=g;$('cp-b').value=b;
  $('cp-swatch').style.background=hex.toLowerCase();
  svCursorSet(hex);
  if(updatePresets!==false){
    document.querySelectorAll('input[name=set-accent-presets]').forEach(x=>{
      x.checked=(x.value.toLowerCase()===hex.toLowerCase());
    });
  }
}
// read the currently staged hex from the custom picker (validates)
function pickerGetHex(){
  const h=$('set-accent-hex').value.trim().toLowerCase();
  if(/^#[0-9a-f]{6}$/.test(h))return h;
  if(/^[0-9a-f]{6}$/.test(h))return '#'+h;
  return null;
}
// reflect current accent onto modal controls (preset radio or custom picker)
function accentControlsSync(hex){
  const h=normAccent(hex);
  if(!h)return;
  pickerSetHex(h,true);
}
mqDark.addEventListener('change',()=>{if(curTheme==='system')applyTheme('system');});
function themeRadio(val){const r=document.querySelector('input[name=set-theme][value="'+val+'"]');if(r)r.checked=true;}

/* ---- settings modal ---- */
const modal=$('settings-modal');
function populateLangSelect(){
  const sel=$('set-lang');
  if(!sel)return;
  sel.innerHTML='';
  Object.keys(I18N).forEach(code=>{
    const o=document.createElement('option');
    o.value=code;o.textContent=LANG_NAMES[code]||code.toUpperCase();
    sel.appendChild(o);
  });
}
function langSelect(val){const sel=$('set-lang');if(sel)sel.value=val;}
function openSettings(){
  $('set-interval').value=(last&&last.interval)?last.interval:curInterval;
  themeRadio((last&&last.theme)?last.theme:curTheme);
  langSelect(last&&last.lang?last.lang:curLang);
  accentControlsSync((last&&last.accent)?last.accent:curAccent);
  $('set-msg').textContent='';$('set-msg').className='modal-msg';
  modal.hidden=false;setTimeout(()=>$('set-interval').focus(),30);
}
function closeSettings(){modal.hidden=true;}
$('settings-btn').onclick=e=>{e.preventDefault();openSettings();};
$('set-cancel').onclick=closeSettings;
modal.onclick=e=>{if(e.target===modal)closeSettings();};
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!modal.hidden)closeSettings();});
/* ---- apply-on-change persistence: every control saves immediately, no Save button ---- */
async function saveField(patch){
  const msg=$('set-msg');
  try{
    const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    const j=await r.json();
    if(j.ok){ msg.textContent=''; msg.className='modal-msg'; }
    else { msg.textContent=t('msg_error_prefix')+' '+(j.error||r.status); msg.className='modal-msg err'; }
  }catch(e){ msg.textContent=t('msg_net_prefix')+' '+e.message; msg.className='modal-msg err'; }
  if(msg.className.indexOf('err')>-1){ clearTimeout(saveField._t); saveField._t=setTimeout(()=>{msg.textContent='';msg.className='modal-msg';},1500); }
}
// interval: fires on blur/enter; validates 1..300, reverts + warns on bad value
$('set-interval').addEventListener('change',()=>{
  const inp=$('set-interval');
  const v=parseFloat(inp.value);
  if(isNaN(v)||v<1||v>300){ inp.value=curInterval; flashMsg(t('msg_interval_invalid'),'err'); return; }
  curInterval=v; $('iv').textContent=v;
  saveField({interval:v});
});
function flashMsg(text,kind){
  const msg=$('set-msg');
  msg.textContent=text;msg.className='modal-msg '+(kind||'');
  clearTimeout(saveField._t);saveField._t=setTimeout(()=>{msg.textContent='';msg.className='modal-msg';},1500);
}
// theme radio: apply to page live + persist
document.querySelectorAll('input[name=set-theme]').forEach(r=>{
  r.addEventListener('change',()=>{
    applyTheme(r.value); updateThemeIcon();
    saveField({theme:r.value});
  });
});
// language select: apply to page live + persist
$('set-lang').addEventListener('change',()=>{
  applyLang($('set-lang').value);
  saveField({lang:$('set-lang').value});
});
// accent preset swatch: fill picker + rainbow center, apply to page live + persist
document.querySelectorAll('input[name=set-accent-presets]').forEach(r=>{
  r.addEventListener('change',()=>{
    pickerSetHex(r.value,false); rcCenterSet(r.value); applyAccent(r.value);
    saveField({accent:r.value});
  });
});
// update the rainbow button's center disc + active-cust outline (when hex is not a preset)
function rcCenterSet(hex){
  const el=$('sw-rc-center');
  if(el)el.style.background=hex;
  const btn=$('set-accent-open');
  if(btn)btn.classList.toggle('active-cust',!PRESET_HEXES.includes(hex));
}
// rainbow button opens the custom-picker popup, synced to the current accent
function openCpPop(){
  rcCenterSet(curAccent);
  pickerSetHex(curAccent,false);
  $('cp-pop').hidden=false;
  setTimeout(()=>$('set-accent-hex').select(),30);
}
function closeCpPop(){ $('cp-pop').hidden=true; }
$('set-accent-open').onclick=openCpPop;
$('cp-close').onclick=closeCpPop;
$('cp-pop').onclick=e=>{ if(e.target===$('cp-pop'))closeCpPop(); };
document.addEventListener('keydown',e=>{ if(e.key==='Escape'&&!$('cp-pop').hidden)closeCpPop(); });
let svHue=0;
// position the SV cursor for a given hex (normalizes hue channel + places dot)
function svCursorSet(hex){
  const rgb=hexToRgb(hex);
  if(!rgb)return;
  const [r,g,b]=rgb.map(x=>x/255);
  const max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min;
  let h=0,s=0,v=max;
  if(max!==0)s=d/max;
  if(d!==0){
    if(max===r)h=((g-b)/d)%6;
    else if(max===g)h=(b-r)/d+2;
    else h=(r-g)/d+4;
    h*=60;if(h<0)h+=360;
  }
  svHue=h;
  $('sv-canvas').style.setProperty('--sv-hue',hsvToHex(h,1,1));
  const sat=s*100,val=(1-v)*100;
  const cur=$('sv-cursor');
  if(cur){cur.style.left=sat+'%';cur.style.top=val+'%';}
}
function hsvToHex(h,s,v){
  h=((h%360)+360)%360;
  const c=v*s,x=c*(1-Math.abs((h/60)%2-1)),m=v-c;
  let r=0,g=0,b=0;
  if(h<60){r=c;g=x;}else if(h<120){r=x;g=c;}
  else if(h<180){g=c;b=x;}else if(h<240){g=x;b=c;}
  else if(h<300){r=x;b=c;}else{r=c;b=x;}
  const to255=t=>Math.round((t+m)*255).toString(16).padStart(2,'0');
  return '#'+to255(r)+to255(g)+to255(b);
}
// pointer events on SV canvas: compute S/V from coords, update sliders/hex/swatch
function bindSvCanvas(){
  const cv=$('sv-canvas');
  const update=e=>{
    const rect=cv.getBoundingClientRect();
    const x=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width));
    const y=Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height));
    const s=x,v=1-y;
    const hex=hsvToHex(svHue,s,v);
    const rgb=hexToRgb(hex);
    $('cp-r').value=rgb[0];$('cp-g').value=rgb[1];$('cp-b').value=rgb[2];
    $('set-accent-hex').value=hex;
    $('cp-swatch').style.background=hex;
    const cur=$('sv-cursor');
    if(cur){cur.style.left=(s*100)+'%';cur.style.top=((1-v)*100)+'%';}
    document.querySelectorAll('input[name=set-accent-presets]').forEach(x=>{x.checked=(x.value.toLowerCase()===hex);});
  };
  cv.onpointerdown=e=>{if(cv.setPointerCapture)cv.setPointerCapture(e.pointerId);update(e);
    const mv=ev=>update(ev),up=()=>{cv.removeEventListener('pointermove',mv);cv.removeEventListener('pointerup',up);};
    cv.addEventListener('pointermove',mv);cv.addEventListener('pointerup',up);};
}
bindSvCanvas();
// R/G/B sliders: update hex field + swatch live (no apply to page yet)
function slidersChanged(){
  const r=+$('cp-r').value,g=+$('cp-g').value,b=+$('cp-b').value;
  const hex=rgbToHex(r,g,b);
  $('set-accent-hex').value=hex;
  $('cp-swatch').style.background=hex;
  document.querySelectorAll('input[name=set-accent-presets]').forEach(x=>{
    x.checked=(x.value.toLowerCase()===hex);
  });
  svCursorSet(hex);
}
['cp-r','cp-g','cp-b'].forEach(id=>{ $(id).oninput=slidersChanged; });
// hex field: validate + sync sliders/swatch on blur
$('set-accent-hex').onblur=function(){
  const h=this.value.trim().toLowerCase();
  let norm=h;
  if(/^#[0-9a-f]{6}$/.test(norm)){}
  else if(/^[0-9a-f]{6}$/.test(norm))norm='#'+norm;
  else{ this.value=curAccent; return; }
  const rgb=hexToRgb(norm);
  if(rgb){ $('cp-r').value=rgb[0];$('cp-g').value=rgb[1];$('cp-b').value=rgb[2];
           $('cp-swatch').style.background=norm;
           document.querySelectorAll('input[name=set-accent-presets]').forEach(x=>{x.checked=(x.value.toLowerCase()===norm);}); }
};
// OK: apply the staged hex to the page immediately + persist, then close the popup
$('set-accent-ok').onclick=function(){
  const hex=pickerGetHex();
  if(!hex)return;
  pickerSetHex(hex,false);
  rcCenterSet(hex);
  applyAccent(hex);
  saveField({accent:hex});
  closeCpPop();
};
// populate the language dropdown from available I18N locales
populateLangSelect();
// apply persisted theme+accent+lang ASAP (before first paint of data)
fetch('/api/status').then(r=>r.json()).then(d=>{applyAppearance(d.theme,d.accent);if(d.lang)curLang=d.lang;applyLang(d.lang);langSelect(d.lang||'en');}).catch(()=>{});
loop();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    sampler: Sampler = None  # type: ignore
    poll_interval: float = 2.0
    vllm_url: str = ''
    api_key: Optional[str] = None
    interval_cv: threading.Condition = None  # type: ignore

    def log_message(self, fmt, *args):  # quiet
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            body = PAGE_HTML.replace('__INTERVAL__', str(int(self.poll_interval)))
            data = body.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/api/status':
            snap = self.sampler.current()
            settings = load_settings()
            payload = {'online': bool(snap), 'ts': time.time(), 'url': self.vllm_url,
                       'interval': self.poll_interval,
                       'theme': settings.get('theme', 'system'),
                       'accent': settings.get('accent', '#3fb950'),
                        'lang': settings.get('lang', 'en')}
            if snap:
                payload.update(snap)
            else:
                payload['error'] = 'waiting for first sample'
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/api/settings':
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length) if length else b'{}'
            data = json.loads(body.decode('utf-8'))
            if not any(k in data for k in ('interval', 'theme', 'lang', 'accent')):
                raise ValueError('nothing to update')
            result = {}
            if 'interval' in data:
                iv = float(data['interval'])
                if not (1.0 <= iv <= 300.0):
                    raise ValueError('interval must be 1..300 seconds')
                save_setting('interval', iv)
                with Handler.interval_cv:
                    Handler.poll_interval = iv
                    Handler.interval_cv.notify_all()
                result['interval'] = iv
            if 'theme' in data:
                th = data['theme']
                if th not in ('system', 'dark', 'light'):
                    raise ValueError('theme must be system, dark or light')
                save_setting('theme', th)
                result['theme'] = th
            if 'lang' in data:
                lg = data['lang']
                if lg not in ('ru', 'en'):
                    raise ValueError('lang must be ru or en')
                save_setting('lang', lg)
                result['lang'] = lg
            if 'accent' in data:
                ac = str(data['accent']).lower()
                if not re.fullmatch(r'#?[0-9a-f]{6}', ac):
                    raise ValueError('accent must be a hex color (#rrggbb)')
                if not ac.startswith('#'):
                    ac = '#' + ac
                save_setting('accent', ac)
                result['accent'] = ac
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, **result}).encode())
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())


def poll_loop(server: Handler, stop: threading.Event):
    url = server.vllm_url.rstrip('/') + '/metrics'
    headers = {'Authorization': f'Bearer {server.api_key}'} if server.api_key else {}
    cv = Handler.interval_cv
    while not stop.is_set():
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                text = resp.read().decode('utf-8', 'replace')
            server.sampler.ingest(parse_prometheus(text), time.time())
        except Exception as e:
            pass
        with cv:
            deadline = time.monotonic() + server.poll_interval
            while not stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                cv.wait(min(remaining, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vllm-url', default='http://localhost:8000')
    ap.add_argument('--api-key', default=None)
    ap.add_argument('--port', type=int, default=8501)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--interval', type=float, default=None,
                    help='Poll interval in seconds. Overrides config.json when set.')
    args = ap.parse_args()

    # Resolution order: CLI flag > config.json > built-in default (2s).
    cfg_interval = load_settings().get('interval')
    effective = args.interval if args.interval is not None else (cfg_interval or 2.0)

    Handler.interval_cv = threading.Condition()
    Handler.vllm_url = args.vllm_url
    Handler.api_key = args.api_key
    Handler.poll_interval = effective
    Handler.sampler = Sampler()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    stop = threading.Event()
    t = threading.Thread(target=poll_loop, args=(Handler, stop), daemon=True)
    t.start()
    src = 'cli' if args.interval is not None else ('config' if cfg_interval else 'default')
    print(f'[vllm-web] listening on http://{args.host}:{args.port}')
    print(f'[vllm-web] polling {args.vllm_url}/metrics every {effective}s (source: {src})')
    try:
        srv.serve_forever()
    finally:
        stop.set()


if __name__ == '__main__':
    main()
