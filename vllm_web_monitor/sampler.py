"""Snapshot sampling and derived-metric computation (rates, latencies, history)."""

from collections import deque
from typing import Dict, Optional

from .prometheus import pick


class Sampler:
    """Keeps the last few parsed snapshots to compute rates & recent means."""

    MIN_RATE_WINDOW = 3.0

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

        # Counter-reset guard: a smaller counter than the previous sample
        # means the vLLM process restarted; report no rate. Warm-up guard:
        # a dt below MIN_RATE_WINDOW (possible right after our own start,
        # when the first poll happens sooner than the nominal interval)
        # would inflate the rate; skip it too.
        def rate(key: str) -> Optional[float]:
            if not prev or dt <= 0 or dt < self.MIN_RATE_WINDOW:
                return None
            new, old = cur.get(key), prev.get(key)
            if new is None or old is None or new < old:
                return None
            return (new - old) / dt

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
        seg = list(self.history)[-n:]
        data['history'] = {
            'ts': [round(h['ts'], 3) for h in seg],
            'active': [round(h.get('running') or 0, 1) for h in seg],
            'gen_tok_s': [_safe_rate(self, i) for i in range(len(self.history) - n, len(self.history))],
            'kv_pct': [round((h.get('kv_usage') or 0) * 100, 1) for h in seg],
        }
        data['history']['gen_tok_s'] = [round(v, 1) if v is not None else None for v in data['history']['gen_tok_s']]
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
    if dt <= 0 or dt < sampler.MIN_RATE_WINDOW:
        return None
    if gt_b < gt_a:
        return None
    return (gt_b - gt_a) / dt
