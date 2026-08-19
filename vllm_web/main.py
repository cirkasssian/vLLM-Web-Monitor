"""Application bootstrap: argparse, config resolution, server startup."""

import argparse
import threading
from http.server import ThreadingHTTPServer

from .config import load_settings
from .sampler import Sampler
from .server import Handler, poll_loop


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
