"""HTTP server: page + /api/status + /api/settings, vLLM /metrics poll loop."""

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Dict, Optional

from .config import load_settings, save_setting
from .i18n import I18N_JS
from .page import PAGE_HTML
from .prometheus import parse_prometheus
from .sampler import Sampler


class Handler(BaseHTTPRequestHandler):
    sampler: Sampler = None  # type: ignore
    poll_interval: float = 2.0
    vllm_url: str = ''
    api_key: Optional[str] = None
    interval_cv: threading.Condition = None  # type: ignore
    paused: bool = False

    def log_message(self, fmt, *args):  # quiet
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            body = PAGE_HTML.replace('__INTERVAL__', str(int(self.poll_interval)))
            body = body.replace('__I18N__', I18N_JS)
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
                        'lang': settings.get('lang', 'en'),
                        'paused': Handler.paused}
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
            if not any(k in data for k in ('interval', 'theme', 'lang', 'accent', 'paused')):
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
            if 'paused' in data:
                Handler.paused = bool(data['paused'])
                with Handler.interval_cv:
                    Handler.interval_cv.notify_all()
                result['paused'] = Handler.paused
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
        if Handler.paused:
            with cv:
                cv.wait(0.5)
            continue
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

