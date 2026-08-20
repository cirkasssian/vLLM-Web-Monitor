#!/usr/bin/env python3
"""Entrypoint shim for vllm-web-monitor.

The implementation lives in the ``vllm_web_monitor/`` package next to this file.
This shim keeps the historical launch command working:
    python3 vllm_web_monitor.py ...
Equivalent invocation:
    python3 -m vllm_web_monitor ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_web_monitor.main import main

main()
