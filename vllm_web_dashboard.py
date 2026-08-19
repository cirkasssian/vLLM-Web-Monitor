#!/usr/bin/env python3
"""Entrypoint shim for the vllm-web dashboard.

The implementation lives in the ``vllm_web/`` package next to this file.
This shim keeps the historical launch command working:
    python3 vllm_web_dashboard.py ...
Equivalent invocation:
    python3 -m vllm_web ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm_web.main import main

main()
