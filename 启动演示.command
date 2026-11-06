#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import numpy' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements-offline.txt
fi
print '启动后点击终端显示的本地网址即可。停止时按 Control+C。'
.venv/bin/python retrieval.py demo
