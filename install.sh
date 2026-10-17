#!/bin/sh
# Install the ur command, then the ur skill for the agents on this machine.
set -eu

here=$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)
if [ -n "$here" ] && [ -f "$here/pyproject.toml" ] && grep -q '^name = "ur"' "$here/pyproject.toml"; then
    src="$here"
else
    src="${UR_SOURCE:-git+https://github.com/abiome-org/ur}"
fi

if command -v uv >/dev/null 2>&1; then
    uv tool install --force --reinstall "$src"
elif command -v pipx >/dev/null 2>&1; then
    pipx install --force "$src"
else
    python3 -m pip install --user --upgrade "$src"
fi

if ! command -v ur >/dev/null 2>&1; then
    echo "ur is installed but not on PATH yet. Add the tool directory to PATH (with uv: uv tool update-shell), then run: ur install" >&2
    exit 1
fi
ur install
