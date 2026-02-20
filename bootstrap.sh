#!/usr/bin/env bash
# ============================================================
# bootstrap.sh — One-shot setup for ParrotOS / Debian / Ubuntu
# Run: chmod +x bootstrap.sh && ./bootstrap.sh
# ============================================================
set -euo pipefail

echo "╔══════════════════════════════════════════════╗"
echo "║  Self-Improving Agent — Hackathon Bootstrap  ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. Install uv (if missing) ──────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "→ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "✓ uv $(uv --version) already installed"
fi

# ── 2. Install Node.js (if missing — needed for npx/Postman MCP) ─
if ! command -v node &>/dev/null; then
    echo "→ Installing Node.js via NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "✓ Node.js $(node --version) already installed"
fi

# ── 3. Sync Python dependencies ─────────────────────────────
echo "→ Syncing Python deps with uv..."
uv sync

# ── 4. Set up .env from template ────────────────────────────
if [ ! -f .env ]; then
    cp .env.template .env
    echo "→ Created .env from template — FILL IN YOUR API KEYS"
    echo "  Edit: nano .env"
else
    echo "✓ .env already exists"
fi

# ── 5. Verify installation ──────────────────────────────────
echo ""
echo "─── Verification ───"
echo "uv:     $(uv --version)"
echo "python: $(uv run python --version)"
echo "node:   $(node --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "npx:    $(npx --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "adk:    $(uv run python -c 'import google.adk; print(google.adk.__version__)' 2>/dev/null || echo 'IMPORT ERROR')"
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✓ Bootstrap complete!                       ║"
echo "║                                              ║"
echo "║  Next steps:                                 ║"
echo "║  1. nano .env          (add API keys)        ║"
echo "║  2. uv run adk web .   (start agent)         ║"
echo "║  3. Open http://localhost:8000                ║"
echo "╚══════════════════════════════════════════════╝"
