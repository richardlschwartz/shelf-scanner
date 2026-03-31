#!/bin/bash
# Shelf Out-of-Stock Scanner - Startup Script
#
# Prerequisites:
#   pip install flask anthropic Pillow
#
# Usage:
#   export ANTHROPIC_API_KEY=your-key-here
#   ./run.sh
#
# Then open http://localhost:5000 in your browser.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "ERROR: ANTHROPIC_API_KEY environment variable is not set."
    echo ""
    echo "  export ANTHROPIC_API_KEY=your-key-here"
    echo "  ./run.sh"
    echo ""
    exit 1
fi

# Install dependencies if needed
pip install -q flask anthropic Pillow 2>/dev/null || pip install -q --break-system-packages flask anthropic Pillow

echo ""
echo "  Shelf Out-of-Stock Scanner"
echo "  =========================="
echo "  Open http://localhost:5001 in your browser"
echo ""

python3 app.py
