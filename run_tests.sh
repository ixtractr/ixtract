#!/bin/bash
# ─────────────────────────────────────────────────────────────
# ixtract test runner
# ─────────────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════════════════"
echo "  ixtract test suite"
echo "═══════════════════════════════════════════════════════"

# Phase 0: simulation tests (no database needed)
echo ""
echo "▶ Phase 0: Simulation tests (no database required)"
echo "─────────────────────────────────────────────────────"
python -m unittest tests.simulation.test_phase0 -v
echo ""

# Check if database is available
echo "▶ Checking test database..."
if python -c "
from tests.integration import db_available
import sys
sys.exit(0 if db_available() else 1)
" 2>/dev/null; then
    echo "  ✓ Database available"
    echo ""
    echo "▶ Phase 1: Integration tests (against PostgreSQL)"
    echo "─────────────────────────────────────────────────────"
    python -m unittest tests.integration.test_pipeline -v
else
    echo "  ✗ Database not available"
    echo ""
    echo "  To run integration tests:"
    echo "    1. docker compose -f docker-compose.test.yml up -d"
    echo "    2. pip install -e '.'"
    echo "    3. python tests/integration/seed_db.py"
    echo "    4. ./run_tests.sh"
    echo ""
    echo "  Skipping integration tests."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done."
echo "═══════════════════════════════════════════════════════"
