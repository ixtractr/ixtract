#!/bin/bash
# ─────────────────────────────────────────────────────────────
# ixtract test runner — v0.2.0
#
# Usage:
#   ./run_tests.sh              # interactive (output to terminal)
#   ./run_tests.sh &            # background (output to test_results.log)
#   tail -f test_results.log    # follow live while running in background
# ─────────────────────────────────────────────────────────────

set -euo pipefail

LOG="test_results.log"
PASS=0
FAIL=0
START=$(date +%s)

# If running in background (no tty), redirect everything to log file
if [ ! -t 1 ]; then
    exec > "$LOG" 2>&1
fi

log() { echo "$1" | tee -a "$LOG" 2>/dev/null || echo "$1"; }

> "$LOG"
log "═══════════════════════════════════════════════════════"
log "  ixtract test suite — $(date '+%Y-%m-%d %H:%M:%S')"
log "═══════════════════════════════════════════════════════"

run_suite() {
    local label="$1"
    local module="$2"
    log ""
    log "▶ $label"
    log "─────────────────────────────────────────────────────"
    if python -m unittest "$module" -v >> "$LOG" 2>&1; then
        log "  ✅ PASSED"
        PASS=$((PASS + 1))
    else
        log "  ❌ FAILED — see $LOG for details"
        FAIL=$((FAIL + 1))
    fi
}

# ── Simulation suites (no database required) ──────────────────
run_suite "Phase 0 simulation (39 tests)"   "tests.simulation.test_phase0"
run_suite "Hardening simulation (17 tests)" "tests.simulation.test_hardening"

# ── Integration suite (requires PostgreSQL) ───────────────────
log ""
log "▶ Checking test database..."
if python -c "
from tests.integration import db_available
import sys
sys.exit(0 if db_available() else 1)
" 2>/dev/null; then
    log "  ✓ Database available"
    run_suite "Integration — PostgreSQL (12 tests)" "tests.integration.test_pipeline"
else
    log "  ✗ Database not available — skipping integration tests"
    log ""
    log "  To run integration tests:"
    log "    1. docker compose -f docker-compose.test.yml up -d"
    log "    2. python tests/integration/seed_db.py"
    log "    3. ./run_tests.sh"
fi

# ── Summary ───────────────────────────────────────────────────
END=$(date +%s)
ELAPSED=$((END - START))
log ""
log "═══════════════════════════════════════════════════════"
log "  Results: $PASS suite(s) passed, $FAIL suite(s) failed"
log "  Elapsed: ${ELAPSED}s"
log "═══════════════════════════════════════════════════════"

if [ "$FAIL" -eq 0 ]; then
    log ""
    log "  ✅ All suites passed. Ready to commit v0.2.0."
    log ""
    log "  Next steps:"
    log "    git add -A"
    log "    git commit -m 'feat: Phase 2A complete + escape mode + battle-hardening'"
    log "    git tag -a v0.2.0 -m 'Phase 2A: Statistical controller + visibility'"
    log "    git push && git push --tags"
    exit 0
else
    log ""
    log "  ❌ $FAIL suite(s) failed. Do not commit until resolved."
    exit 1
fi
