#!/usr/bin/env bash
# ==============================================================================
# tests/run_e2e_tests.sh
# ==============================================================================
# Test runner for End-to-End (E2E) verification of the SBC Scientific Article
#
# Usage:
#   ./tests/run_e2e_tests.sh [OPTIONS]
#
# Options:
#   -t, --tier <1|2|3|4>    Run only a specific test tier (default: all tiers)
#   -v, --verbose           Enable verbose test output (default: verbose on)
#   -q, --quiet             Minimal output
#   -h, --help              Show this help message
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve Python interpreter
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: Python interpreter not found." >&2
    exit 1
fi

# Tectonic binary detection
if command -v tectonic &>/dev/null; then
    TECTONIC_STATUS="$(tectonic --version 2>/dev/null | head -n1)"
elif [[ -x "/home/luizg/.local/bin/tectonic" ]]; then
    TECTONIC_STATUS="$(/home/luizg/.local/bin/tectonic --version 2>/dev/null | head -n1)"
else
    TECTONIC_STATUS="NOT INSTALLED"
fi

# Colors
BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
NC="\033[0m"

# Parse CLI arguments
TARGET_TIER=""
VERBOSITY="-v"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tier)
            TARGET_TIER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSITY="-v"
            shift
            ;;
        -q|--quiet)
            VERBOSITY=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-t tier] [-v|-q]"
            echo "  -t 1   Tier 1: Feature Coverage (F1-F14)"
            echo "  -t 2   Tier 2: Boundary & Corner Cases"
            echo "  -t 3   Tier 3: Cross-Feature Integration"
            echo "  -t 4   Tier 4: Real-World Workload Scenarios"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo -e "${CYAN}${BOLD}====================================================================${NC}"
echo -e "${CYAN}${BOLD}  SBC Scientific Article E2E Test Suite Runner                     ${NC}"
echo -e "${CYAN}${BOLD}====================================================================${NC}"
echo -e " Project Root  : ${PROJECT_ROOT}"
echo -e " Python Engine : $(${PYTHON_BIN} --version) at ${PYTHON_BIN}"
echo -e " Tectonic      : ${TECTONIC_STATUS}"
if [[ -n "${TARGET_TIER}" ]]; then
    echo -e " Target Tier   : Tier ${TARGET_TIER}"
else
    echo -e " Target Tier   : All Tiers (1 to 4)"
fi
echo -e "${CYAN}--------------------------------------------------------------------${NC}"

# Determine test target pattern
TEST_MODULE="tests.test_e2e_article"
TARGET_ARG=""

case "${TARGET_TIER}" in
    1)
        TARGET_ARG="${TEST_MODULE}.TestTier1FeatureCoverage"
        ;;
    2)
        TARGET_ARG="${TEST_MODULE}.TestTier2BoundaryAndCornerCases"
        ;;
    3)
        TARGET_ARG="${TEST_MODULE}.TestTier3CrossFeatureIntegration"
        ;;
    4)
        TARGET_ARG="${TEST_MODULE}.TestTier4RealWorldWorkloads"
        ;;
    "")
        TARGET_ARG="${TEST_MODULE}"
        ;;
    *)
        echo -e "${RED}ERROR: Invalid tier '${TARGET_TIER}'. Must be 1, 2, 3, or 4.${NC}" >&2
        exit 1
        ;;
esac

cd "${PROJECT_ROOT}"

# Run unittests
set +e
"${PYTHON_BIN}" -m unittest ${VERBOSITY} ${TARGET_ARG}
EXIT_CODE=$?
set -e

echo -e "${CYAN}--------------------------------------------------------------------${NC}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✓ ALL E2E TESTS PASSED SUCCESSFULLY!${NC}"
else
    echo -e "${YELLOW}${BOLD}⚠ TEST SUITE EXECUTION COMPLETED WITH FAILURES (Exit code: ${EXIT_CODE})${NC}"
    echo -e "${YELLOW}Note: Failures for ungenerated files (e.g. artigo.pdf, figures) are expected"
    echo -e "until Milestones M1, M2, and M3 produce their respective deliverables.${NC}"
fi
echo -e "${CYAN}${BOLD}====================================================================${NC}"

exit ${EXIT_CODE}
