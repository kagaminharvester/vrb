#!/bin/bash
###############################################################################
# Test Runner Script for VR Body Segmentation
#
# This script runs various test suites with different configurations.
# Usage: ./scripts/run_tests.sh [options]
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
MARKERS=""
GPU_TESTS=true
SLOW_TESTS=false
HTML_REPORT=false
OUTPUT_DIR="test_results"

# Print usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Test type: all, unit, integration, performance (default: all)"
    echo "  -c, --coverage         Enable coverage reporting"
    echo "  -v, --verbose          Verbose output"
    echo "  -m, --markers MARKERS  Run tests matching markers (e.g., 'gpu', 'slow')"
    echo "  --no-gpu               Skip GPU tests"
    echo "  --slow                 Include slow tests"
    echo "  --html                 Generate HTML report"
    echo "  -o, --output DIR       Output directory for reports (default: test_results)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -t unit -c                    # Run unit tests with coverage"
    echo "  $0 -t integration --no-gpu       # Run integration tests without GPU"
    echo "  $0 -m 'not slow' --html          # Run all tests except slow ones with HTML report"
    echo "  $0 --slow -v                     # Run all tests including slow ones, verbose"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        --no-gpu)
            GPU_TESTS=false
            shift
            ;;
        --slow)
            SLOW_TESTS=true
            shift
            ;;
        --html)
            HTML_REPORT=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════════"
echo "  VR Body Segmentation - Test Runner"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install pytest pytest-cov pytest-html"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build pytest command
PYTEST_CMD="pytest"

# Add test path based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit"
        echo -e "${YELLOW}Running unit tests...${NC}"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration"
        echo -e "${YELLOW}Running integration tests...${NC}"
        ;;
    performance)
        PYTEST_CMD="$PYTEST_CMD tests/integration/test_performance.py"
        echo -e "${YELLOW}Running performance tests...${NC}"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests"
        echo -e "${YELLOW}Running all tests...${NC}"
        ;;
    *)
        echo -e "${RED}Error: Invalid test type '$TEST_TYPE'${NC}"
        exit 1
        ;;
esac

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add markers
if [ ! -z "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m '$MARKERS'"
elif [ "$GPU_TESTS" = false ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not gpu'"
fi

# Handle slow tests
if [ "$SLOW_TESTS" = false ]; then
    if [ ! -z "$MARKERS" ]; then
        PYTEST_CMD="$PYTEST_CMD and not slow"
    else
        PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
    fi
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=term --cov-report=html:$OUTPUT_DIR/coverage"
    echo -e "${YELLOW}Coverage reporting enabled${NC}"
fi

# Add HTML report
if [ "$HTML_REPORT" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --html=$OUTPUT_DIR/report.html --self-contained-html"
    echo -e "${YELLOW}HTML report will be generated${NC}"
fi

# Add JUnit XML for CI/CD
PYTEST_CMD="$PYTEST_CMD --junitxml=$OUTPUT_DIR/junit.xml"

# Print configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Test Type:    $TEST_TYPE"
echo "  Coverage:     $COVERAGE"
echo "  GPU Tests:    $GPU_TESTS"
echo "  Slow Tests:   $SLOW_TESTS"
echo "  Verbose:      $VERBOSE"
echo "  Output Dir:   $OUTPUT_DIR"
if [ ! -z "$MARKERS" ]; then
    echo "  Markers:      $MARKERS"
fi
echo ""

# Print system info
echo -e "${BLUE}System Information:${NC}"
echo "  Python:       $(python --version 2>&1)"
echo "  Pytest:       $(pytest --version 2>&1 | head -n 1)"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "  CUDA:         Available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
else
    echo "  CUDA:         Not Available"
fi

echo ""

# Run tests
echo -e "${BLUE}Running tests...${NC}"
echo "Command: $PYTEST_CMD"
echo ""

# Execute pytest
if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ Tests passed successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

    # Print report locations
    if [ "$COVERAGE" = true ]; then
        echo -e "${BLUE}Coverage report: $OUTPUT_DIR/coverage/index.html${NC}"
    fi

    if [ "$HTML_REPORT" = true ]; then
        echo -e "${BLUE}Test report: $OUTPUT_DIR/report.html${NC}"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ✗ Tests failed!${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    exit 1
fi
