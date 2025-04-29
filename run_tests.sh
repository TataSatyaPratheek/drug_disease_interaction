#!/bin/bash
# run_tests.sh - Test harness script for Drug-Disease Interaction project

# Set up colored output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Initialize counters
PASSED=0
FAILED=0
SKIPPED=0

# Print header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  Drug-Disease Interaction Project - Test Harness        ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo

# Create necessary directories
echo -e "${YELLOW}Setting up test environment...${NC}"
mkdir -p tests/test_output
mkdir -p tests/test_data

# Function to run a test and report results
run_test() {
    TEST_FILE=$1
    TEST_NAME=$2
    
    echo -e "${YELLOW}Running test: ${TEST_NAME}${NC}"
    if pytest $TEST_FILE -v; then
        echo -e "${GREEN}✓ ${TEST_NAME} passed${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ ${TEST_NAME} failed${NC}"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$TEST_NAME")
    fi
    echo
}

# Run unit tests for each component
echo -e "${BLUE}Running Unit Tests${NC}"
echo -e "${BLUE}-----------------${NC}"

# DrugBank tests
run_test "tests/unit/data/sources/drugbank/test_vocabulary.py" "DrugBank Vocabulary Parser"
run_test "tests/unit/data/sources/drugbank/test_xml_parser.py" "DrugBank XML Parser"
run_test "tests/unit/data/sources/drugbank/test_integration.py" "DrugBank Integration"

# MeSH tests
run_test "tests/unit/data/sources/mesh/test_parser.py" "MeSH Parser"

# OpenTargets tests
run_test "tests/unit/data/sources/opentargets/test_parser.py" "OpenTargets Parser"

# Graph Builder tests
run_test "tests/unit/graph/test_builder.py" "Graph Builder"

# Check if DGL is available for graph conversion tests
if python -c "import dgl" &> /dev/null; then
    run_test "tests/unit/graph/test_conversion.py" "Graph Conversion"
else
    echo -e "${YELLOW}Skipping Graph Conversion tests (DGL not installed)${NC}"
    SKIPPED=$((SKIPPED + 1))
fi

# Run integration tests
echo -e "${BLUE}Running Integration Tests${NC}"
echo -e "${BLUE}------------------------${NC}"

run_test "tests/integration/test_data_flow.py" "Data Flow Tests"
run_test "tests/integration/test_end_to_end.py" "End-to-End Tests"

# Print summary
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}                Test Summary                             ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Tests passed:  ${PASSED}${NC}"
echo -e "${RED}Tests failed:  ${FAILED}${NC}"
echo -e "${YELLOW}Tests skipped: ${SKIPPED}${NC}"
echo -e "${BLUE}Total tests:   $((PASSED + FAILED + SKIPPED))${NC}"
echo

# Print failed tests if any
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "${RED}  - $test${NC}"
    done
    echo
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    echo
    exit 0
fi