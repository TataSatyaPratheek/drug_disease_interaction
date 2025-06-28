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
FAILED_TESTS=() # Initialize as an array

# Print header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  Drug-Disease Interaction Project - Test Harness        ${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo

# Create necessary directories
echo -e "${YELLOW}Setting up test environment...${NC}"
mkdir -p tests/test_output
# mkdir -p tests/test_data # This should already exist or be managed by fixtures/git

# Function to run a test and report results
run_test() {
    TEST_TARGET=$1 # Can be file or file::class
    TEST_NAME=$2

    echo -e "${YELLOW}Running test: ${TEST_NAME}${NC}"
    # Use pytest's ability to target specific files/classes/functions
    if pytest "$TEST_TARGET" -v; then
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

# DrugBank tests (Target specific classes within the file)
run_test "tests/unit/data/sources/drugbank/test_vocabulary.py::TestDrugBankVocabulary" "DrugBank Vocabulary Parser"
run_test "tests/unit/data/sources/drugbank/test_vocabulary.py::TestDrugBankXMLParser" "DrugBank XML Parser"
run_test "tests/unit/data/sources/drugbank/test_vocabulary.py::TestDrugBankIntegrator" "DrugBank Integration"

# MeSH tests
run_test "tests/unit/data/sources/mesh/test_mesh_parser.py" "MeSH Parser" # Use renamed file

# OpenTargets tests
run_test "tests/unit/data/sources/opentargets/test_opentargets_parser.py" "OpenTargets Parser" # Use renamed file

# Graph Builder tests
run_test "tests/unit/graph/test_builder.py" "Graph Builder"

# Graph Analysis tests
run_test "tests/unit/analysis/test_graph_analysis.py" "Graph Analysis"

# Graph Visualization tests
run_test "tests/unit/visualization/test_graph_viz.py" "Graph Visualization"

# Check if PyTorch Geometric is available for graph conversion tests
if python -c "import torch_geometric" &> /dev/null; then
    run_test "tests/unit/graph/test_conversion.py" "Graph Conversion"
else
    echo -e "${YELLOW}Skipping Graph Conversion tests (PyTorch Geometric not installed)${NC}"
    SKIPPED=$((SKIPPED + 1))
fi

# Run integration tests
echo -e "${BLUE}Running Integration Tests${NC}"
echo -e "${BLUE}------------------------${NC}"

# Target specific classes within the integration test file
run_test "tests/integration/test_end_to_end.py::TestDataFlow" "Data Flow Tests"
run_test "tests/integration/test_end_to_end.py::TestEndToEnd" "End-to-End Tests"


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
