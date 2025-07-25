#!/usr/bin/env python3
"""
Test runner for evaluation pipeline tests.
This script runs all evaluation-related tests and provides clear output.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests(test_pattern=None, verbose=False):
    """
    Run evaluation tests with specified pattern.
    
    Args:
        test_pattern: Pattern to match test files (e.g., "test_evaluation_runner")
        verbose: Whether to run tests in verbose mode
    """
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if test_pattern:
        cmd.extend(["-k", test_pattern])
    
    if verbose:
        cmd.append("-v")
    
    # Add test discovery paths
    cmd.extend([
        "tests/core/evaluation/",
        "--tb=short",
        "--disable-warnings"
    ])
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        return False

def run_specific_test_suite(suite_name):
    """Run a specific test suite with helpful output."""
    
    test_suites = {
        "unit": "test_evaluation_runner",
        "integration": "test_integration", 
        "script": "test_evaluate_script",
        "all": None
    }
    
    if suite_name not in test_suites:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {list(test_suites.keys())}")
        return False
    
    pattern = test_suites[suite_name]
    print(f"Running {suite_name} test suite...")
    return run_tests(pattern, verbose=True)

def main():
    """Main function to run tests based on command line arguments."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_evaluation_tests.py <test_suite>")
        print("\nAvailable test suites:")
        print("  unit       - Unit tests for EvaluationRunner")
        print("  integration - Integration tests with real configs")
        print("  script     - Tests for the main evaluate.py script")
        print("  all        - Run all evaluation tests")
        print("\nExamples:")
        print("  python run_evaluation_tests.py unit")
        print("  python run_evaluation_tests.py integration")
        print("  python run_evaluation_tests.py all")
        return
    
    suite_name = sys.argv[1].lower()
    
    if suite_name == "all":
        print("Running all evaluation tests...")
        success = run_tests(verbose=True)
    else:
        success = run_specific_test_suite(suite_name)
    
    if success:
        print("\n🎉 Test execution completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test execution failed!")
        sys.exit(1)