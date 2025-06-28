#!/usr/bin/env python
"""
Test runner script for the PneumoniaCNN project.
Provides convenient commands for running different test suites.
"""
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], verbose: bool = False) -> int:
    """Run a command and return exit code."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=not verbose)
    
    if not verbose and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(result.stdout.decode() if result.stdout else "")
        print(result.stderr.decode() if result.stderr else "")
    
    return result.returncode


def run_unit_tests(verbose: bool = False, specific_test: Optional[str] = None) -> int:
    """Run unit tests."""
    cmd = ["pytest", "tests/unit", "-v" if verbose else "-q", "-m", "unit"]
    
    if specific_test:
        cmd.append(f"tests/unit/{specific_test}")
    
    return run_command(cmd, verbose)


def run_integration_tests(verbose: bool = False, specific_test: Optional[str] = None) -> int:
    """Run integration tests."""
    cmd = ["pytest", "tests/integration", "-v" if verbose else "-q", "-m", "integration"]
    
    if specific_test:
        cmd.append(f"tests/integration/{specific_test}")
    
    return run_command(cmd, verbose)


def run_performance_tests(verbose: bool = False) -> int:
    """Run performance tests."""
    cmd = ["pytest", "tests/performance", "-v" if verbose else "-q", "-m", "performance"]
    return run_command(cmd, verbose)


def run_all_tests(verbose: bool = False) -> int:
    """Run all tests."""
    cmd = ["pytest", "-v" if verbose else "-q"]
    return run_command(cmd, verbose)


def run_coverage(verbose: bool = False) -> int:
    """Run tests with coverage report."""
    cmd = [
        "pytest",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v" if verbose else "-q"
    ]
    
    exit_code = run_command(cmd, verbose)
    
    if exit_code == 0:
        print("\nCoverage report generated in htmlcov/index.html")
    
    return exit_code


def run_specific_marker(marker: str, verbose: bool = False) -> int:
    """Run tests with specific marker."""
    cmd = ["pytest", "-v" if verbose else "-q", "-m", marker]
    return run_command(cmd, verbose)


def list_tests(pattern: Optional[str] = None) -> int:
    """List all available tests."""
    cmd = ["pytest", "--collect-only", "-q"]
    
    if pattern:
        cmd.extend(["-k", pattern])
    
    return run_command(cmd, True)


def run_failed_tests(verbose: bool = False) -> int:
    """Re-run only failed tests from last run."""
    cmd = ["pytest", "--lf", "-v" if verbose else "-q"]
    return run_command(cmd, verbose)


def run_quick_tests(verbose: bool = False) -> int:
    """Run quick tests (exclude slow and gpu tests)."""
    cmd = ["pytest", "-v" if verbose else "-q", "-m", "not slow and not gpu"]
    return run_command(cmd, verbose)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for PneumoniaCNN project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py unit              # Run unit tests
  python run_tests.py integration       # Run integration tests
  python run_tests.py performance       # Run performance tests
  python run_tests.py all              # Run all tests
  python run_tests.py coverage         # Run with coverage report
  python run_tests.py quick            # Run quick tests only
  python run_tests.py failed           # Re-run failed tests
  python run_tests.py list             # List all tests
  python run_tests.py list -p config   # List tests matching pattern
  python run_tests.py unit -t test_config.py  # Run specific test file
  python run_tests.py marker gpu       # Run tests with 'gpu' marker
        """
    )
    
    parser.add_argument(
        "command",
        choices=[
            "unit", "integration", "performance", "all",
            "coverage", "quick", "failed", "list", "marker"
        ],
        help="Test suite to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-t", "--test",
        help="Specific test file to run (for unit/integration commands)"
    )
    
    parser.add_argument(
        "-p", "--pattern",
        help="Pattern to match test names (for list command)"
    )
    
    parser.add_argument(
        "-m", "--marker",
        help="Marker name (for marker command)"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    if project_root != Path.cwd():
        print(f"Changing to project root: {project_root}")
        import os
        os.chdir(project_root)
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed. Please run: pip install pytest pytest-cov")
        return 1
    
    # Run appropriate command
    exit_code = 0
    
    if args.command == "unit":
        exit_code = run_unit_tests(args.verbose, args.test)
    elif args.command == "integration":
        exit_code = run_integration_tests(args.verbose, args.test)
    elif args.command == "performance":
        exit_code = run_performance_tests(args.verbose)
    elif args.command == "all":
        exit_code = run_all_tests(args.verbose)
    elif args.command == "coverage":
        exit_code = run_coverage(args.verbose)
    elif args.command == "quick":
        exit_code = run_quick_tests(args.verbose)
    elif args.command == "failed":
        exit_code = run_failed_tests(args.verbose)
    elif args.command == "list":
        exit_code = list_tests(args.pattern)
    elif args.command == "marker":
        if not args.marker:
            print("Error: --marker is required for marker command")
            parser.print_help()
            return 1
        exit_code = run_specific_marker(args.marker, args.verbose)
    
    # Print summary
    if exit_code == 0:
        print(f"\n✅ {args.command.capitalize()} tests passed!")
    else:
        print(f"\n❌ {args.command.capitalize()} tests failed!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())