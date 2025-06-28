#!/usr/bin/env python
"""
Quick test to verify pytest setup is working correctly.
Run this to ensure the test infrastructure is properly configured.
"""
import subprocess
import sys
from pathlib import Path


def test_pytest_installation():
    """Test that pytest is installed."""
    try:
        import pytest
        print("✅ pytest is installed")
        print(f"   Version: {pytest.__version__}")
        return True
    except ImportError:
        print("❌ pytest is not installed")
        print("   Run: pip install pytest pytest-cov pytest-mock")
        return False


def test_project_structure():
    """Test that project structure is correct."""
    expected_dirs = [
        "tests",
        "tests/unit", 
        "tests/integration",
        "tests/performance",
        "src",
        "configs"
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} is missing")
            all_exist = False
    
    return all_exist


def test_config_files():
    """Test that test configuration files exist."""
    config_files = [
        "pytest.ini",
        "conftest.py",
        "tests/__init__.py"
    ]
    
    all_exist = True
    for file_path in config_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} is missing")
            all_exist = False
    
    return all_exist


def run_simple_test():
    """Run a simple test to verify pytest works."""
    print("\n🧪 Running simple test...")
    
    # Create a temporary test file
    test_content = '''
def test_simple():
    """Simple test to verify pytest works."""
    assert 1 + 1 == 2
    assert True is True
'''
    
    test_file = Path("test_simple_verify.py")
    test_file.write_text(test_content)
    
    try:
        # Run the test
        result = subprocess.run(
            ["pytest", str(test_file), "-v"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Simple test passed")
            return True
        else:
            print("❌ Simple test failed")
            print(result.stdout)
            print(result.stderr)
            return False
    finally:
        # Clean up
        test_file.unlink()


def test_imports():
    """Test that key modules can be imported."""
    modules_to_test = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("pytest", "pytest"),
        ("pytest_cov", "pytest-cov"),
    ]
    
    all_imported = True
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name} can be imported")
        except ImportError:
            print(f"❌ {display_name} cannot be imported")
            all_imported = False
    
    return all_imported


def main():
    """Run all verification tests."""
    print("🔍 Verifying PneumoniaCNN test setup...\n")
    
    tests = [
        ("pytest installation", test_pytest_installation),
        ("Project structure", test_project_structure),
        ("Config files", test_config_files),
        ("Module imports", test_imports),
        ("Simple test execution", run_simple_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        results.append(test_func())
    
    # Summary
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} checks passed! Test infrastructure is ready.")
        print("\n🚀 You can now run tests with:")
        print("   pytest                    # Run all tests")
        print("   python run_tests.py unit  # Run unit tests")
        print("   python run_tests.py coverage  # Run with coverage")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} checks failed.")
        print("\n⚠️  Please fix the issues above before running tests.")
        return 1


if __name__ == "__main__":
    sys.exit(main())