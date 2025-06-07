import os
import subprocess
import difflib
from pathlib import Path

def run_rpal_exe(file_path):
    """Run the test file using rpal.exe"""
    try:
        result = subprocess.run(['rpal-wine/rpal.exe', str(file_path)], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error running rpal.exe: {e.stderr}"

def run_myrpal(file_path):
    """Run the test file using myrpal.py"""
    try:
        result = subprocess.run(['python', 'myrpal.py', str(file_path)], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error running myrpal.py: {e.stderr}"

def compare_outputs(original, mine, filename):
    """Compare outputs and return True if they match, False otherwise"""
    if original == mine:
        return True, ""
    
    # Generate a diff if outputs don't match
    diff = list(difflib.unified_diff(
        original.splitlines(),
        mine.splitlines(),
        fromfile='rpal.exe output',
        tofile='myrpal.py output',
        lineterm=''
    ))
    return False, '\n'.join(diff)

def run_tests():
    # Get all .rpal files from test-programs directory
    test_dir = Path('test-programs')
    test_files = sorted(test_dir.glob('*.rpal'))
    
    total_tests = len(test_files)
    passed_tests = 0
    failed_tests = []
    
    print(f"\nRunning {total_tests} test files...\n")
    print("=" * 80)
    
    for test_file in test_files:
        print(f"\nTesting: {test_file.name}")
        print("-" * 40)
        
        # Run both interpreters
        rpal_output = run_rpal_exe(test_file)
        myrpal_output = run_myrpal(test_file)
        
        # Compare outputs
        passed, diff = compare_outputs(rpal_output, myrpal_output, test_file.name)
        
        if passed:
            print("✅ PASSED")
            passed_tests += 1
        else:
            print("❌ FAILED")
            print("\nDifferences found:")
            print(diff)
            failed_tests.append(test_file.name)
        
        print("-" * 40)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"\nTest Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"- {test}")

if __name__ == "__main__":
    run_tests() 