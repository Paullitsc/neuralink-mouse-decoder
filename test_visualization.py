#!/usr/bin/env python3
"""
test_visualization.py

Simple test script to verify the visualization functionality.
"""

import os
import sys

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        from decode_mouse_velocities import load_csv, reconstruct_positions, segment_strokes
        print("✅ decode_mouse_velocities imported successfully")
    except ImportError as e:
        print(f"❌ decode_mouse_velocities import failed: {e}")
        return False
    
    try:
        from visualize_letters import LetterVisualizer
        print("✅ LetterVisualizer imported successfully")
    except ImportError as e:
        print(f"❌ LetterVisualizer import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test that the CSV data can be loaded."""
    if not os.path.exists('mouse_velocities.csv'):
        print("❌ mouse_velocities.csv not found")
        return False
    
    try:
        from decode_mouse_velocities import load_csv, reconstruct_positions
        df = load_csv('mouse_velocities.csv')
        x, y = reconstruct_positions(df)
        print(f"✅ Data loaded successfully: {len(df)} rows, {len(x)} points")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_visualizer_creation():
    """Test that the visualizer can be created."""
    try:
        from visualize_letters import LetterVisualizer
        visualizer = LetterVisualizer()
        print("✅ LetterVisualizer created successfully")
        return True
    except Exception as e:
        print(f"❌ Visualizer creation failed: {e}")
        return False

def main():
    print("=== VISUALIZATION TEST SUITE ===\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Visualizer Creation", test_visualizer_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED\n")
        else:
            print(f"❌ {test_name} FAILED\n")
    
    print(f"=== TEST RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Visualization system is ready.")
        print("\nTo run the visualization:")
        print("python visualize_letters.py")
        print("\nOr with custom options:")
        print("python visualize_letters.py --csv mouse_velocities.csv --output my_visualizations")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
