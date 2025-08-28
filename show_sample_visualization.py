#!/usr/bin/env python3
"""
show_sample_visualization.py

Simple script to display a sample visualization from the generated output.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def show_sample_visualization():
    """Display a sample visualization from the generated output."""
    
    # Check if visualizations exist
    if not os.path.exists('letter_visualizations'):
        print("No visualizations found. Run 'python3 visualize_letters.py' first.")
        return
    
    # List available visualizations
    viz_dir = 'letter_visualizations'
    files = [f for f in os.listdir(viz_dir) if f.endswith('.png') and f.startswith('stroke_')]
    
    if not files:
        print("No stroke visualizations found.")
        return
    
    print(f"Found {len(files)} stroke visualizations:")
    for i, file in enumerate(sorted(files)):
        print(f"  {i+1:2d}. {file}")
    
    # Show the first visualization
    sample_file = sorted(files)[0]
    sample_path = os.path.join(viz_dir, sample_file)
    
    print(f"\nDisplaying sample visualization: {sample_file}")
    
    # Load and display the image
    img = mpimg.imread(sample_path)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Sample Letter Visualization: {sample_file}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    print(f"\nSample visualization displayed: {sample_file}")
    print("This shows the quality and detail of the generated letter visualizations.")
    print("Each visualization includes:")
    print("  - Blue stroke path with thickness")
    print("  - Green start marker (●)")
    print("  - Red end marker (●)")
    print("  - Direction arrows showing writing flow")
    print("  - Confidence scores and letter classification")

if __name__ == '__main__':
    show_sample_visualization()
