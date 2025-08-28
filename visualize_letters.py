#!/usr/bin/env python3
"""
visualize_letters.py

Creates visualizations for each letter detected in the mouse velocity data.
Uses the existing robust decoding algorithms to ensure high-quality segmentation
and letter recognition.

Features:
- Automatic pause detection for stroke segmentation
- High-quality letter visualization with proper scaling
- Support for all 26 uppercase letters
- Optimized for success using proven algorithms
- Clean, publication-ready output
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# Import the existing decoding logic
from decode_mouse_velocities import (
    load_csv, reconstruct_positions, segment_strokes, 
    normalize_unit_box, resample_polyline, build_prototypes,
    classify_stroke, StrokeFeatures, build_adapted_prototypes,
    blend_prototypes
)


class LetterVisualizer:
    """High-quality letter visualization with optimized success rate."""
    
    def __init__(self, use_self_training: bool = True, clusters: int = 6, blend_weight: float = 0.5):
        self.use_self_training = use_self_training
        self.clusters = clusters
        self.blend_weight = blend_weight
        self.prototypes = build_prototypes()
        self.adapted_prototypes = None
        
    def load_and_process_data(self, csv_path: str):
        """Load and process the mouse velocity data."""
        print(f"Loading data from {csv_path}...")
        self.df = load_csv(csv_path)
        self.x, self.y = reconstruct_positions(self.df)
        self.vx = self.df['velocity_x'].to_numpy(float)
        self.vy = self.df['velocity_y'].to_numpy(float)
        
        # Use automatic threshold detection for optimal segmentation
        print("Segmenting strokes using automatic pause detection...")
        self.intervals = segment_strokes(self.x, self.y, self.vx, self.vy, 
                                       speed_thresh=0.3, min_pause=5, min_len=80)
        
        print(f"Detected {len(self.intervals)} strokes")
        
        # Apply self-training if enabled for better success rate
        if self.use_self_training and len(self.intervals) > 0:
            print("Applying self-training for writer adaptation...")
            self.adapted_prototypes = build_adapted_prototypes(
                self.x, self.y, self.intervals, self.prototypes, 
                k=self.clusters, verbose=True
            )
            self.prototypes = blend_prototypes(
                self.prototypes, self.adapted_prototypes, self.blend_weight
            )
    
    def extract_stroke_data(self, interval: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract and normalize stroke data for visualization."""
        s, e = interval
        xs = self.x[s:e]
        ys = self.y[s:e]
        
        # Normalize to unit box for consistent visualization
        pts = np.c_[xs - xs.min(), ys - ys.min()]
        pts = normalize_unit_box(pts)
        
        # Resample for smooth visualization
        pts = resample_polyline(pts, 300)
        
        return pts[:, 0], pts[:, 1], np.arange(len(pts))
    
    def classify_stroke_optimized(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[str, float, StrokeFeatures]:
        """Classify stroke using the optimized algorithm."""
        return classify_stroke(xs, ys, self.prototypes)
    
    def create_letter_plot(self, ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, 
                          letter: str, confidence: float, stroke_idx: int):
        """Create a high-quality plot for a single letter."""
        # Plot the stroke path
        ax.plot(xs, ys, 'b-', linewidth=3, alpha=0.8, label=f'Stroke {stroke_idx}')
        
        # Add start and end markers
        ax.plot(xs[0], ys[0], 'go', markersize=8, label='Start')
        ax.plot(xs[-1], ys[-1], 'ro', markersize=8, label='End')
        
        # Add direction arrows every 50 points
        for i in range(0, len(xs)-1, 50):
            if i + 1 < len(xs):
                dx = xs[i+1] - xs[i]
                dy = ys[i+1] - ys[i]
                ax.arrow(xs[i], ys[i], dx*0.1, dy*0.1, 
                        head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7)
        
        # Set title and labels
        ax.set_title(f'Letter: {letter} (Confidence: {confidence:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (normalized)', fontsize=12)
        ax.set_ylabel('Y (normalized)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set equal aspect ratio for proper letter proportions
        ax.set_aspect('equal')
        
        # Add bounding box
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        margin = 0.05
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    
    def create_summary_plot(self, fig: plt.Figure, letters: List[str], confidences: List[float]):
        """Create a summary plot showing all detected letters."""
        ax = fig.add_subplot(111)
        
        # Create a bar chart of letter frequencies
        letter_counts = {}
        for letter in letters:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        if letter_counts:
            letters_list = list(letter_counts.keys())
            counts = list(letter_counts.values())
            
            bars = ax.bar(letters_list, counts, color='skyblue', alpha=0.7, edgecolor='navy')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       str(count), ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Letter Frequency Analysis', fontsize=16, fontweight='bold')
            ax.set_xlabel('Detected Letters', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def visualize_all_letters(self, output_dir: str = "letter_visualizations"):
        """Create comprehensive visualizations for all detected letters."""
        if not hasattr(self, 'intervals'):
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        letters = []
        confidences = []
        
        print(f"\nCreating visualizations for {len(self.intervals)} strokes...")
        
        # Create individual letter plots
        for i, interval in enumerate(self.intervals):
            print(f"Processing stroke {i+1}/{len(self.intervals)}...")
            
            # Extract stroke data
            xs, ys, _ = self.extract_stroke_data(interval)
            
            # Classify the stroke
            letter, confidence, features = self.classify_stroke_optimized(xs, ys)
            letters.append(letter)
            confidences.append(confidence)
            
            # Create individual letter plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            self.create_letter_plot(ax, xs, ys, letter, confidence, i+1)
            
            # Save individual plot
            filename = f"stroke_{i+1:02d}_{letter}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Saved: {filename} (Letter: {letter}, Confidence: {confidence:.3f})")
        
        # Create summary visualization
        print("\nCreating summary visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Summary plot
        self.create_summary_plot(fig, letters, confidences)
        
        # Confidence distribution
        ax2.hist(confidences, bins=20, color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax2.set_title('Confidence Score Distribution', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save summary plot
        summary_filepath = os.path.join(output_dir, "summary_analysis.png")
        plt.savefig(summary_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create text summary
        summary_text = self.create_text_summary(letters, confidences)
        text_filepath = os.path.join(output_dir, "letter_summary.txt")
        with open(text_filepath, 'w') as f:
            f.write(summary_text)
        
        print(f"\nVisualization complete! Output saved to: {output_dir}")
        print(f"Summary: {text_filepath}")
        print(f"Summary plot: {summary_filepath}")
        
        return letters, confidences
    
    def create_text_summary(self, letters: List[str], confidences: List[float]) -> str:
        """Create a text summary of the analysis."""
        summary = "LETTER VISUALIZATION SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        # Overall statistics
        summary += f"Total strokes analyzed: {len(letters)}\n"
        summary += f"Average confidence: {np.mean(confidences):.3f}\n"
        summary += f"Confidence std dev: {np.std(confidences):.3f}\n\n"
        
        # Letter breakdown
        letter_counts = {}
        for letter in letters:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        summary += "LETTER FREQUENCIES:\n"
        summary += "-" * 30 + "\n"
        for letter, count in sorted(letter_counts.items()):
            summary += f"{letter}: {count}\n"
        
        summary += "\nDETAILED STROKE ANALYSIS:\n"
        summary += "-" * 30 + "\n"
        for i, (letter, confidence) in enumerate(zip(letters, confidences)):
            summary += f"Stroke {i+1:2d}: {letter} (confidence: {confidence:.3f})\n"
        
        # Decoded string
        decoded_string = ''.join(letters)
        summary += f"\nDECODED STRING: {decoded_string}\n"
        summary += f"Length: {len(decoded_string)} characters\n"
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize letters from mouse velocity data")
    parser.add_argument('--csv', default='mouse_velocities.csv', 
                       help='Path to mouse velocities CSV file')
    parser.add_argument('--output', default='letter_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--no_self_train', action='store_true',
                       help='Disable self-training (may reduce success rate)')
    parser.add_argument('--clusters', type=int, default=6,
                       help='Number of clusters for self-training')
    parser.add_argument('--blend_weight', type=float, default=0.5,
                       help='Blend weight for adapted prototypes [0..1]')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found!")
        sys.exit(1)
    
    # Check if matplotlib is available
    try:
        import matplotlib
    except ImportError:
        print("Error: matplotlib is required for visualization.")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    print("=== LETTER VISUALIZATION TOOL ===")
    print("Optimized for success using proven decoding algorithms\n")
    
    # Create visualizer
    visualizer = LetterVisualizer(
        use_self_training=not args.no_self_train,
        clusters=args.clusters,
        blend_weight=args.blend_weight
    )
    
    try:
        # Load and process data
        visualizer.load_and_process_data(args.csv)
        
        # Create visualizations
        letters, confidences = visualizer.visualize_all_letters(args.output)
        
        # Print final summary
        decoded_string = ''.join(letters)
        print(f"\n=== FINAL RESULT ===")
        print(f"Decoded string: {decoded_string}")
        print(f"Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
        
        # Check if we can verify the answer
        if os.path.exists('check_answer.py'):
            print("\nVerifying with provided checker...")
            import subprocess
            result = subprocess.run([sys.executable, 'check_answer.py', decoded_string], 
                                 capture_output=True, text=True)
            print(result.stdout.strip())
            if result.returncode == 0:
                print("✅ CHECKER INDICATES CORRECT ANSWER!")
            else:
                print("❌ Checker indicates incorrect answer.")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
