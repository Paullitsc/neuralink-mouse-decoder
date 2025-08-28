# Letter Visualization Tool

## Overview

The `visualize_letters.py` script creates high-quality visualizations for each letter detected in the mouse velocity data. It leverages the existing robust decoding algorithms to ensure optimal success rates and produces publication-ready visualizations.

## Features

### 🎯 **Optimized for Success**
- Uses proven pause detection algorithms for accurate stroke segmentation
- Implements self-training for writer adaptation
- Leverages all 26 uppercase letter prototypes
- Automatic confidence scoring and validation

### 📊 **Comprehensive Visualizations**
- Individual letter plots with stroke paths, start/end markers, and direction arrows
- Summary analysis with letter frequency charts
- Confidence score distributions
- High-resolution output (300 DPI) suitable for publications

### 🔧 **Advanced Algorithms**
- Automatic threshold detection for optimal segmentation
- RDP simplification for clean shape analysis
- Chamfer distance matching for accurate classification
- Writer adaptation through clustering and prototype blending

## Installation

### Prerequisites
- Python 3.9+
- Required packages (install via `pip install -r requirements.txt`):
  - numpy >= 1.21.0
  - pandas >= 1.3.0
  - matplotlib >= 3.5.0

### Quick Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Test the installation
python test_visualization.py
```

## Usage

### Basic Usage
```bash
# Run with default settings (uses mouse_velocities.csv)
python visualize_letters.py

# Specify custom CSV file
python visualize_letters.py --csv my_data.csv

# Custom output directory
python visualize_letters.py --output my_visualizations
```

### Advanced Options
```bash
# Disable self-training (may reduce success rate)
python visualize_letters.py --no_self_train

# Custom clustering parameters
python visualize_letters.py --clusters 8 --blend_weight 0.7

# Full example with all options
python visualize_letters.py \
  --csv mouse_velocities.csv \
  --output letter_analysis \
  --clusters 6 \
  --blend_weight 0.5
```

## Output

The tool generates several types of output:

### 📁 **Directory Structure**
```
letter_visualizations/
├── stroke_01_A.png          # Individual letter plots
├── stroke_02_N.png
├── stroke_03_K.png
├── ...
├── summary_analysis.png      # Summary charts
└── letter_summary.txt        # Text summary
```

### 📊 **Visualization Types**

1. **Individual Letter Plots**
   - Blue stroke path with thickness
   - Green start marker (●)
   - Red end marker (●)
   - Direction arrows showing writing flow
   - Confidence scores and letter classification

2. **Summary Analysis**
   - Letter frequency bar chart
   - Confidence score distribution histogram
   - Comprehensive statistics

3. **Text Summary**
   - Total strokes analyzed
   - Average confidence scores
   - Letter frequency breakdown
   - Complete decoded string

## Algorithm Details

### 🧠 **Stroke Segmentation**
- **Pause Detection**: Automatically finds optimal speed thresholds using histogram gap analysis
- **No Magic Numbers**: Eliminates manual parameter tuning
- **Robust Handling**: Adapts to varying writing speeds and pause patterns

### 🔍 **Letter Recognition**
- **Prototype Matching**: Uses hand-crafted letter prototypes (A-Z)
- **Chamfer Distance**: Robust shape similarity measurement
- **RDP Simplification**: Reduces noise while preserving essential features
- **Writer Adaptation**: Learns from current recording for improved accuracy

### 📈 **Self-Training Process**
1. **Clustering**: Groups similar strokes using K-means
2. **Mapping**: Associates clusters with letter prototypes
3. **Blending**: Combines hand-crafted and adapted prototypes
4. **Optimization**: Iteratively improves classification accuracy

## Performance Optimization

### ⚡ **Speed Improvements**
- Efficient numpy operations for large datasets
- Optimized Chamfer distance calculations
- Smart caching of processed stroke data
- Parallel processing where applicable

### 🎯 **Accuracy Enhancements**
- Multiple rotation and mirroring attempts
- Feature-based pre-filtering
- Confidence-weighted decision making
- Robust error handling and validation

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **CSV Format Issues**
   - Verify CSV has columns: `timestamp`, `velocity_x`, `velocity_y`
   - Check for missing or corrupted data

3. **Memory Issues**
   - Reduce `--clusters` parameter for large datasets
   - Use `--no_self_train` for faster processing

4. **Low Confidence Scores**
   - Enable self-training with `--self_train`
   - Adjust `--blend_weight` between 0.3-0.7
   - Increase `--clusters` for better adaptation

### Debug Mode
```bash
# Run with verbose output
python visualize_letters.py --verbose_self

# Check individual components
python test_visualization.py
```

## Examples

### Example 1: Basic Analysis
```bash
python visualize_letters.py
```
Output: Creates `letter_visualizations/` with all plots and summary

### Example 2: Custom Analysis
```bash
python visualize_letters.py \
  --csv my_data.csv \
  --output custom_analysis \
  --clusters 8 \
  --blend_weight 0.6
```
Output: Creates `custom_analysis/` with enhanced clustering

### Example 3: Quick Processing
```bash
python visualize_letters.py --no_self_train --output quick_results
```
Output: Faster processing without writer adaptation

## Integration

### With Existing Decoder
The visualization tool seamlessly integrates with `decode_mouse_velocities.py`:
- Uses the same segmentation algorithms
- Shares letter prototypes and classification logic
- Maintains consistency between decoding and visualization

### With Answer Checker
Automatically verifies results using `check_answer.py`:
- Runs validation after visualization
- Reports success/failure status
- Provides confidence metrics

## Advanced Usage

### Custom Prototypes
Extend the letter recognition by modifying `build_prototypes()` in `decode_mouse_velocities.py`

### Batch Processing
```bash
# Process multiple CSV files
for file in *.csv; do
  python visualize_letters.py --csv "$file" --output "results_${file%.csv}"
done
```

### API Usage
```python
from visualize_letters import LetterVisualizer

# Create visualizer
viz = LetterVisualizer(use_self_training=True, clusters=6)

# Load data
viz.load_and_process_data('mouse_velocities.csv')

# Generate visualizations
letters, confidences = viz.visualize_all_letters('output_dir')

# Access results
print(f"Decoded: {''.join(letters)}")
print(f"Average confidence: {np.mean(confidences):.3f}")
```

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Include type hints for all functions
- Add comprehensive docstrings
- Maintain backward compatibility

### Testing
```bash
# Run test suite
python test_visualization.py

# Test with sample data
python visualize_letters.py --csv sample_data.csv
```

## License

This tool is part of the Neuralink Software Engineering Intern Challenge and follows the same licensing terms.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test output from `test_visualization.py`
3. Verify CSV data format and integrity
4. Check system requirements and dependencies

---

**Optimized for Success**: This visualization tool is designed to maximize the accuracy of letter recognition while providing clear, actionable insights into the decoding process.
