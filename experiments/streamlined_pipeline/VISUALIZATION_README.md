# CTU Graph Visualization Tools

This directory contains tools for visualizing CTU (Conceptual Text Units) graphs from your relation data files. These tools are perfect for demonstrating your research to supervisors and colleagues.

## Quick Start

### Option 1: Super Quick Demo (Recommended for first-time users)
```bash
cd scripts
python quick_demo.py
```
This will create visualizations for the Advance Authorisation scheme and save them to `../demo_visualizations/`.

### Option 2: Interactive Demo
```bash
cd scripts
python demo_visualization.py
```
This will show you all available schemes and let you choose which one to visualize.

### Option 3: Command Line Tool
```bash
cd scripts
python ctu_graph_visualizer.py ../output_data/ctu_relations_production_ready/aa_ctus_production_ready.json
```

## Installation

First, install the required dependencies:
```bash
pip install -r visualization_requirements.txt
```

For interactive features (recommended):
```bash
pip install plotly
```

## What You Get

The visualization tools create several types of outputs:

### 1. Static Visualizations (.png)
- **Hierarchical Layout**: Shows CTUs organized by sections
- **Spring Layout**: Shows CTUs in a natural network layout
- Color-coded by CTU roles (ContextObjective, BenefitsAssistance, etc.)
- Edge thickness indicates confidence levels

### 2. Interactive Visualizations (.html)
- **Interactive Graph**: Hover over nodes to see details
- **Dashboard**: Comprehensive overview with multiple charts
- Open in any web browser
- Zoom, pan, and explore the graph interactively

### 3. Data Exports (.json)
- Graph data in JSON format for further analysis
- Compatible with network analysis tools

## Understanding the Visualizations

### Node Colors
- 🔵 **ContextObjective**: Blue - Describes the scheme's purpose
- 🟣 **BenefitsAssistance**: Purple - Lists benefits and assistance
- 🟠 **Eligibility**: Orange - Eligibility criteria
- 🔴 **ApplicationProcess**: Red - How to apply
- 🟣 **AuthoritiesGovernance**: Dark Purple - Governing authorities
- 🟢 **TimelineFrequency**: Teal - Time-related information
- 🟡 **DefinitionsReferences**: Light Orange - Definitions and references

### Edge Types
- **PRECEDES**: Sequential relationship (most common)
- **PREREQUISITE_OF**: Prerequisite relationship
- **ENABLES**: Enabling relationship
- **ADMINISTERED_BY**: Administrative relationship

### Edge Thickness
- Thicker edges = Higher confidence in the relationship
- Color intensity = Confidence level

## Files Created

When you run the visualization, you'll get files like:
```
demo_visualizations/
├── aa_demo_static.png              # Static graph visualization
├── aa_demo_interactive.html        # Interactive graph
├── aa_demo_dashboard.html          # Comprehensive dashboard
└── aa_demo_graph_data.json         # Raw graph data
```

## Advanced Usage

### Custom Visualizations
```python
from ctu_graph_visualizer import CTUGraphVisualizer

# Load your data
visualizer = CTUGraphVisualizer("path/to/your/ctu_data.json")

# Create custom visualizations
visualizer.create_static_visualization("output.png", layout='spring', max_nodes=50)
visualizer.create_interactive_visualization("output.html")
visualizer.create_summary_dashboard("dashboard.html")
```

### Command Line Options
```bash
python ctu_graph_visualizer.py data.json \
    --output-dir ./my_visualizations \
    --format all \
    --layout hierarchical \
    --max-nodes 30
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError"**: Make sure you're in the `scripts` directory
2. **"File not found"**: Check that your relation files exist in `../output_data/ctu_relations_production_ready/`
3. **"Plotly not available"**: Install with `pip install plotly` for interactive features
4. **Large graphs**: Use `--max-nodes` to limit the number of nodes shown

### Performance Tips

- For large graphs (>100 nodes), use `max_nodes=30` for static visualizations
- Interactive visualizations work better with smaller graphs
- Use hierarchical layout for document-like structures
- Use spring layout for network-like structures

## Example Output

The visualizations will show:
- **Node positions**: Organized by document structure or network relationships
- **Node colors**: Different colors for different CTU roles
- **Node sizes**: Larger nodes for higher confidence CTUs
- **Edge colors**: Color-coded by relationship type
- **Edge thickness**: Proportional to confidence levels

## For Supervisors

When demonstrating to your supervisor:

1. **Start with the dashboard** (`*_dashboard.html`) - gives a comprehensive overview
2. **Show the static visualization** (`*_static.png`) - easy to understand at a glance
3. **Use the interactive version** (`*_interactive.html`) - allows exploration of details
4. **Explain the color coding** - different colors represent different types of information
5. **Highlight key relationships** - show how CTUs connect to form the complete picture

The visualizations make it easy to see:
- How information flows through the document
- Which concepts are most central
- How different types of information relate to each other
- The overall structure and organization of the scheme

## Support

If you encounter issues:
1. Check that all required files exist
2. Verify you're in the correct directory
3. Install missing dependencies
4. Try the quick demo first to test your setup
