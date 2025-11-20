"""Report Engine - Generate comprehensive analysis reports"""
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import base64
from io import BytesIO

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("Matplotlib not available for report generation")

from .utils import ensure_dir
from .data_engine import get_data_summary
from .autolabel_engine import get_annotation_summary

logger = logging.getLogger(__name__)


def plot_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded PNG image
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def create_ppi_plot(df: pd.DataFrame) -> Optional[str]:
    """Create PPI (Plan Position Indicator) plot
    
    Args:
        df: DataFrame with position data
        
    Returns:
        Base64 encoded image or None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Convert to polar coordinates
    r = np.sqrt(df['x']**2 + df['y']**2) / 1000  # Convert to km
    theta = np.arctan2(df['y'], df['x'])
    
    # Color by track
    for trackid in df['trackid'].unique():
        mask = df['trackid'] == trackid
        ax.scatter(theta[mask], r[mask], s=10, alpha=0.6, label=f'Track {int(trackid)}')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, r.max() * 1.1)
    ax.set_title('PPI - Plan Position Indicator', fontsize=14, pad=20)
    ax.grid(True)
    
    if df['trackid'].nunique() <= 10:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return plot_to_base64(fig)


def create_altitude_plot(df: pd.DataFrame) -> Optional[str]:
    """Create altitude vs time plot
    
    Args:
        df: DataFrame with altitude data
        
    Returns:
        Base64 encoded image or None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for trackid in df['trackid'].unique():
        track_df = df[df['trackid'] == trackid].sort_values('time')
        ax.plot(track_df['time'], track_df['z'], label=f'Track {int(trackid)}', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Altitude (m)', fontsize=12)
    ax.set_title('Altitude vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if df['trackid'].nunique() <= 10:
        ax.legend()
    
    return plot_to_base64(fig)


def create_speed_plot(df: pd.DataFrame) -> Optional[str]:
    """Create speed vs time plot
    
    Args:
        df: DataFrame with speed data
        
    Returns:
        Base64 encoded image or None
    """
    if not HAS_MATPLOTLIB or 'speed' not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for trackid in df['trackid'].unique():
        track_df = df[df['trackid'] == trackid].sort_values('time')
        ax.plot(track_df['time'], track_df['speed'], label=f'Track {int(trackid)}', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.set_title('Speed vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if df['trackid'].nunique() <= 10:
        ax.legend()
    
    return plot_to_base64(fig)


def create_annotation_distribution_plot(annotation_summary: Dict[str, Any]) -> Optional[str]:
    """Create annotation distribution bar plot
    
    Args:
        annotation_summary: Annotation summary dictionary
        
    Returns:
        Base64 encoded image or None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    annotation_dist = annotation_summary.get('annotation_distribution', {})
    
    if not annotation_dist:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = list(annotation_dist.keys())
    counts = [annotation_dist[label]['count'] for label in labels]
    
    ax.bar(labels, counts, alpha=0.7)
    ax.set_xlabel('Annotation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Annotation Distribution', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    return plot_to_base64(fig)


def create_confusion_matrix_plot(confusion_matrix: list, classes: list) -> Optional[str]:
    """Create confusion matrix heatmap
    
    Args:
        confusion_matrix: Confusion matrix as list of lists
        classes: Class labels
        
    Returns:
        Base64 encoded image or None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    cm = np.array(confusion_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    return plot_to_base64(fig)


def generate_html_report(data: Dict[str, Any], output_path: str) -> str:
    """Generate HTML report
    
    Args:
        data: Report data dictionary
        output_path: Output HTML file path
        
    Returns:
        Path to generated report
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Radar Data Annotation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <h1>📡 Radar Data Annotation Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    
    <div class="section">
        <h2>📊 Data Summary</h2>
        <div class="metric">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{total_records}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Number of Tracks</div>
            <div class="metric-value">{num_tracks}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{duration:.1f}s</div>
        </div>
    </div>
    
    {annotation_section}
    
    {model_section}
    
    {visualizations_section}
    
</body>
</html>
"""
    
    # Build sections
    annotation_section = ""
    if 'annotation_summary' in data:
        ann_sum = data['annotation_summary']
        ann_section = f"""
    <div class="section">
        <h2>🏷️ Annotation Summary</h2>
        <div class="metric">
            <div class="metric-label">Valid Records</div>
            <div class="metric-value">{ann_sum.get('valid_records', 0)}</div>
        </div>
        <table>
            <tr>
                <th>Annotation</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""
        for ann, stats in ann_sum.get('annotation_distribution', {}).items():
            ann_section += f"""
            <tr>
                <td>{ann}</td>
                <td>{stats['count']}</td>
                <td>{stats['percentage']:.2f}%</td>
            </tr>
"""
        ann_section += """
        </table>
    </div>
"""
        annotation_section = ann_section
    
    # Model section
    model_section = ""
    if 'model_metrics' in data:
        metrics = data['model_metrics']
        model_name = metrics.get('model_name', 'Unknown')
        train_metrics = metrics.get('train', {})
        test_metrics = metrics.get('test', {})
        
        model_section = f"""
    <div class="section">
        <h2>🤖 Model Performance - {model_name}</h2>
        <div class="metric">
            <div class="metric-label">Train Accuracy</div>
            <div class="metric-value">{train_metrics.get('train_accuracy', 0):.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">{test_metrics.get('accuracy', 0):.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{test_metrics.get('f1_score', 0):.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Training Time</div>
            <div class="metric-value">{train_metrics.get('training_time', 0):.2f}s</div>
        </div>
    </div>
"""
    
    # Visualizations section
    vis_section = '<div class="section"><h2>📈 Visualizations</h2>'
    
    if 'plots' in data:
        plots = data['plots']
        
        if plots.get('ppi'):
            vis_section += f'<h3>PPI Plot</h3><img src="data:image/png;base64,{plots["ppi"]}" alt="PPI Plot">'
        
        if plots.get('altitude'):
            vis_section += f'<h3>Altitude vs Time</h3><img src="data:image/png;base64,{plots["altitude"]}" alt="Altitude Plot">'
        
        if plots.get('speed'):
            vis_section += f'<h3>Speed vs Time</h3><img src="data:image/png;base64,{plots["speed"]}" alt="Speed Plot">'
        
        if plots.get('annotation_dist'):
            vis_section += f'<h3>Annotation Distribution</h3><img src="data:image/png;base64,{plots["annotation_dist"]}" alt="Annotation Distribution">'
        
        if plots.get('confusion_matrix'):
            vis_section += f'<h3>Confusion Matrix</h3><img src="data:image/png;base64,{plots["confusion_matrix"]}" alt="Confusion Matrix">'
    
    vis_section += '</div>'
    
    # Fill template
    data_summary = data.get('data_summary', {})
    html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_records=data_summary.get('total_records', 0),
        num_tracks=data_summary.get('num_tracks', 0),
        duration=data_summary.get('duration_seconds', 0),
        annotation_section=annotation_section,
        model_section=model_section,
        visualizations_section=vis_section
    )
    
    # Write to file
    ensure_dir(Path(output_path).parent)
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {output_path}")
    return output_path


def generate_report(folder: str, out_path: str, include_model_metrics: bool = True) -> str:
    """Generate comprehensive report from data folder
    
    Args:
        folder: Folder containing data files (raw_data.csv, labelled_data.csv, etc.)
        out_path: Output path for report (HTML)
        include_model_metrics: Whether to include model metrics
        
    Returns:
        Path to generated report
    """
    logger.info(f"Generating report for folder: {folder}")
    
    report_data = {}
    
    # Load data
    folder_path = Path(folder)
    
    # Try to load labeled data first, fall back to raw data
    data_file = folder_path / "labelled_data.csv"
    if not data_file.exists():
        data_file = folder_path / "raw_data.csv"
    
    if not data_file.exists():
        logger.error(f"No data file found in {folder}")
        raise FileNotFoundError(f"No data file found in {folder}")
    
    df = pd.read_csv(data_file)
    
    # Data summary
    from .data_engine import get_data_summary
    report_data['data_summary'] = get_data_summary(df)
    
    # Annotation summary (if available)
    if 'Annotation' in df.columns:
        from .autolabel_engine import get_annotation_summary
        report_data['annotation_summary'] = get_annotation_summary(df)
    
    # Model metrics (if available)
    if include_model_metrics:
        metrics_files = list(folder_path.glob('*_metrics.json'))
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                report_data['model_metrics'] = json.load(f)
    
    # Generate plots
    plots = {}
    
    if HAS_MATPLOTLIB:
        logger.info("Generating plots...")
        
        plots['ppi'] = create_ppi_plot(df.head(1000))  # Limit points for performance
        plots['altitude'] = create_altitude_plot(df)
        
        if 'speed' in df.columns:
            plots['speed'] = create_speed_plot(df)
        
        if 'annotation_summary' in report_data:
            plots['annotation_dist'] = create_annotation_distribution_plot(report_data['annotation_summary'])
        
        if 'model_metrics' in report_data:
            test_metrics = report_data['model_metrics'].get('test', {})
            if 'confusion_matrix' in test_metrics and 'classes' in test_metrics:
                plots['confusion_matrix'] = create_confusion_matrix_plot(
                    test_metrics['confusion_matrix'],
                    test_metrics['classes']
                )
    
    report_data['plots'] = plots
    
    # Generate HTML report
    report_path = generate_html_report(report_data, out_path)
    
    return report_path


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Report Engine')
    parser.add_argument('--folder', required=True, help='Data folder')
    parser.add_argument('--out', required=True, help='Output report path (HTML)')
    parser.add_argument('--no-model', action='store_true', help='Exclude model metrics')
    
    args = parser.parse_args()
    
    report_path = generate_report(args.folder, args.out, not args.no_model)
    
    print(f"\nReport generated: {report_path}")
