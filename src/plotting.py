"""Plotting utilities for GUI visualization using pyqtgraph"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

try:
    import pyqtgraph as pg
    from PyQt6 import QtCore, QtGui, QtWidgets
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    QtCore = None  # Fallback for when PyQt6 is not available
    logging.warning("PyQtGraph not available")

from .utils import cartesian_to_polar, polar_to_cartesian

logger = logging.getLogger(__name__)


def get_annotation_color(annotation: str) -> tuple:
    """Get color for annotation combination
    
    Args:
        annotation: Annotation string (may be composite like 'LevelFlight+HighSpeed')
        
    Returns:
        RGB color tuple
    """
    # Define color mapping for common annotation patterns
    color_map = {
        # Single annotations
        'LevelFlight': (0, 150, 255),      # Sky blue
        'Climbing': (255, 128, 0),          # Orange
        'Descending': (255, 0, 128),        # Pink
        'HighSpeed': (255, 0, 0),           # Red
        'LowSpeed': (0, 255, 0),            # Green
        'Turning': (255, 255, 0),           # Yellow
        'Straight': (100, 200, 100),        # Light green
        'LightManeuver': (150, 150, 255),   # Light blue
        'HighManeuver': (255, 0, 255),      # Magenta
        'Incoming': (255, 165, 0),          # Orange
        'Outgoing': (0, 255, 255),          # Cyan
        'FixedRange': (128, 128, 128),      # Gray
        
        # Common combinations
        'LevelFlight+HighSpeed': (255, 100, 100),       # Light red
        'LevelFlight+LowSpeed': (100, 255, 100),        # Light green
        'Climbing+HighSpeed': (255, 150, 0),            # Deep orange
        'Descending+HighSpeed': (255, 50, 150),         # Hot pink
        'Turning+HighSpeed': (255, 200, 0),             # Gold
        'Turning+LowSpeed': (200, 255, 100),            # Yellow-green
        'LevelFlight+Straight': (100, 180, 255),        # Bright sky blue
        'HighManeuver+Turning': (200, 0, 255),          # Purple
    }
    
    # Try exact match first
    if annotation in color_map:
        return color_map[annotation]
    
    # Check for partial matches and blend colors
    parts = annotation.split('+') if '+' in annotation else [annotation]
    colors = []
    for part in parts:
        if part in color_map:
            colors.append(color_map[part])
    
    if colors:
        # Average the colors
        avg_color = tuple(int(sum(c[i] for c in colors) / len(colors)) for i in range(3))
        return avg_color
    
    # Default fallback color
    return (128, 128, 128)  # Gray


class PPIPlotWidget:
    """PPI (Plan Position Indicator) plot widget"""
    
    def __init__(self, parent=None):
        """Initialize PPI plot"""
        if not HAS_PYQTGRAPH:
            raise RuntimeError("PyQtGraph is required")
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget(parent=parent)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('left', 'North-South', units='km')
        self.plot_widget.setLabel('bottom', 'East-West', units='km')
        self.plot_widget.setTitle('PPI - Plan Position Indicator (Radar View)')
        self.plot_widget.showGrid(x=False, y=False)  # Disable default grid, we'll draw custom
        
        # Remove default background
        self.plot_widget.setBackground('#0a0a0a')  # Dark background like radar screen
        
        # Add legend
        self.plot_widget.addLegend()
        
        # Store plot items and data
        self.scatter_plots = {}
        self.selected_track = None
        self.track_data = {}  # Store track data for tooltips
        self.coordinate_mode = 'polar_circular'  # New circular polar mode
        
        # Store range rings and azimuth lines
        self.range_rings = []
        self.azimuth_lines = []
        self.range_labels = []
        self.azimuth_labels = []
        
        # Create tooltip text item
        self.tooltip = pg.TextItem(anchor=(0, 1), color='white', fill=(0, 0, 0, 180))
        self.tooltip.setZValue(100)
        self.plot_widget.addItem(self.tooltip)
        self.tooltip.hide()
        
        # Connect hover event
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        # Color map for tracks (fallback)
        self.colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring green
            (255, 0, 128),    # Rose
        ]
    
    def set_coordinate_mode(self, mode: str):
        """Set coordinate display mode
        
        Args:
            mode: 'cartesian', 'polar', or 'polar_circular'
        """
        if mode not in ['cartesian', 'polar', 'polar_circular']:
            raise ValueError("Mode must be 'cartesian', 'polar', or 'polar_circular'")
        
        self.coordinate_mode = mode
        
        # Update axis labels and display
        if mode == 'polar_circular':
            self.plot_widget.setLabel('left', 'North-South', units='km')
            self.plot_widget.setLabel('bottom', 'East-West', units='km')
            self.plot_widget.setTitle('PPI - Plan Position Indicator (Radar View)')
            self.plot_widget.showGrid(x=False, y=False)
        elif mode == 'polar':
            self.plot_widget.setLabel('left', 'Range', units='km')
            self.plot_widget.setLabel('bottom', 'Azimuth', units='degrees')
            self.plot_widget.setTitle('PPI - Range vs Azimuth')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        else:
            self.plot_widget.setLabel('left', 'Y Position', units='km')
            self.plot_widget.setLabel('bottom', 'X Position', units='km')
            self.plot_widget.setTitle('PPI - Plan Position Indicator')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
    
    def clear(self):
        """Clear all plots"""
        self.plot_widget.clear()
        self.scatter_plots = {}
        self.selected_track = None
        self.track_data = {}
        self.range_rings = []
        self.azimuth_lines = []
        self.range_labels = []
        self.azimuth_labels = []
        
        # Re-add tooltip after clear
        self.tooltip = pg.TextItem(anchor=(0, 1), color='white', fill=(0, 0, 0, 180))
        self.tooltip.setZValue(100)
        self.plot_widget.addItem(self.tooltip)
        self.tooltip.hide()
    
    def draw_circular_ppi_background(self, max_range_km: float):
        """Draw circular PPI background with range rings and azimuth lines
        
        Args:
            max_range_km: Maximum range to display in kilometers
        """
        # Clear previous background elements
        for item in self.range_rings + self.azimuth_lines + self.range_labels + self.azimuth_labels:
            self.plot_widget.removeItem(item)
        
        self.range_rings = []
        self.azimuth_lines = []
        self.range_labels = []
        self.azimuth_labels = []
        
        # Draw range rings (concentric circles)
        num_rings = 5
        ring_spacing = max_range_km / num_rings
        
        for i in range(1, num_rings + 1):
            range_km = i * ring_spacing
            
            # Create circle
            circle = pg.QtWidgets.QGraphicsEllipseItem(-range_km, -range_km, 2*range_km, 2*range_km)
            circle.setPen(pg.mkPen(color=(0, 150, 0), width=1, style=QtCore.Qt.PenStyle.DashLine))
            circle.setZValue(-1)
            self.plot_widget.addItem(circle)
            self.range_rings.append(circle)
            
            # Add range label
            label = pg.TextItem(f'{range_km:.1f} km', color=(0, 200, 0), anchor=(0.5, 0.5))
            label.setPos(0, range_km)
            label.setZValue(1)
            self.plot_widget.addItem(label)
            self.range_labels.append(label)
        
        # Draw azimuth lines (radial lines every 30 degrees)
        angles_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        angle_labels = ['E', '30°', '60°', 'N', '120°', '150°', 'W', '210°', '240°', 'S', '300°', '330°']
        
        for angle_deg, label_text in zip(angles_deg, angle_labels):
            # Convert angle to radians (0° = East, 90° = North)
            angle_rad = np.radians(angle_deg)
            
            # Calculate end point
            x_end = max_range_km * np.cos(angle_rad)
            y_end = max_range_km * np.sin(angle_rad)
            
            # Create line from center to edge
            line = pg.PlotCurveItem([0, x_end], [0, y_end], 
                                    pen=pg.mkPen(color=(0, 150, 0), width=1))
            line.setZValue(-1)
            self.plot_widget.addItem(line)
            self.azimuth_lines.append(line)
            
            # Add azimuth label at the edge
            label_distance = max_range_km * 1.05
            label_x = label_distance * np.cos(angle_rad)
            label_y = label_distance * np.sin(angle_rad)
            
            label = pg.TextItem(label_text, color=(100, 255, 100), anchor=(0.5, 0.5))
            label.setPos(label_x, label_y)
            label.setZValue(1)
            self.plot_widget.addItem(label)
            self.azimuth_labels.append(label)
        
        # Draw outer circle boundary
        boundary = pg.QtWidgets.QGraphicsEllipseItem(-max_range_km, -max_range_km, 
                                                       2*max_range_km, 2*max_range_km)
        boundary.setPen(pg.mkPen(color=(0, 255, 0), width=2))
        boundary.setZValue(-1)
        self.plot_widget.addItem(boundary)
        self.range_rings.append(boundary)
        
        # Set view range
        margin = max_range_km * 0.15
        self.plot_widget.setXRange(-max_range_km - margin, max_range_km + margin)
        self.plot_widget.setYRange(-max_range_km - margin, max_range_km + margin)
    
    def plot_tracks(self, df: pd.DataFrame, color_by: str = 'trackid'):
        """Plot tracks on PPI
        
        Args:
            df: DataFrame with x, y, trackid columns
            color_by: Column to use for coloring ('trackid' or 'Annotation')
        """
        self.clear()
        
        if len(df) == 0:
            return
        
        # Convert to km
        x_km = df['x'].values / 1000.0
        y_km = df['y'].values / 1000.0
        
        # Compute polar coordinates
        range_km, azimuth_deg = cartesian_to_polar(df['x'].values, df['y'].values)
        range_km = range_km / 1000.0  # Convert to km
        
        # Choose which coordinates to plot
        if self.coordinate_mode == 'polar':
            plot_x = azimuth_deg
            plot_y = range_km
        elif self.coordinate_mode == 'polar_circular':
            # For circular PPI, plot in Cartesian but draw circular background
            plot_x = x_km
            plot_y = y_km
            
            # Determine max range for background
            max_range = np.max(np.sqrt(x_km**2 + y_km**2))
            if max_range < 10:
                max_range = 10  # Minimum 10 km display
            max_range = np.ceil(max_range / 10) * 10  # Round up to nearest 10 km
            
            # Draw circular PPI background
            self.draw_circular_ppi_background(max_range)
        else:
            plot_x = x_km
            plot_y = y_km
        
        # Store track data for tooltips (store both coordinate systems)
        for trackid in df['trackid'].unique():
            track_df = df[df['trackid'] == trackid].copy()
            track_df['x_km'] = track_df['x'] / 1000.0
            track_df['y_km'] = track_df['y'] / 1000.0
            
            # Add polar coordinates
            track_range, track_azimuth = cartesian_to_polar(track_df['x'].values, track_df['y'].values)
            track_df['range_km'] = track_range / 1000.0
            track_df['azimuth_deg'] = track_azimuth
            
            self.track_data[trackid] = track_df
        
        if color_by == 'trackid':
            # Plot by track
            for idx, trackid in enumerate(df['trackid'].unique()):
                mask = df['trackid'] == trackid
                color_idx = idx % len(self.colors)
                
                scatter = pg.ScatterPlotItem(
                    x=plot_x[mask],
                    y=plot_y[mask],
                    size=8,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*self.colors[color_idx]),
                    name=f'Track {int(trackid)}',
                    hoverable=True,
                    hoverPen=pg.mkPen('yellow', width=2),
                    hoverBrush=pg.mkBrush(255, 255, 0, 150)
                )
                
                self.plot_widget.addItem(scatter)
                self.scatter_plots[trackid] = scatter
        
        elif color_by == 'Annotation' and 'Annotation' in df.columns:
            # Plot by annotation type with color coding
            annotations = df['Annotation'].unique()
            annotation_legend = {}
            
            for annotation in annotations:
                if annotation == 'invalid' or pd.isna(annotation):
                    continue
                
                mask = df['Annotation'] == annotation
                
                # Use annotation-based color
                color = get_annotation_color(annotation)
                
                scatter = pg.ScatterPlotItem(
                    x=plot_x[mask],
                    y=plot_y[mask],
                    size=8,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*color),
                    name=annotation[:25],  # Truncate long names
                    hoverable=True,
                    hoverPen=pg.mkPen('yellow', width=2),
                    hoverBrush=pg.mkBrush(255, 255, 0, 150)
                )
                
                self.plot_widget.addItem(scatter)
                annotation_legend[annotation] = color
    
    def on_mouse_moved(self, pos):
        """Handle mouse movement for tooltip display
        
        Args:
            pos: Mouse position in scene coordinates
        """
        # Convert scene position to data coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find nearest point across all tracks
        min_dist = float('inf')
        nearest_info = None
        
        for trackid, track_df in self.track_data.items():
            # Calculate distances based on current coordinate mode
            if self.coordinate_mode == 'polar':
                # In polar mode: x=azimuth, y=range
                distances = np.sqrt((track_df['azimuth_deg'] - x)**2 + (track_df['range_km'] - y)**2)
            else:
                # In cartesian mode: x=x, y=y
                distances = np.sqrt((track_df['x_km'] - x)**2 + (track_df['y_km'] - y)**2)
            
            min_track_dist = distances.min()
            
            if min_track_dist < min_dist:
                min_dist = min_track_dist
                nearest_idx = distances.idxmin()
                nearest_row = track_df.loc[nearest_idx]
                nearest_info = {
                    'trackid': int(trackid),
                    'time': nearest_row['time'],
                    'x': nearest_row['x_km'],
                    'y': nearest_row['y_km'],
                    'range': nearest_row['range_km'],
                    'azimuth': nearest_row['azimuth_deg'],
                    'annotation': nearest_row.get('Annotation', 'N/A')
                }
        
        # Show tooltip if close enough (threshold depends on mode)
        if self.coordinate_mode == 'polar':
            threshold = 5.0
        else:
            threshold = 0.5
        
        if min_dist < threshold and nearest_info:
            # Format tooltip text with both coordinate systems
            tooltip_text = (
                f"Track ID: {nearest_info['trackid']}\n"
                f"Time: {nearest_info['time']:.2f} s\n"
                f"Cartesian: ({nearest_info['x']:.2f}, {nearest_info['y']:.2f}) km\n"
                f"Polar: Range={nearest_info['range']:.2f} km, Az={nearest_info['azimuth']:.1f}°\n"
                f"Annotation: {nearest_info['annotation']}"
            )
            
            self.tooltip.setText(tooltip_text)
            self.tooltip.setPos(x, y)
            self.tooltip.show()
        else:
            self.tooltip.hide()
    
    def highlight_track(self, trackid: int):
        """Highlight a specific track
        
        Args:
            trackid: Track ID to highlight
        """
        self.selected_track = trackid
        
        for tid, scatter in self.scatter_plots.items():
            if tid == trackid:
                scatter.setSize(12)
                scatter.setZValue(10)
            else:
                scatter.setSize(8)
                scatter.setZValue(1)
    
    def get_widget(self):
        """Get the underlying Qt widget"""
        return self.plot_widget


class TimeSeriesPlotWidget:
    """Time series plot widget for multiple variables"""
    
    def __init__(self, parent=None):
        """Initialize time series plot"""
        if not HAS_PYQTGRAPH:
            raise RuntimeError("PyQtGraph is required")
        
        # Create graphics layout widget with multiple plots
        self.layout_widget = pg.GraphicsLayoutWidget(parent=parent)
        
        # Create subplots
        self.altitude_plot = self.layout_widget.addPlot(row=0, col=0, title="Altitude vs Time")
        self.altitude_plot.setLabel('left', 'Altitude', units='m')
        self.altitude_plot.setLabel('bottom', 'Time', units='s')
        self.altitude_plot.showGrid(x=True, y=True, alpha=0.3)
        self.altitude_plot.addLegend()
        
        self.speed_plot = self.layout_widget.addPlot(row=1, col=0, title="Speed vs Time")
        self.speed_plot.setLabel('left', 'Speed', units='m/s')
        self.speed_plot.setLabel('bottom', 'Time', units='s')
        self.speed_plot.showGrid(x=True, y=True, alpha=0.3)
        self.speed_plot.addLegend()
        
        self.curvature_plot = self.layout_widget.addPlot(row=2, col=0, title="Curvature vs Time")
        self.curvature_plot.setLabel('left', 'Curvature', units='rad/m')
        self.curvature_plot.setLabel('bottom', 'Time', units='s')
        self.curvature_plot.showGrid(x=True, y=True, alpha=0.3)
        self.curvature_plot.addLegend()
        
        # Link x-axes
        self.speed_plot.setXLink(self.altitude_plot)
        self.curvature_plot.setXLink(self.altitude_plot)
        
        # Store plot items
        self.plot_items = {}
        
        # Color map
        self.colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
        ]
    
    def clear(self):
        """Clear all plots"""
        self.altitude_plot.clear()
        self.speed_plot.clear()
        self.curvature_plot.clear()
        self.plot_items = {}
    
    def plot_tracks(self, df: pd.DataFrame):
        """Plot track time series
        
        Args:
            df: DataFrame with time series data
        """
        self.clear()
        
        if len(df) == 0:
            return
        
        for idx, trackid in enumerate(df['trackid'].unique()):
            track_df = df[df['trackid'] == trackid].sort_values('time')
            
            if len(track_df) < 2:
                continue
            
            color_idx = idx % len(self.colors)
            color = self.colors[color_idx]
            pen = pg.mkPen(color=color, width=2)
            
            # Altitude plot
            alt_curve = self.altitude_plot.plot(
                track_df['time'].values,
                track_df['z'].values,
                pen=pen,
                name=f'Track {int(trackid)}'
            )
            
            # Speed plot (if available)
            if 'speed' in track_df.columns:
                speed_curve = self.speed_plot.plot(
                    track_df['time'].values,
                    track_df['speed'].values,
                    pen=pen,
                    name=f'Track {int(trackid)}'
                )
            
            # Curvature plot (if available)
            if 'curvature' in track_df.columns:
                curv_curve = self.curvature_plot.plot(
                    track_df['time'].values,
                    track_df['curvature'].values,
                    pen=pen,
                    name=f'Track {int(trackid)}'
                )
            
            self.plot_items[trackid] = {
                'altitude': alt_curve,
                'speed': speed_curve if 'speed' in track_df.columns else None,
                'curvature': curv_curve if 'curvature' in track_df.columns else None
            }
    
    def highlight_track(self, trackid: int):
        """Highlight a specific track
        
        Args:
            trackid: Track ID to highlight
        """
        for tid, items in self.plot_items.items():
            width = 4 if tid == trackid else 2
            alpha = 255 if tid == trackid else 100
            
            for plot_name, item in items.items():
                if item is not None:
                    pen = item.opts['pen']
                    color = pen.color()
                    color.setAlpha(alpha)
                    new_pen = pg.mkPen(color=color, width=width)
                    item.setPen(new_pen)
    
    def get_widget(self):
        """Get the underlying Qt widget"""
        return self.layout_widget


class AnnotationHighlighter:
    """Highlight specific annotations on plots"""
    
    def __init__(self, ppi_widget: PPIPlotWidget, ts_widget: TimeSeriesPlotWidget):
        """Initialize highlighter
        
        Args:
            ppi_widget: PPI plot widget
            ts_widget: Time series plot widget
        """
        self.ppi_widget = ppi_widget
        self.ts_widget = ts_widget
        self.highlight_items = []
    
    def clear_highlights(self):
        """Clear all highlights"""
        for item in self.highlight_items:
            self.ppi_widget.plot_widget.removeItem(item)
        self.highlight_items = []
    
    def highlight_annotation(self, df: pd.DataFrame, annotation: str):
        """Highlight points with specific annotation
        
        Args:
            df: DataFrame with data
            annotation: Annotation to highlight
        """
        self.clear_highlights()
        
        if 'Annotation' not in df.columns:
            return
        
        # Filter points with annotation
        mask = df['Annotation'].str.contains(annotation, case=False, na=False)
        highlighted_df = df[mask]
        
        if len(highlighted_df) == 0:
            return
        
        # Add highlighted points to PPI
        x_km = highlighted_df['x'].values / 1000.0
        y_km = highlighted_df['y'].values / 1000.0
        
        scatter = pg.ScatterPlotItem(
            x=x_km,
            y=y_km,
            size=15,
            symbol='o',
            pen=pg.mkPen('y', width=2),
            brush=pg.mkBrush(255, 255, 0, 100),
            name=f'Highlighted: {annotation}'
        )
        
        self.ppi_widget.plot_widget.addItem(scatter)
        self.highlight_items.append(scatter)
        
        logger.info(f"Highlighted {len(highlighted_df)} points with annotation: {annotation}")


def create_ppi_widget(parent=None):
    """Create PPI plot widget
    
    Args:
        parent: Parent Qt widget
        
    Returns:
        PPIPlotWidget instance
    """
    return PPIPlotWidget(parent)


def create_timeseries_widget(parent=None):
    """Create time series plot widget
    
    Args:
        parent: Parent Qt widget
        
    Returns:
        TimeSeriesPlotWidget instance
    """
    return TimeSeriesPlotWidget(parent)
