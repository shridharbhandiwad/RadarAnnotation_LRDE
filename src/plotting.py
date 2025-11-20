"""Plotting utilities for GUI visualization using pyqtgraph"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

try:
    import pyqtgraph as pg
    from PyQt6 import QtCore, QtGui
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    logging.warning("PyQtGraph not available")

logger = logging.getLogger(__name__)


class PPIPlotWidget:
    """PPI (Plan Position Indicator) plot widget"""
    
    def __init__(self, parent=None):
        """Initialize PPI plot"""
        if not HAS_PYQTGRAPH:
            raise RuntimeError("PyQtGraph is required")
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget(parent=parent)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('left', 'Y Position', units='km')
        self.plot_widget.setLabel('bottom', 'X Position', units='km')
        self.plot_widget.setTitle('PPI - Plan Position Indicator')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add legend
        self.plot_widget.addLegend()
        
        # Store plot items
        self.scatter_plots = {}
        self.selected_track = None
        
        # Color map for tracks
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
    
    def clear(self):
        """Clear all plots"""
        self.plot_widget.clear()
        self.scatter_plots = {}
        self.selected_track = None
    
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
        
        if color_by == 'trackid':
            # Plot by track
            for idx, trackid in enumerate(df['trackid'].unique()):
                mask = df['trackid'] == trackid
                color_idx = idx % len(self.colors)
                
                scatter = pg.ScatterPlotItem(
                    x=x_km[mask],
                    y=y_km[mask],
                    size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*self.colors[color_idx]),
                    name=f'Track {int(trackid)}'
                )
                
                self.plot_widget.addItem(scatter)
                self.scatter_plots[trackid] = scatter
        
        elif color_by == 'Annotation' and 'Annotation' in df.columns:
            # Plot by annotation type
            annotations = df['Annotation'].unique()
            
            for idx, annotation in enumerate(annotations):
                if annotation == 'invalid':
                    continue
                
                mask = df['Annotation'] == annotation
                color_idx = idx % len(self.colors)
                
                scatter = pg.ScatterPlotItem(
                    x=x_km[mask],
                    y=y_km[mask],
                    size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*self.colors[color_idx]),
                    name=annotation[:20]  # Truncate long names
                )
                
                self.plot_widget.addItem(scatter)
    
    def highlight_track(self, trackid: int):
        """Highlight a specific track
        
        Args:
            trackid: Track ID to highlight
        """
        self.selected_track = trackid
        
        for tid, scatter in self.scatter_plots.items():
            if tid == trackid:
                scatter.setSize(10)
                scatter.setZValue(10)
            else:
                scatter.setSize(5)
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
