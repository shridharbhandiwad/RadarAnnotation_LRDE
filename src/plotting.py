"""Plotting utilities for GUI visualization using pyqtgraph"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from copy import deepcopy
from pathlib import Path

try:
    import pyqtgraph as pg
    from PyQt6 import QtCore, QtGui, QtWidgets
    from pyqtgraph.exporters import ImageExporter
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    QtCore = None  # Fallback for when PyQt6 is not available
    logging.warning("PyQtGraph not available")

from .utils import cartesian_to_polar, polar_to_cartesian

logger = logging.getLogger(__name__)


class PlotViewHistory:
    """Manages plot view history for undo/redo functionality"""
    
    def __init__(self, max_history: int = 50):
        """Initialize history manager
        
        Args:
            max_history: Maximum number of history states to keep
        """
        self.history = []
        self.current_index = -1
        self.max_history = max_history
    
    def save_state(self, view_range: Dict):
        """Save current view state
        
        Args:
            view_range: Dictionary with 'x' and 'y' ranges
        """
        # Remove any states after current index (for redo after undo)
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Add new state
        self.history.append(deepcopy(view_range))
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self.current_index < len(self.history) - 1
    
    def undo(self) -> Optional[Dict]:
        """Go back to previous state"""
        if self.can_undo():
            self.current_index -= 1
            return deepcopy(self.history[self.current_index])
        return None
    
    def redo(self) -> Optional[Dict]:
        """Go forward to next state"""
        if self.can_redo():
            self.current_index += 1
            return deepcopy(self.history[self.current_index])
        return None
    
    def clear(self):
        """Clear all history"""
        self.history = []
        self.current_index = -1


class MagnifierLens(QtWidgets.QGraphicsEllipseItem):
    """Magnifier lens overlay for plot inspection"""
    
    def __init__(self, radius: float = 50):
        """Initialize magnifier lens
        
        Args:
            radius: Radius of the magnifier lens in pixels
        """
        super().__init__(-radius, -radius, 2*radius, 2*radius)
        self.radius = radius
        self.zoom_factor = 2.0
        
        # Style the lens - pulsing highlight circle
        self.setPen(pg.mkPen(color=(255, 200, 0), width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.setBrush(pg.mkBrush(255, 255, 0, 40))
        self.setZValue(1000)
        self.hide()
    
    def set_zoom_factor(self, factor: float):
        """Set magnification factor"""
        self.zoom_factor = max(1.5, min(10.0, factor))
    
    def set_radius(self, radius: float):
        """Set lens radius"""
        self.radius = max(30, min(150, radius))
        self.setRect(-self.radius, -self.radius, 2*self.radius, 2*self.radius)


def get_annotation_color(annotation: str) -> tuple:
    """Get color for annotation combination with consistent color theme
    
    Args:
        annotation: Annotation string (may be composite like 'LevelFlight+HighSpeed' or 'level,high_speed')
        
    Returns:
        RGB color tuple (consistent with application theme)
    """
    # Define color mapping with consistent, vibrant colors for better visibility on dark radar background
    # Supports both formats: TitleCase+Plus and lowercase,comma
    color_map = {
        # Single annotations - Primary flight characteristics (both formats)
        'LevelFlight': (52, 152, 219),      # Blue (matches app theme)
        'level': (52, 152, 219),
        'level_flight': (52, 152, 219),
        'Climbing': (255, 128, 0),          # Orange
        'ascending': (255, 128, 0),
        'Descending': (255, 85, 150),       # Rose pink
        'descending': (255, 85, 150),
        'HighSpeed': (231, 76, 60),         # Red
        'high_speed': (231, 76, 60),
        'LowSpeed': (46, 204, 113),         # Green (matches app theme)
        'low_speed': (46, 204, 113),
        'Turning': (241, 196, 15),          # Yellow/Gold
        'curved': (241, 196, 15),
        'Straight': (100, 200, 150),        # Mint green
        'linear': (100, 200, 150),
        'LightManeuver': (155, 89, 182),    # Purple
        'light_maneuver': (155, 89, 182),
        'HighManeuver': (236, 77, 177),     # Magenta
        'high_maneuver': (236, 77, 177),
        'Incoming': (230, 126, 34),         # Dark orange
        'incoming': (230, 126, 34),
        'Outgoing': (26, 188, 156),         # Turquoise (matches app theme)
        'outgoing': (26, 188, 156),
        'FixedRange': (149, 165, 166),      # Gray
        'fixed_range': (149, 165, 166),
        'fixed_range_ascending': (180, 180, 180),
        'fixed_range_descending': (120, 120, 120),
        'invalid': (80, 80, 80),            # Dark gray for invalid
        'normal': (200, 200, 200),          # Light gray for normal
    }
    
    # Normalize annotation format (handle both separators)
    normalized = annotation.strip()
    
    # Try exact match first
    if normalized in color_map:
        return color_map[normalized]
    
    # Parse composite annotations (supports both ',' and '+' separators)
    if ',' in normalized:
        parts = [p.strip() for p in normalized.split(',')]
    elif '+' in normalized:
        parts = [p.strip() for p in normalized.split('+')]
    else:
        parts = [normalized]
    
    # Check for partial matches and blend colors
    colors = []
    for part in parts:
        if part in color_map:
            colors.append(color_map[part])
    
    if colors:
        # Average the colors for smooth blending
        avg_color = tuple(int(sum(c[i] for c in colors) / len(colors)) for i in range(3))
        return avg_color
    
    # Default fallback color (neutral gray)
    return (149, 165, 166)  # Gray (matches app theme)


class PPIPlotWidget:
    """PPI (Plan Position Indicator) plot widget with interactive features"""
    
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
        
        # Interactive features
        self.view_history = PlotViewHistory()
        self.magnifier_enabled = False
        self.magnifier = MagnifierLens(radius=60)
        self.plot_widget.addItem(self.magnifier)
        self.magnifier.hide()
        
        # Enable mouse interactions
        self.plot_widget.setMouseEnabled(x=True, y=True)  # Enable panning
        self.plot_widget.enableAutoRange(enable=False)
        
        # Connect events
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)
        self.plot_widget.plotItem.vb.sigRangeChanged.connect(self.on_range_changed)
        
        # Initial view state tracking
        self._last_saved_range = None
        self._range_change_timer = QtCore.QTimer()
        self._range_change_timer.timeout.connect(self._save_current_range)
        self._range_change_timer.setSingleShot(True)
        
        # Color map for tracks (fallback) - matches application theme
        self.colors = [
            (231, 76, 60),      # Red (app theme)
            (46, 204, 113),     # Green (app theme)
            (52, 152, 219),     # Blue (app theme)
            (241, 196, 15),     # Yellow/Gold (app theme)
            (155, 89, 182),     # Purple (app theme)
            (26, 188, 156),     # Turquoise (app theme)
            (230, 126, 34),     # Orange (app theme)
            (236, 77, 177),     # Magenta (app theme)
            (22, 160, 133),     # Dark turquoise
            (192, 57, 43),      # Dark red
        ]
    
    def on_range_changed(self):
        """Handle range change events for history tracking"""
        # Debounce: only save after 500ms of no changes
        self._range_change_timer.start(500)
    
    def _save_current_range(self):
        """Save current view range to history"""
        view_range = self.get_view_range()
        if view_range != self._last_saved_range:
            self.view_history.save_state(view_range)
            self._last_saved_range = view_range
    
    def get_view_range(self) -> Dict:
        """Get current view range"""
        [[xmin, xmax], [ymin, ymax]] = self.plot_widget.viewRange()
        return {'x': (xmin, xmax), 'y': (ymin, ymax)}
    
    def set_view_range(self, view_range: Dict):
        """Set view range"""
        self.plot_widget.setXRange(view_range['x'][0], view_range['x'][1], padding=0)
        self.plot_widget.setYRange(view_range['y'][0], view_range['y'][1], padding=0)
        self._last_saved_range = view_range
    
    def undo_view(self):
        """Undo to previous view state"""
        prev_range = self.view_history.undo()
        if prev_range:
            self.set_view_range(prev_range)
            logger.info("View state undone")
    
    def redo_view(self):
        """Redo to next view state"""
        next_range = self.view_history.redo()
        if next_range:
            self.set_view_range(next_range)
            logger.info("View state redone")
    
    def reset_view(self):
        """Reset view to show all data"""
        self.plot_widget.autoRange()
        self._save_current_range()
        logger.info("View reset to default")
    
    def zoom_in(self):
        """Zoom in by 20%"""
        self.plot_widget.plotItem.vb.scaleBy((0.8, 0.8))
        logger.info("Zoomed in")
    
    def zoom_out(self):
        """Zoom out by 20%"""
        self.plot_widget.plotItem.vb.scaleBy((1.25, 1.25))
        logger.info("Zoomed out")
    
    def zoom_to_rect(self, rect: QtCore.QRectF):
        """Zoom to specific rectangle
        
        Args:
            rect: Rectangle in data coordinates to zoom to
        """
        self.plot_widget.setXRange(rect.left(), rect.right(), padding=0)
        self.plot_widget.setYRange(rect.top(), rect.bottom(), padding=0)
    
    def toggle_magnifier(self, enabled: bool):
        """Toggle magnifier lens
        
        Args:
            enabled: Whether to enable magnifier
        """
        self.magnifier_enabled = enabled
        if not enabled:
            self.magnifier.hide()
        else:
            # Show magnifier hint
            logger.info("Magnifier enabled: Click to zoom in on an area")
        logger.info(f"Magnifier {'enabled' if enabled else 'disabled'}")
    
    def set_magnifier_zoom(self, zoom: float):
        """Set magnifier zoom factor
        
        Args:
            zoom: Zoom factor (1.5 to 10.0)
        """
        self.magnifier.set_zoom_factor(zoom)
    
    def set_magnifier_size(self, size: float):
        """Set magnifier lens size
        
        Args:
            size: Radius in pixels (30 to 150)
        """
        self.magnifier.set_radius(size)
    
    def export_image(self, filepath: str):
        """Export plot to image file
        
        Args:
            filepath: Output file path
        """
        try:
            exporter = ImageExporter(self.plot_widget.plotItem)
            exporter.export(filepath)
            logger.info(f"Plot exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export plot: {e}")
            return False
    
    def set_plot_size(self, width: int, height: int):
        """Set plot widget size
        
        Args:
            width: Width in pixels
            height: Height in pixels
        """
        self.plot_widget.setMinimumSize(width, height)
        self.plot_widget.setMaximumSize(width, height)
        logger.info(f"Plot size set to {width}x{height}")
    
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
            color_by: Column to use for coloring ('trackid', 'Annotation', or 'track_segments')
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
                    size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*self.colors[color_idx]),
                    name=f'Track {int(trackid)}',
                    hoverable=True,
                    hoverPen=pg.mkPen('yellow', width=2),
                    hoverBrush=pg.mkBrush(255, 255, 0, 150)
                )
                
                self.plot_widget.addItem(scatter)
                self.scatter_plots[trackid] = scatter
        
        elif color_by == 'track_segments' and 'Annotation' in df.columns:
            # Plot each track with segments colored by annotation
            # This shows different colored segments within the same track
            annotation_colors_used = set()
            
            for trackid in df['trackid'].unique():
                track_mask = df['trackid'] == trackid
                track_df_subset = df[track_mask].copy().reset_index(drop=False)
                
                # Get unique annotations for this track
                track_annotations = track_df_subset['Annotation'].unique()
                
                for annotation in track_annotations:
                    if annotation == 'invalid' or pd.isna(annotation):
                        continue
                    
                    # Mask for this specific annotation within this track
                    annotation_mask = track_df_subset['Annotation'] == annotation
                    
                    # Get the original DataFrame indices for these points
                    original_indices = track_df_subset.loc[annotation_mask, 'index'].values
                    
                    # Create boolean mask for the original dataframe
                    local_mask = df.index.isin(original_indices)
                    
                    # Use annotation-based color
                    color = get_annotation_color(annotation)
                    
                    # Create scatter plot for this segment
                    scatter = pg.ScatterPlotItem(
                        x=plot_x[local_mask],
                        y=plot_y[local_mask],
                        size=5,
                        pen=pg.mkPen(color, width=1),
                        brush=pg.mkBrush(*color),
                        name=f'{annotation[:20]}' if annotation not in annotation_colors_used else '',
                        hoverable=True,
                        hoverPen=pg.mkPen('yellow', width=2),
                        hoverBrush=pg.mkBrush(255, 255, 0, 150)
                    )
                    
                    self.plot_widget.addItem(scatter)
                    annotation_colors_used.add(annotation)
        
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
                    size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*color),
                    name=annotation[:25],  # Truncate long names
                    hoverable=True,
                    hoverPen=pg.mkPen('yellow', width=2),
                    hoverBrush=pg.mkBrush(255, 255, 0, 150)
                )
                
                self.plot_widget.addItem(scatter)
                annotation_legend[annotation] = color
    
    def on_mouse_clicked(self, event):
        """Handle mouse click for magnifier zoom
        
        Args:
            event: Mouse click event
        """
        if self.magnifier_enabled and event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Get click position in data coordinates
            scene_pos = event.scenePos()
            if self.plot_widget.plotItem.vb.sceneBoundingRect().contains(scene_pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                # Calculate zoom window around click point
                [[xmin, xmax], [ymin, ymax]] = self.plot_widget.viewRange()
                current_width = xmax - xmin
                current_height = ymax - ymin
                
                # Zoom in by factor (make view smaller around the point)
                zoom_factor = 0.5  # Zoom in 2x
                new_width = current_width * zoom_factor
                new_height = current_height * zoom_factor
                
                # Center new view on click point
                new_xmin = x - new_width / 2
                new_xmax = x + new_width / 2
                new_ymin = y - new_height / 2
                new_ymax = y + new_height / 2
                
                # Apply zoom
                self.plot_widget.setXRange(new_xmin, new_xmax, padding=0)
                self.plot_widget.setYRange(new_ymin, new_ymax, padding=0)
                
                logger.info(f"Magnifier zoom at ({x:.2f}, {y:.2f})")
    
    def on_mouse_moved(self, pos):
        """Handle mouse movement for tooltip display and magnifier
        
        Args:
            pos: Mouse position in scene coordinates
        """
        # Convert scene position to data coordinates
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Update magnifier position if enabled
        if self.magnifier_enabled:
            self.magnifier.setPos(x, y)
            self.magnifier.show()
        
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
    """Time series plot widget for multiple variables with interactive features"""
    
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
        
        # Enable mouse interactions
        self.altitude_plot.setMouseEnabled(x=True, y=True)
        self.speed_plot.setMouseEnabled(x=True, y=True)
        self.curvature_plot.setMouseEnabled(x=True, y=True)
        
        # Store plot items
        self.plot_items = {}
        
        # Interactive features
        self.view_history = PlotViewHistory()
        
        # Color map - matches application theme
        self.colors = [
            (231, 76, 60),      # Red (app theme)
            (46, 204, 113),     # Green (app theme)
            (52, 152, 219),     # Blue (app theme)
            (241, 196, 15),     # Yellow/Gold (app theme)
            (155, 89, 182),     # Purple (app theme)
            (26, 188, 156),     # Turquoise (app theme)
            (230, 126, 34),     # Orange (app theme)
            (236, 77, 177),     # Magenta (app theme)
        ]
    
    def zoom_in(self):
        """Zoom in all plots"""
        self.altitude_plot.vb.scaleBy((0.8, 0.8))
        self.speed_plot.vb.scaleBy((0.8, 0.8))
        self.curvature_plot.vb.scaleBy((0.8, 0.8))
    
    def zoom_out(self):
        """Zoom out all plots"""
        self.altitude_plot.vb.scaleBy((1.25, 1.25))
        self.speed_plot.vb.scaleBy((1.25, 1.25))
        self.curvature_plot.vb.scaleBy((1.25, 1.25))
    
    def reset_view(self):
        """Reset view to show all data"""
        self.altitude_plot.autoRange()
        self.speed_plot.autoRange()
        self.curvature_plot.autoRange()
    
    def export_image(self, filepath: str):
        """Export plots to image file
        
        Args:
            filepath: Output file path
        """
        try:
            exporter = ImageExporter(self.layout_widget.scene())
            exporter.export(filepath)
            logger.info(f"Time series plots exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export time series plots: {e}")
            return False
    
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
