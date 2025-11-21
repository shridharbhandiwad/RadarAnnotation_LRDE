"""Main GUI Application using PyQt6"""
import sys
import os
from pathlib import Path
import pandas as pd
import logging
from typing import Optional

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget,
                                 QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                                 QProgressBar, QTableWidget, QTableWidgetItem, QSplitter,
                                 QListWidget, QStackedWidget, QMessageBox, QSlider, QSizePolicy)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6 import QtWidgets
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    logging.error("PyQt6 is not installed")
    
    # Create stub classes to prevent import errors
    # These are actual classes (not None) so inheritance won't fail
    class QThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
    
    class Qt:
        class Orientation:
            Vertical = None
        class StandardButton:
            Yes = None
            No = None
    
    def pyqtSignal(*args):
        return lambda: None
    
    # Create stub classes (actual classes, not None) for all Qt widgets
    class _QtStub:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    QApplication = QMainWindow = QWidget = QVBoxLayout = QHBoxLayout = _QtStub
    QPushButton = QLabel = QFileDialog = QTextEdit = QTabWidget = _QtStub
    QComboBox = QSpinBox = QDoubleSpinBox = QFormLayout = QGroupBox = _QtStub
    QProgressBar = QTableWidget = QTableWidgetItem = QSplitter = _QtStub
    QListWidget = QStackedWidget = QMessageBox = QSlider = QFont = QSizePolicy = _QtStub
    
    class QtWidgets:
        class QSizePolicy:
            class Policy:
                Preferred = None
                Expanding = None

# Import engines
from . import data_engine, autolabel_engine, ai_engine, report_engine, sim_engine
from .config import get_config, save_default_config
from .plotting import create_ppi_widget, create_timeseries_widget, HAS_PYQTGRAPH

logger = logging.getLogger(__name__)


class WorkerThread(QThread):
    """Worker thread for long-running tasks"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            logger.error(f"Worker thread error: {error_msg}", exc_info=True)
            self.error.emit(error_msg)


class DataExtractionPanel(QWidget):
    """Panel for Data Extraction Engine"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_df = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # File selection
        file_group = QGroupBox("Binary File Selection")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select Binary File")
        self.file_button.clicked.connect(self.select_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        file_group.setLayout(file_layout)
        
        # Extract button
        self.extract_button = QPushButton("Extract Data")
        self.extract_button.clicked.connect(self.extract_data)
        self.extract_button.setEnabled(False)
        
        # Save options
        save_group = QGroupBox("Save Options")
        save_layout = QFormLayout()
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['csv', 'xlsx'])
        save_layout.addRow("Format:", self.format_combo)
        
        self.save_button = QPushButton("Save Extracted Data")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)
        save_layout.addWidget(self.save_button)
        
        save_group.setLayout(save_layout)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        
        layout.addWidget(file_group)
        layout.addWidget(self.extract_button)
        layout.addWidget(save_group)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Binary File", "", "Binary Files (*.bin);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected: {Path(file_path).name}")
            self.extract_button.setEnabled(True)
            self.status_text.append(f"Selected file: {file_path}")
    
    def extract_data(self):
        try:
            self.status_text.append("Extracting data...")
            self.current_df = data_engine.extract_binary_to_dataframe(self.file_path)
            
            summary = data_engine.get_data_summary(self.current_df)
            self.status_text.append(f"✓ Extracted {summary['total_records']} records")
            self.status_text.append(f"  Tracks: {summary['num_tracks']}")
            self.status_text.append(f"  Duration: {summary['duration_seconds']:.2f}s")
            
            self.save_button.setEnabled(True)
            
        except Exception as e:
            self.status_text.append(f"✗ Error: {str(e)}")
            logger.error(f"Extraction error: {e}", exc_info=True)
    
    def save_data(self):
        if self.current_df is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "raw_data.csv", 
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                fmt = 'xlsx' if file_path.endswith('.xlsx') else 'csv'
                data_engine.save_dataframe(self.current_df, file_path, fmt)
                self.status_text.append(f"✓ Saved to: {file_path}")
            except Exception as e:
                self.status_text.append(f"✗ Save error: {str(e)}")


class AutoLabelingPanel(QWidget):
    """Panel for AutoLabeling Engine"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_df = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # File selection
        file_group = QGroupBox("Input Data")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_button = QPushButton("Select CSV File")
        self.file_button.clicked.connect(self.select_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        file_group.setLayout(file_layout)
        
        # Threshold adjustments
        threshold_group = QGroupBox("Threshold Settings")
        threshold_layout = QFormLayout()
        
        self.level_flight_spin = QDoubleSpinBox()
        self.level_flight_spin.setRange(0, 100)
        self.level_flight_spin.setValue(5.0)
        threshold_layout.addRow("Level Flight (m):", self.level_flight_spin)
        
        self.low_speed_spin = QDoubleSpinBox()
        self.low_speed_spin.setRange(0, 500)
        self.low_speed_spin.setValue(50.0)
        threshold_layout.addRow("Low Speed (m/s):", self.low_speed_spin)
        
        self.high_speed_spin = QDoubleSpinBox()
        self.high_speed_spin.setRange(0, 1000)
        self.high_speed_spin.setValue(200.0)
        threshold_layout.addRow("High Speed (m/s):", self.high_speed_spin)
        
        threshold_group.setLayout(threshold_layout)
        
        # Process button
        self.process_button = QPushButton("Run Auto-Labeling")
        self.process_button.clicked.connect(self.run_autolabeling)
        self.process_button.setEnabled(False)
        
        # Results
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['Annotation', 'Count', 'Percentage'])
        
        # Save button
        self.save_button = QPushButton("Save Labeled Data")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        
        layout.addWidget(file_group)
        layout.addWidget(threshold_group)
        layout.addWidget(self.process_button)
        layout.addWidget(QLabel("Annotation Results:"))
        layout.addWidget(self.results_table)
        layout.addWidget(self.save_button)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        
        self.setLayout(layout)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected: {Path(file_path).name}")
            self.process_button.setEnabled(True)
            self.status_text.append(f"Selected file: {file_path}")
    
    def run_autolabeling(self):
        try:
            self.status_text.append("Running auto-labeling...")
            
            # Load data
            df = pd.read_csv(self.file_path)
            
            # Compute features
            df = autolabel_engine.compute_motion_features(df)
            
            # Apply rules with custom thresholds
            rules_config = {
                'level_flight_threshold': self.level_flight_spin.value(),
                'low_speed_threshold': self.low_speed_spin.value(),
                'high_speed_threshold': self.high_speed_spin.value(),
                'curvature_threshold': 0.01,
                'light_maneuver_threshold': 2.0,
                'high_maneuver_threshold': 5.0,
                'range_rate_threshold': 1.0,
                'fixed_range_threshold': 10.0,
                'min_points_per_track': 3
            }
            
            self.current_df = autolabel_engine.apply_rules_and_flags(df, rules_config)
            
            # Get summary
            summary = autolabel_engine.get_annotation_summary(self.current_df)
            
            # Update results table
            self.results_table.setRowCount(0)
            for annotation, data in summary['annotation_distribution'].items():
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem(annotation))
                self.results_table.setItem(row, 1, QTableWidgetItem(str(data['count'])))
                self.results_table.setItem(row, 2, QTableWidgetItem(f"{data['percentage']:.2f}%"))
            
            self.status_text.append(f"✓ Processed {summary['valid_records']}/{summary['total_records']} records")
            self.save_button.setEnabled(True)
            
        except Exception as e:
            self.status_text.append(f"✗ Error: {str(e)}")
            logger.error(f"Auto-labeling error: {e}", exc_info=True)
    
    def save_data(self):
        if self.current_df is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Labeled Data", "labelled_data.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.current_df.to_csv(file_path, index=False)
                self.status_text.append(f"✓ Saved to: {file_path}")
            except Exception as e:
                self.status_text.append(f"✗ Save error: {str(e)}")


class AITaggingPanel(QWidget):
    """Panel for AI Tagging Engine"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(['xgboost', 'lstm', 'transformer'])
        model_layout.addRow("Model Type:", self.model_combo)
        
        model_group.setLayout(model_layout)
        
        # Data selection
        data_group = QGroupBox("Training Data")
        data_layout = QVBoxLayout()
        
        self.data_label = QLabel("No file selected")
        self.data_button = QPushButton("Select Labeled Data CSV")
        self.data_button.clicked.connect(self.select_data)
        
        data_layout.addWidget(self.data_label)
        data_layout.addWidget(self.data_button)
        data_group.setLayout(data_layout)
        
        # Train button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        layout.addWidget(model_group)
        layout.addWidget(data_group)
        layout.addWidget(self.train_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Training Results:"))
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
    
    def select_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Labeled Data", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.data_path = file_path
            self.data_label.setText(f"Selected: {Path(file_path).name}")
            self.train_button.setEnabled(True)
            self.results_text.append(f"Selected data: {file_path}")
    
    def train_model(self):
        model_name = self.model_combo.currentText()
        
        self.results_text.append(f"\nTraining {model_name} model...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.train_button.setEnabled(False)
        
        # Run training in worker thread
        self.worker = WorkerThread(
            ai_engine.train_model,
            model_name,
            self.data_path,
            'output/models'
        )
        self.worker.finished.connect(self.training_finished)
        self.worker.error.connect(self.training_error)
        self.worker.start()
    
    def training_finished(self, result):
        model, metrics = result
        
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        
        train_metrics = metrics['train']
        test_metrics = metrics['test']
        
        self.results_text.append(f"\n✓ Training completed!")
        self.results_text.append(f"Training Time: {train_metrics.get('training_time', 0):.2f}s")
        self.results_text.append(f"Train Accuracy: {train_metrics.get('train_accuracy', 0):.4f}")
        self.results_text.append(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        self.results_text.append(f"Test F1 Score: {test_metrics.get('f1_score', 0):.4f}")
        
        # Show multi-output metrics if available
        if train_metrics.get('multi_output', False):
            self.results_text.append(f"\nMulti-output Results:")
            if 'outputs' in test_metrics:
                for output_name, output_metrics in test_metrics['outputs'].items():
                    self.results_text.append(f"  {output_name}: Acc={output_metrics['accuracy']:.4f}, F1={output_metrics['f1_score']:.4f}")
        
        self.results_text.append(f"\nModel saved to: output/models/")
    
    def training_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.results_text.append(f"\n✗ Training error: {error_msg}")


class ReportPanel(QWidget):
    """Panel for Report Engine"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Folder selection
        folder_group = QGroupBox("Data Folder")
        folder_layout = QVBoxLayout()
        
        self.folder_label = QLabel("No folder selected")
        self.folder_button = QPushButton("Select Data Folder")
        self.folder_button.clicked.connect(self.select_folder)
        
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_button)
        folder_group.setLayout(folder_layout)
        
        # Generate button
        self.generate_button = QPushButton("Generate Report")
        self.generate_button.clicked.connect(self.generate_report)
        self.generate_button.setEnabled(False)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        
        layout.addWidget(folder_group)
        layout.addWidget(self.generate_button)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder_path:
            self.folder_path = folder_path
            self.folder_label.setText(f"Selected: {Path(folder_path).name}")
            self.generate_button.setEnabled(True)
            self.status_text.append(f"Selected folder: {folder_path}")
    
    def generate_report(self):
        try:
            self.status_text.append("\nGenerating report...")
            
            report_path = Path(self.folder_path) / "report.html"
            report_engine.generate_report(self.folder_path, str(report_path))
            
            self.status_text.append(f"✓ Report generated: {report_path}")
            
            # Ask to open report
            reply = QMessageBox.question(
                self, 'Open Report',
                'Report generated successfully! Open in browser?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                import webbrowser
                webbrowser.open(f'file://{report_path}')
            
        except Exception as e:
            self.status_text.append(f"✗ Error: {str(e)}")
            logger.error(f"Report generation error: {e}", exc_info=True)


class SettingsPanel(QWidget):
    """Panel for Application Settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Theme selection
        theme_group = QGroupBox("Theme Settings")
        theme_layout = QVBoxLayout()
        
        theme_label = QLabel("Select Application Theme:")
        theme_layout.addWidget(theme_label)
        
        # Theme buttons
        theme_buttons_layout = QHBoxLayout()
        
        self.black_theme_button = QPushButton("⚫ Black Theme")
        self.black_theme_button.setCheckable(True)
        self.black_theme_button.clicked.connect(lambda: self.apply_theme('black'))
        self.black_theme_button.setObjectName("themeButton")
        self.black_theme_button.setMinimumHeight(60)
        theme_buttons_layout.addWidget(self.black_theme_button)
        
        self.white_theme_button = QPushButton("⚪ White Theme")
        self.white_theme_button.setCheckable(True)
        self.white_theme_button.clicked.connect(lambda: self.apply_theme('white'))
        self.white_theme_button.setObjectName("themeButton")
        self.white_theme_button.setMinimumHeight(60)
        theme_buttons_layout.addWidget(self.white_theme_button)
        
        theme_layout.addLayout(theme_buttons_layout)
        
        # Theme preview/description
        self.theme_description = QTextEdit()
        self.theme_description.setReadOnly(True)
        self.theme_description.setMaximumHeight(120)
        self.theme_description.setPlainText(
            "Black Theme: Dark professional interface optimized for low-light environments.\n"
            "Perfect for defense and tactical applications.\n\n"
            "White Theme: Light clean interface for bright environments.\n"
            "Reduces eye strain in well-lit conditions."
        )
        theme_layout.addWidget(self.theme_description)
        
        theme_group.setLayout(theme_layout)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        
        layout.addWidget(theme_group)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Set initial theme button state
        self.update_theme_buttons()
    
    def update_theme_buttons(self):
        """Update theme button states based on current theme"""
        if hasattr(self.parent, 'current_theme'):
            current = self.parent.current_theme
            self.black_theme_button.setChecked(current == 'black')
            self.white_theme_button.setChecked(current == 'white')
    
    def apply_theme(self, theme_name):
        """Apply selected theme"""
        try:
            # Find main window and apply theme
            main_window = self.parent
            while main_window and not isinstance(main_window, MainWindow):
                main_window = main_window.parent()
            
            if main_window:
                main_window.set_theme(theme_name)
                self.update_theme_buttons()
                self.status_text.append(f"✓ Applied {theme_name.capitalize()} theme successfully")
            else:
                self.status_text.append("✗ Could not find main window")
                
        except Exception as e:
            self.status_text.append(f"✗ Error applying theme: {str(e)}")
            logger.error(f"Theme application error: {e}", exc_info=True)


class SimulationPanel(QWidget):
    """Panel for Simulation Engine"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Settings
        settings_group = QGroupBox("Simulation Settings")
        settings_layout = QFormLayout()
        
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 20)
        self.count_spin.setValue(10)
        settings_layout.addRow("Number of Simulations:", self.count_spin)
        
        settings_group.setLayout(settings_layout)
        
        # Output folder
        folder_group = QGroupBox("Output Folder")
        folder_layout = QVBoxLayout()
        
        self.folder_label = QLabel("data/simulations")
        self.folder_button = QPushButton("Select Output Folder")
        self.folder_button.clicked.connect(self.select_folder)
        
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_button)
        folder_group.setLayout(folder_layout)
        
        # Generate button
        self.generate_button = QPushButton("Generate Simulations")
        self.generate_button.clicked.connect(self.generate_simulations)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        
        layout.addWidget(settings_group)
        layout.addWidget(folder_group)
        layout.addWidget(self.generate_button)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        
        self.setLayout(layout)
        self.output_folder = "data/simulations"
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder = folder_path
            self.folder_label.setText(folder_path)
    
    def generate_simulations(self):
        try:
            count = self.count_spin.value()
            self.status_text.append(f"\nGenerating {count} simulations...")
            
            folders = sim_engine.create_simulation_folders(self.output_folder, count)
            
            self.status_text.append(f"✓ Generated {len(folders)} simulation folders:")
            for folder in folders:
                self.status_text.append(f"  - {folder}")
            
        except Exception as e:
            self.status_text.append(f"✗ Error: {str(e)}")
            logger.error(f"Simulation error: {e}", exc_info=True)


class VisualizationPanel(QWidget):
    """Panel for data visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_df = None
        self.selected_tracks = []  # For track filtering
        self.setup_ui()
    
    def set_theme(self, theme: str):
        """Update visualization theme
        
        Args:
            theme: Theme name ('white' or 'black')
        """
        if HAS_PYQTGRAPH and hasattr(self, 'ppi_widget'):
            self.ppi_widget.set_theme(theme)
        if HAS_PYQTGRAPH and hasattr(self, 'ts_widget'):
            self.ts_widget.set_theme(theme)
        # Re-plot if data is loaded
        if self.current_df is not None:
            self.update_visualization()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Top controls panel
        controls_layout = QHBoxLayout()
        
        # Load data button
        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.load_data)
        load_button.setObjectName("primaryButton")
        controls_layout.addWidget(load_button)
        
        if HAS_PYQTGRAPH:
            # Coordinate mode selector
            controls_layout.addWidget(QLabel("Display:"))
            self.coord_combo = QComboBox()
            self.coord_combo.addItems(['Radar View (Circular)', 'Cartesian (X, Y)', 'Polar (Range, Azimuth)'])
            self.coord_combo.setCurrentIndex(0)  # Set Radar View as default
            self.coord_combo.currentTextChanged.connect(self.update_visualization)
            controls_layout.addWidget(self.coord_combo)
            
            # Color by selector
            controls_layout.addWidget(QLabel("Color:"))
            self.color_combo = QComboBox()
            self.color_combo.addItems(['Track ID', 'Annotation', 'Track Segments (Colored by Annotation)'])
            self.color_combo.currentTextChanged.connect(self.update_visualization)
            controls_layout.addWidget(self.color_combo)
            
            # Track ID filter
            controls_layout.addWidget(QLabel("Track:"))
            self.track_filter = QComboBox()
            self.track_filter.addItem("All Tracks")
            self.track_filter.currentTextChanged.connect(self.update_visualization)
            controls_layout.addWidget(self.track_filter)
            
            # Add separator
            controls_layout.addWidget(QLabel(" | "))
            
            # Show time series toggle
            self.show_timeseries_checkbox = QPushButton("Time Series")
            self.show_timeseries_checkbox.setCheckable(True)
            self.show_timeseries_checkbox.setChecked(False)
            self.show_timeseries_checkbox.clicked.connect(self.toggle_timeseries)
            controls_layout.addWidget(self.show_timeseries_checkbox)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Interactive controls panel (new)
        if HAS_PYQTGRAPH:
            interactive_layout = QHBoxLayout()
            
            # Zoom controls - Icon only
            btn_zoom_in = QPushButton("➕")
            btn_zoom_in.setToolTip("Zoom In")
            btn_zoom_in.setMaximumWidth(40)
            btn_zoom_in.clicked.connect(self.zoom_in)
            btn_zoom_in.setObjectName("iconButton")
            interactive_layout.addWidget(btn_zoom_in)
            
            btn_zoom_out = QPushButton("➖")
            btn_zoom_out.setToolTip("Zoom Out")
            btn_zoom_out.setMaximumWidth(40)
            btn_zoom_out.clicked.connect(self.zoom_out)
            btn_zoom_out.setObjectName("iconButton")
            interactive_layout.addWidget(btn_zoom_out)
            
            btn_reset = QPushButton("🔄")
            btn_reset.setToolTip("Reset View")
            btn_reset.setMaximumWidth(40)
            btn_reset.clicked.connect(self.reset_view)
            btn_reset.setObjectName("iconButton")
            interactive_layout.addWidget(btn_reset)
            
            # History controls - Icon only
            interactive_layout.addWidget(QLabel(" | "))
            
            self.btn_undo = QPushButton("⬅")
            self.btn_undo.setToolTip("Undo")
            self.btn_undo.setMaximumWidth(40)
            self.btn_undo.clicked.connect(self.undo_view)
            self.btn_undo.setObjectName("iconButton")
            interactive_layout.addWidget(self.btn_undo)
            
            self.btn_redo = QPushButton("➡")
            self.btn_redo.setToolTip("Redo")
            self.btn_redo.setMaximumWidth(40)
            self.btn_redo.clicked.connect(self.redo_view)
            self.btn_redo.setObjectName("iconButton")
            interactive_layout.addWidget(self.btn_redo)
            
            # Magnifier controls - Icon only
            interactive_layout.addWidget(QLabel(" | "))
            
            self.magnifier_toggle = QPushButton("🔍")
            self.magnifier_toggle.setToolTip("Enable/Disable Magnifier")
            self.magnifier_toggle.setMaximumWidth(40)
            self.magnifier_toggle.setCheckable(True)
            self.magnifier_toggle.clicked.connect(self.toggle_magnifier)
            self.magnifier_toggle.setObjectName("iconButton")
            interactive_layout.addWidget(self.magnifier_toggle)
            
            # Export controls - Icon only
            interactive_layout.addWidget(QLabel(" | "))
            
            btn_export_ppi = QPushButton("💾")
            btn_export_ppi.setToolTip("Save PPI Plot")
            btn_export_ppi.setMaximumWidth(40)
            btn_export_ppi.clicked.connect(self.export_ppi_plot)
            btn_export_ppi.setObjectName("iconButton")
            interactive_layout.addWidget(btn_export_ppi)
            
            btn_export_ts = QPushButton("📊")
            btn_export_ts.setToolTip("Save Time Series")
            btn_export_ts.setMaximumWidth(40)
            btn_export_ts.clicked.connect(self.export_timeseries_plot)
            btn_export_ts.setObjectName("iconButton")
            interactive_layout.addWidget(btn_export_ts)
            
            interactive_layout.addStretch()
            layout.addLayout(interactive_layout)
        
        if HAS_PYQTGRAPH:
            # Get theme from main window
            main_window = self.window()
            theme = 'white'  # Default
            if hasattr(main_window, 'current_theme'):
                theme = main_window.current_theme
            
            # Create visualization widgets with theme
            self.ppi_widget = create_ppi_widget(self, theme)
            self.ts_widget = create_timeseries_widget(self, theme)
            
            # Add PPI plot (always visible)
            layout.addWidget(self.ppi_widget.get_widget())
            
            # Add time series widget (initially hidden)
            ts_container = self.ts_widget.get_widget()
            ts_container.setVisible(False)
            layout.addWidget(ts_container)
            self.ts_container = ts_container
        else:
            layout.addWidget(QLabel("PyQtGraph not available for visualization"))
        
        self.setLayout(layout)
    
    def toggle_timeseries(self):
        """Toggle visibility of time series charts"""
        if HAS_PYQTGRAPH and hasattr(self, 'ts_container'):
            is_visible = self.show_timeseries_checkbox.isChecked()
            self.ts_container.setVisible(is_visible)
            # Update button text
            if is_visible:
                self.show_timeseries_checkbox.setText("Hide Time Series Charts")
            else:
                self.show_timeseries_checkbox.setText("Show Time Series Charts")
    
    def update_visualization(self):
        """Update visualization based on current settings"""
        if self.current_df is not None and HAS_PYQTGRAPH:
            # Set coordinate mode
            coord_text = self.coord_combo.currentText()
            if 'Radar' in coord_text or 'Circular' in coord_text:
                coord_mode = 'polar_circular'
            elif 'Polar' in coord_text:
                coord_mode = 'polar'
            else:
                coord_mode = 'cartesian'
            self.ppi_widget.set_coordinate_mode(coord_mode)
            
            # Apply track filter
            df_to_plot = self.current_df
            selected_track = self.track_filter.currentText()
            if selected_track != "All Tracks":
                try:
                    track_id = int(selected_track.split()[-1])
                    df_to_plot = self.current_df[self.current_df['trackid'] == track_id]
                except (ValueError, IndexError):
                    pass
            
            # Determine color mode
            color_text = self.color_combo.currentText()
            if color_text == 'Track ID':
                color_by = 'trackid'
            elif color_text == 'Track Segments (Colored by Annotation)':
                color_by = 'track_segments'
            else:
                color_by = 'Annotation'
            
            self.ppi_widget.plot_tracks(df_to_plot, color_by=color_by)
            
            # Only update time series if visible
            if hasattr(self, 'ts_container') and self.ts_container.isVisible():
                self.ts_widget.plot_tracks(df_to_plot)
    
    def zoom_in(self):
        """Zoom in on PPI plot"""
        if HAS_PYQTGRAPH and hasattr(self, 'ppi_widget'):
            self.ppi_widget.zoom_in()
            if hasattr(self, 'ts_container') and self.ts_container.isVisible():
                self.ts_widget.zoom_in()
    
    def zoom_out(self):
        """Zoom out on PPI plot"""
        if HAS_PYQTGRAPH and hasattr(self, 'ppi_widget'):
            self.ppi_widget.zoom_out()
            if hasattr(self, 'ts_container') and self.ts_container.isVisible():
                self.ts_widget.zoom_out()
    
    def reset_view(self):
        """Reset view to default"""
        if HAS_PYQTGRAPH and hasattr(self, 'ppi_widget'):
            self.ppi_widget.reset_view()
            if hasattr(self, 'ts_container') and self.ts_container.isVisible():
                self.ts_widget.reset_view()
    
    def undo_view(self):
        """Undo view change"""
        if HAS_PYQTGRAPH and hasattr(self, 'ppi_widget'):
            self.ppi_widget.undo_view()
    
    def redo_view(self):
        """Redo view change"""
        if HAS_PYQTGRAPH and hasattr(self, 'ppi_widget'):
            self.ppi_widget.redo_view()
    
    def toggle_magnifier(self):
        """Toggle magnifier lens"""
        if HAS_PYQTGRAPH and hasattr(self, 'ppi_widget'):
            enabled = self.magnifier_toggle.isChecked()
            self.ppi_widget.toggle_magnifier(enabled)
    
    def export_ppi_plot(self):
        """Export PPI plot to image file"""
        if not HAS_PYQTGRAPH or not hasattr(self, 'ppi_widget'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PPI Plot", "ppi_plot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            success = self.ppi_widget.export_image(file_path)
            if success:
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to export plot")
    
    def export_timeseries_plot(self):
        """Export time series plot to image file"""
        if not HAS_PYQTGRAPH or not hasattr(self, 'ts_widget'):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Time Series Plot", "timeseries_plot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            success = self.ts_widget.export_image(file_path)
            if success:
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to export plot")
    
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "CSV Files (*.csv)"
        )
        
        if file_path and HAS_PYQTGRAPH:
            try:
                self.current_df = pd.read_csv(file_path)
                
                # Update track filter dropdown
                self.track_filter.clear()
                self.track_filter.addItem("All Tracks")
                if 'trackid' in self.current_df.columns:
                    for track_id in sorted(self.current_df['trackid'].unique()):
                        self.track_filter.addItem(f"Track {int(track_id)}")
                
                # Plot data with current color setting
                self.update_visualization()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Data Annotation Application")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize config
        config_path = "config/default_config.json"
        if not os.path.exists(config_path):
            save_default_config(config_path)
        self.config = get_config(config_path)
        
        # Set current theme (default to white)
        self.current_theme = self.config.get('theme', 'white') if hasattr(self.config, 'get') else 'white'
        
        self.setup_ui()
        self.apply_stylesheet()
    
    def setup_ui(self):
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel: Engine selector
        self.engine_list = QListWidget()
        self.engine_list.addItems([
            "📊 Data Extraction",
            "🏷️ AutoLabeling",
            "🤖 AI Tagging",
            "📈 Report",
            "🔬 Simulation",
            "📉 Visualization",
            "⚙️ Settings"
        ])
        self.engine_list.setMinimumWidth(220)
        self.engine_list.setMaximumWidth(280)
        self.engine_list.setSpacing(2)
        self.engine_list.currentRowChanged.connect(self.change_panel)
        
        # Set size policy to expand vertically
        self.engine_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        
        # Right panel: Stacked widget for engine panels
        self.stack = QStackedWidget()
        
        # Add panels
        self.stack.addWidget(DataExtractionPanel())
        self.stack.addWidget(AutoLabelingPanel())
        self.stack.addWidget(AITaggingPanel())
        self.stack.addWidget(ReportPanel())
        self.stack.addWidget(SimulationPanel())
        self.stack.addWidget(VisualizationPanel())
        self.stack.addWidget(SettingsPanel(self))
        
        # Layout
        main_layout.addWidget(self.engine_list)
        main_layout.addWidget(self.stack)
        
        central_widget.setLayout(main_layout)
        
        # Select first panel
        self.engine_list.setCurrentRow(0)
    
    def change_panel(self, index):
        self.stack.setCurrentIndex(index)
    
    def set_theme(self, theme_name):
        """Set application theme"""
        self.current_theme = theme_name
        
        # Save to config
        try:
            config_path = "config/default_config.json"
            config = get_config(config_path)
            config.set('theme', theme_name)
            config.save(config_path)
        except Exception as e:
            logger.error(f"Failed to save theme preference: {e}")
        
        # Apply stylesheet
        self.apply_stylesheet()
        
        # Update visualization panel theme if it exists
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if hasattr(widget, 'set_theme'):
                widget.set_theme(theme_name)
    
    def get_theme_stylesheet(self):
        """Get stylesheet for current theme"""
        if self.current_theme == 'white':
            return self.get_white_theme()
        else:
            return self.get_black_theme()
    
    def apply_stylesheet(self):
        """Apply current theme stylesheet"""
        stylesheet = self.get_theme_stylesheet()
        self.setStyleSheet(stylesheet)
    
    def get_black_theme(self):
        """Get Black (Dark) theme stylesheet"""
        return """
        /* Main Window - Dark slate background */
        QMainWindow {
            background-color: #2b3440;
        }
        
        /* List Widget (Engine Selector) - Slate mono-color */
        QListWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #1c2329, stop:1 #2b3440);
            color: #b8c5d6;
            border: none;
            border-right: 2px solid #4a5a6b;
            font-size: 14px;
            font-weight: 500;
            padding: 8px;
        }
        
        QListWidget::item {
            padding: 18px 15px;
            border-radius: 6px;
            margin: 4px 6px;
            min-height: 50px;
            border-left: 3px solid transparent;
        }
        
        QListWidget::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #4a5a6b, stop:1 #5a6b7d);
            color: #e8eef5;
            border-left: 3px solid #7a8a9b;
            font-weight: bold;
        }
        
        QListWidget::item:hover {
            background-color: rgba(74, 90, 107, 0.3);
            border-left: 3px solid #5a6b7d;
        }
        
        /* Group Boxes - Subtle slate container */
        QGroupBox {
            border: 2px solid #3d4a58;
            border-radius: 8px;
            margin-top: 16px;
            padding-top: 20px;
            font-weight: 600;
            font-size: 13px;
            background-color: #343e4c;
            color: #c5d1df;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 6px 12px;
            color: #d5dfe9;
            background-color: #3d4a58;
            border-radius: 4px;
            margin-left: 10px;
        }
        
        /* Push Buttons - Mono slate gradient */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #5a6b7d, stop:1 #4a5a6b);
            color: #e8eef5;
            border: 1px solid #3d4a58;
            border-radius: 6px;
            padding: 12px 24px;
            font-size: 13px;
            font-weight: 600;
            min-height: 36px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #6a7b8d, stop:1 #5a6b7d);
            border: 1px solid #7a8a9b;
        }
        
        QPushButton:pressed {
            background-color: #3d4a58;
            padding: 13px 23px 11px 25px;
        }
        
        QPushButton:disabled {
            background-color: #3d4a58;
            color: #5a6b7d;
            border: 1px solid #2b3440;
        }
        
        QPushButton#primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #6a7b8d, stop:1 #5a6b7d);
            border: 1px solid #7a8a9b;
        }
        
        QPushButton#primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #7a8a9b, stop:1 #6a7b8d);
            border: 1px solid #8a9aab;
        }
        
        QPushButton#iconButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #4a5a6b, stop:1 #3d4a58);
            color: #b8c5d6;
            border: 1px solid #2b3440;
            border-radius: 5px;
            padding: 8px;
            font-size: 16px;
            font-weight: 600;
            min-height: 32px;
            max-height: 32px;
            min-width: 36px;
            max-width: 36px;
        }
        
        QPushButton#iconButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #5a6b7d, stop:1 #4a5a6b);
            border: 1px solid #6a7b8d;
            color: #e8eef5;
        }
        
        QPushButton#iconButton:pressed {
            background-color: #2b3440;
            padding: 9px 7px 7px 9px;
        }
        
        QPushButton#iconButton:checked {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #7a8a9b, stop:1 #6a7b8d);
            border: 1px solid #8a9aab;
            color: #ffffff;
        }
        
        /* Labels - Light slate text */
        QLabel {
            color: #c5d1df;
            font-size: 13px;
            padding: 3px;
            font-weight: 500;
        }
        
        /* Text Edits - Dark input fields */
        QTextEdit {
            background-color: #242b34;
            border: 2px solid #3d4a58;
            border-radius: 5px;
            padding: 10px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            color: #b8c5d6;
            line-height: 1.4;
        }
        
        QTextEdit:focus {
            border: 2px solid #5a6b7d;
            background-color: #1c2329;
        }
        
        /* Combo Boxes */
        QComboBox {
            background-color: #343e4c;
            border: 2px solid #3d4a58;
            border-radius: 5px;
            padding: 8px 12px;
            min-height: 32px;
            color: #c5d1df;
            font-size: 13px;
        }
        
        QComboBox:hover {
            border: 2px solid #5a6b7d;
            background-color: #3d4a58;
        }
        
        QComboBox:focus {
            border: 2px solid #6a7b8d;
            background-color: #3d4a58;
        }
        
        QComboBox::drop-down {
            border: none;
            padding-right: 12px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #b8c5d6;
            margin-right: 8px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #343e4c;
            border: 2px solid #5a6b7d;
            border-radius: 5px;
            selection-background-color: #5a6b7d;
            selection-color: #e8eef5;
            padding: 4px;
            color: #c5d1df;
        }
        
        /* Spin Boxes */
        QSpinBox, QDoubleSpinBox {
            background-color: #343e4c;
            border: 2px solid #3d4a58;
            border-radius: 5px;
            padding: 8px;
            min-height: 32px;
            color: #c5d1df;
            font-size: 13px;
        }
        
        QSpinBox:hover, QDoubleSpinBox:hover {
            border: 2px solid #5a6b7d;
            background-color: #3d4a58;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #6a7b8d;
            background-color: #3d4a58;
        }
        
        /* Tables */
        QTableWidget {
            background-color: #343e4c;
            border: 2px solid #3d4a58;
            border-radius: 6px;
            gridline-color: #3d4a58;
            color: #c5d1df;
        }
        
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #3d4a58;
        }
        
        QTableWidget::item:selected {
            background-color: #5a6b7d;
            color: #e8eef5;
        }
        
        QTableWidget::item:hover {
            background-color: #3d4a58;
        }
        
        QHeaderView::section {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #3d4a58, stop:1 #2b3440);
            color: #d5dfe9;
            padding: 10px;
            border: none;
            border-right: 1px solid #2b3440;
            font-weight: 600;
            font-size: 13px;
        }
        
        QHeaderView::section:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #4a5a6b, stop:1 #3d4a58);
        }
        
        /* Progress Bar */
        QProgressBar {
            border: 2px solid #3d4a58;
            border-radius: 6px;
            text-align: center;
            background-color: #2b3440;
            color: #c5d1df;
            font-weight: 600;
            font-size: 12px;
            min-height: 24px;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #5a6b7d, stop:1 #7a8a9b);
            border-radius: 4px;
        }
        
        /* Scroll Bars */
        QScrollBar:vertical {
            background: #2b3440;
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical {
            background: #4a5a6b;
            border-radius: 6px;
            min-height: 30px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #5a6b7d;
        }
        
        QScrollBar::handle:vertical:pressed {
            background: #6a7b8d;
        }
        
        QScrollBar:horizontal {
            background: #2b3440;
            height: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal {
            background: #4a5a6b;
            border-radius: 6px;
            min-width: 30px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #5a6b7d;
        }
        
        QScrollBar::handle:horizontal:pressed {
            background: #6a7b8d;
        }
        
        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }
        
        /* Splitter */
        QSplitter::handle {
            background-color: #3d4a58;
        }
        
        QSplitter::handle:hover {
            background-color: #5a6b7d;
        }
        
        QSplitter::handle:horizontal {
            width: 2px;
        }
        
        QSplitter::handle:vertical {
            height: 2px;
        }
        
        /* Slider */
        QSlider::groove:horizontal {
            border: 2px solid #3d4a58;
            height: 6px;
            background: #2b3440;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #6a7b8d, stop:1 #5a6b7d);
            border: 2px solid #4a5a6b;
            width: 18px;
            height: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #7a8a9b, stop:1 #6a7b8d);
            border: 2px solid #5a6b7d;
            width: 20px;
            height: 20px;
            margin: -8px 0;
        }
        
        /* Theme buttons */
        QPushButton#themeButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #5a6b7d, stop:1 #4a5a6b);
            color: #e8eef5;
            border: 2px solid #3d4a58;
            border-radius: 8px;
            padding: 18px 24px;
            font-size: 15px;
            font-weight: 700;
            min-height: 60px;
        }
        
        QPushButton#themeButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #6a7b8d, stop:1 #5a6b7d);
            border: 2px solid #7a8a9b;
        }
        
        QPushButton#themeButton:checked {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #7a8a9b, stop:1 #6a7b8d);
            border: 3px solid #8a9aab;
            color: #ffffff;
        }
        """
    
    def get_white_theme(self):
        """Get White (Light) theme stylesheet"""
        return """
        /* Main Window - Light background */
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        /* List Widget (Engine Selector) - Light theme */
        QListWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #f5f5f5);
            color: #2b3440;
            border: none;
            border-right: 2px solid #d0d0d0;
            font-size: 14px;
            font-weight: 500;
            padding: 8px;
        }
        
        QListWidget::item {
            padding: 18px 15px;
            border-radius: 6px;
            margin: 4px 6px;
            min-height: 50px;
            border-left: 3px solid transparent;
        }
        
        QListWidget::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #e0e0e0, stop:1 #d5d5d5);
            color: #1c2329;
            border-left: 3px solid #5a6b7d;
            font-weight: bold;
        }
        
        QListWidget::item:hover {
            background-color: rgba(224, 224, 224, 0.5);
            border-left: 3px solid #b0b0b0;
        }
        
        /* Group Boxes - Light container */
        QGroupBox {
            border: 2px solid #d0d0d0;
            border-radius: 8px;
            margin-top: 16px;
            padding-top: 20px;
            font-weight: 600;
            font-size: 13px;
            background-color: #ffffff;
            color: #2b3440;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 6px 12px;
            color: #1c2329;
            background-color: #e8e8e8;
            border-radius: 4px;
            margin-left: 10px;
        }
        
        /* Push Buttons - Light gradient */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f0f0f0, stop:1 #e0e0e0);
            color: #2b3440;
            border: 1px solid #c0c0c0;
            border-radius: 6px;
            padding: 12px 24px;
            font-size: 13px;
            font-weight: 600;
            min-height: 36px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #f0f0f0);
            border: 1px solid #a0a0a0;
        }
        
        QPushButton:pressed {
            background-color: #d0d0d0;
            padding: 13px 23px 11px 25px;
        }
        
        QPushButton:disabled {
            background-color: #f0f0f0;
            color: #b0b0b0;
            border: 1px solid #e0e0e0;
        }
        
        QPushButton#primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #f5f5f5);
            border: 2px solid #5a6b7d;
            color: #1c2329;
        }
        
        QPushButton#primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f5f5f5, stop:1 #e8e8e8);
            border: 2px solid #4a5a6b;
        }
        
        QPushButton#iconButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f8f8f8, stop:1 #e8e8e8);
            color: #2b3440;
            border: 1px solid #c0c0c0;
            border-radius: 5px;
            padding: 8px;
            font-size: 16px;
            font-weight: 600;
            min-height: 32px;
            max-height: 32px;
            min-width: 36px;
            max-width: 36px;
        }
        
        QPushButton#iconButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #f0f0f0);
            border: 1px solid #a0a0a0;
            color: #1c2329;
        }
        
        QPushButton#iconButton:pressed {
            background-color: #d8d8d8;
            padding: 9px 7px 7px 9px;
        }
        
        QPushButton#iconButton:checked {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #e0e0e0, stop:1 #d0d0d0);
            border: 1px solid #7a8a9b;
            color: #1c2329;
        }
        
        /* Labels - Dark text */
        QLabel {
            color: #2b3440;
            font-size: 13px;
            padding: 3px;
            font-weight: 500;
        }
        
        /* Text Edits - Light input fields */
        QTextEdit {
            background-color: #ffffff;
            border: 2px solid #d0d0d0;
            border-radius: 5px;
            padding: 10px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            color: #2b3440;
            line-height: 1.4;
        }
        
        QTextEdit:focus {
            border: 2px solid #5a6b7d;
            background-color: #fafafa;
        }
        
        /* Combo Boxes */
        QComboBox {
            background-color: #ffffff;
            border: 2px solid #d0d0d0;
            border-radius: 5px;
            padding: 8px 12px;
            min-height: 32px;
            color: #2b3440;
            font-size: 13px;
        }
        
        QComboBox:hover {
            border: 2px solid #a0a0a0;
            background-color: #fafafa;
        }
        
        QComboBox:focus {
            border: 2px solid #5a6b7d;
            background-color: #fafafa;
        }
        
        QComboBox::drop-down {
            border: none;
            padding-right: 12px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #2b3440;
            margin-right: 8px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            border: 2px solid #a0a0a0;
            border-radius: 5px;
            selection-background-color: #e0e0e0;
            selection-color: #1c2329;
            padding: 4px;
            color: #2b3440;
        }
        
        /* Spin Boxes */
        QSpinBox, QDoubleSpinBox {
            background-color: #ffffff;
            border: 2px solid #d0d0d0;
            border-radius: 5px;
            padding: 8px;
            min-height: 32px;
            color: #2b3440;
            font-size: 13px;
        }
        
        QSpinBox:hover, QDoubleSpinBox:hover {
            border: 2px solid #a0a0a0;
            background-color: #fafafa;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #5a6b7d;
            background-color: #fafafa;
        }
        
        /* Tables */
        QTableWidget {
            background-color: #ffffff;
            border: 2px solid #d0d0d0;
            border-radius: 6px;
            gridline-color: #e0e0e0;
            color: #2b3440;
        }
        
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        QTableWidget::item:selected {
            background-color: #e0e0e0;
            color: #1c2329;
        }
        
        QTableWidget::item:hover {
            background-color: #f0f0f0;
        }
        
        QHeaderView::section {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f5f5f5, stop:1 #e8e8e8);
            color: #1c2329;
            padding: 10px;
            border: none;
            border-right: 1px solid #d0d0d0;
            font-weight: 600;
            font-size: 13px;
        }
        
        QHeaderView::section:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #f0f0f0);
        }
        
        /* Progress Bar */
        QProgressBar {
            border: 2px solid #d0d0d0;
            border-radius: 6px;
            text-align: center;
            background-color: #f5f5f5;
            color: #2b3440;
            font-weight: 600;
            font-size: 12px;
            min-height: 24px;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #5a6b7d, stop:1 #7a8a9b);
            border-radius: 4px;
        }
        
        /* Scroll Bars */
        QScrollBar:vertical {
            background: #f5f5f5;
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 6px;
            min-height: 30px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
        
        QScrollBar::handle:vertical:pressed {
            background: #808080;
        }
        
        QScrollBar:horizontal {
            background: #f5f5f5;
            height: 12px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal {
            background: #c0c0c0;
            border-radius: 6px;
            min-width: 30px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #a0a0a0;
        }
        
        QScrollBar::handle:horizontal:pressed {
            background: #808080;
        }
        
        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }
        
        /* Splitter */
        QSplitter::handle {
            background-color: #d0d0d0;
        }
        
        QSplitter::handle:hover {
            background-color: #b0b0b0;
        }
        
        QSplitter::handle:horizontal {
            width: 2px;
        }
        
        QSplitter::handle:vertical {
            height: 2px;
        }
        
        /* Slider */
        QSlider::groove:horizontal {
            border: 2px solid #d0d0d0;
            height: 6px;
            background: #f5f5f5;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #f0f0f0, stop:1 #e0e0e0);
            border: 2px solid #b0b0b0;
            width: 18px;
            height: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #f0f0f0);
            border: 2px solid #909090;
            width: 20px;
            height: 20px;
            margin: -8px 0;
        }
        
        /* Theme buttons */
        QPushButton#themeButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #f0f0f0);
            color: #2b3440;
            border: 2px solid #d0d0d0;
            border-radius: 8px;
            padding: 18px 24px;
            font-size: 15px;
            font-weight: 700;
            min-height: 60px;
        }
        
        QPushButton#themeButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #fafafa, stop:1 #e8e8e8);
            border: 2px solid #b0b0b0;
        }
        
        QPushButton#themeButton:checked {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #e8e8e8, stop:1 #d0d0d0);
            border: 3px solid #5a6b7d;
            color: #1c2329;
        }
        """


def main():
    """Main entry point"""
    if not HAS_PYQT6:
        # Force flush to ensure error message is visible
        error_msg = """
================================================================================
 ERROR: PyQt6 is not installed!
================================================================================

The GUI application requires PyQt6 and PyQtGraph to run.

To install the required packages, run one of the following commands:

  Option 1 - Install GUI packages only (Recommended):
    pip install PyQt6 pyqtgraph

  Option 2 - Install all project requirements:
    pip install -r requirements.txt

  Option 3 - Install from conda (if using Anaconda):
    conda install -c conda-forge pyqt
    pip install pyqtgraph

================================================================================
After installation, run the GUI again with:
  python -m src.gui
  
  OR double-click:
  run.bat (Windows)
  run.sh (Linux/Mac)
================================================================================
"""
        print(error_msg, flush=True)
        sys.stderr.write(error_msg)
        sys.stderr.flush()
        
        # Wait for user acknowledgment on Windows
        if sys.platform == 'win32':
            input("\nPress Enter to exit...")
        
        sys.exit(1)
    
    try:
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        error_msg = f"\n\nFATAL ERROR: Failed to start GUI application!\n\nError: {str(e)}\n\n"
        print(error_msg, flush=True)
        sys.stderr.write(error_msg)
        sys.stderr.flush()
        
        if sys.platform == 'win32':
            input("Press Enter to exit...")
        
        logger.error(f"GUI startup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
