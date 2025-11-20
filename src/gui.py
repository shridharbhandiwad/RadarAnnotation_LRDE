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
                                 QListWidget, QStackedWidget, QMessageBox, QSlider)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont
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
    QListWidget = QStackedWidget = QMessageBox = QSlider = QFont = _QtStub

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
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Load data button
        load_button = QPushButton("Load Data for Visualization")
        load_button.clicked.connect(self.load_data)
        layout.addWidget(load_button)
        
        if HAS_PYQTGRAPH:
            # Create visualization widgets
            self.ppi_widget = create_ppi_widget(self)
            self.ts_widget = create_timeseries_widget(self)
            
            # Add to layout
            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.addWidget(self.ppi_widget.get_widget())
            splitter.addWidget(self.ts_widget.get_widget())
            splitter.setSizes([400, 600])
            
            layout.addWidget(splitter)
        else:
            layout.addWidget(QLabel("PyQtGraph not available for visualization"))
        
        self.setLayout(layout)
    
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "CSV Files (*.csv)"
        )
        
        if file_path and HAS_PYQTGRAPH:
            try:
                self.current_df = pd.read_csv(file_path)
                
                # Plot data
                self.ppi_widget.plot_tracks(self.current_df)
                self.ts_widget.plot_tracks(self.current_df)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Data Annotation Application")
        self.setGeometry(100, 100, 1400, 900)
        
        self.setup_ui()
        
        # Initialize config
        config_path = "config/default_config.json"
        if not os.path.exists(config_path):
            save_default_config(config_path)
        get_config(config_path)
    
    def setup_ui(self):
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel: Engine selector
        self.engine_list = QListWidget()
        self.engine_list.addItems([
            "Data Extraction",
            "AutoLabeling",
            "AI Tagging",
            "Report",
            "Simulation",
            "Visualization"
        ])
        self.engine_list.setMaximumWidth(200)
        self.engine_list.currentRowChanged.connect(self.change_panel)
        
        # Right panel: Stacked widget for engine panels
        self.stack = QStackedWidget()
        
        # Add panels
        self.stack.addWidget(DataExtractionPanel())
        self.stack.addWidget(AutoLabelingPanel())
        self.stack.addWidget(AITaggingPanel())
        self.stack.addWidget(ReportPanel())
        self.stack.addWidget(SimulationPanel())
        self.stack.addWidget(VisualizationPanel())
        
        # Layout
        main_layout.addWidget(self.engine_list)
        main_layout.addWidget(self.stack)
        
        central_widget.setLayout(main_layout)
        
        # Select first panel
        self.engine_list.setCurrentRow(0)
    
    def change_panel(self, index):
        self.stack.setCurrentIndex(index)


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
