# Deployment Guide - Radar Data Annotation Application

## For Windows Users

### Prerequisites
- Windows 10 or 11
- Python 3.10 or higher ([Download](https://www.python.org/downloads/))
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space

### Installation Steps

1. **Install Python**
   - Download Python 3.10+ from python.org
   - During installation, check "Add Python to PATH"
   - Verify: Open Command Prompt and type `python --version`

2. **Extract Application**
   - Extract the radar-annotator folder to `C:\radar-annotator`
   - Open Command Prompt
   - Navigate: `cd C:\radar-annotator`

3. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```
   
   This may take 5-10 minutes depending on your internet speed.

4. **Verify Installation**
   ```cmd
   python verify_installation.py
   ```

5. **Run Application**
   ```cmd
   run.bat
   ```
   
   Or directly:
   ```cmd
   python -m src.gui
   ```

### Quick Test
```cmd
REM Generate sample data
python -m src.sim_engine --outdir data/simulations --count 3

REM Launch GUI
run.bat
```

### Common Windows Issues

**"python not recognized"**
- Add Python to PATH: System Properties → Environment Variables
- Or reinstall Python with "Add to PATH" checked

**SSL Certificate errors**
- Run: `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`

**TensorFlow installation issues**
- LSTM model is optional - use XGBoost instead
- Or install TensorFlow separately: `pip install tensorflow`

## For Linux Users

### Prerequisites
- Ubuntu 20.04+ / Debian 11+ / Fedora 35+ / CentOS 8+
- Python 3.10 or higher
- 4GB RAM minimum
- 500MB free disk space

### Installation Steps

1. **Install Python 3.10+**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.10 python3-pip python3-venv
   
   # Fedora
   sudo dnf install python3.10 python3-pip
   
   # Arch
   sudo pacman -S python python-pip
   ```

2. **Extract and Setup**
   ```bash
   cd ~/
   unzip radar-annotator.zip  # or extract from archive
   cd radar-annotator
   
   # Create virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python verify_installation.py
   ```

4. **Run Application**
   ```bash
   ./run.sh
   ```
   
   Or:
   ```bash
   python -m src.gui
   ```

### Quick Test
```bash
# Generate sample data
python -m src.sim_engine --outdir data/simulations --count 3

# Run demo
./demo.sh
```

### Common Linux Issues

**Missing system libraries for PyQt6**
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libegl1 libxkbcommon-x11-0

# Fedora
sudo dnf install mesa-libGL libxkbcommon-x11
```

**Permission denied on scripts**
```bash
chmod +x run.sh demo.sh verify_installation.py
```

## For macOS Users

### Prerequisites
- macOS 11 (Big Sur) or higher
- Python 3.10+ (install via Homebrew or python.org)
- 4GB RAM minimum

### Installation Steps

1. **Install Python via Homebrew**
   ```bash
   # Install Homebrew if not present
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Python
   brew install python@3.10
   ```

2. **Setup Application**
   ```bash
   cd ~/Downloads/radar-annotator
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Verify and Run**
   ```bash
   python verify_installation.py
   ./run.sh
   ```

### macOS Issues

**"Application is damaged" error**
```bash
xattr -cr radar-annotator/
```

**GUI doesn't start**
- Install XQuartz: `brew install --cask xquartz`
- Restart terminal

## Docker Deployment (All Platforms)

For a consistent environment across platforms:

```dockerfile
# Dockerfile (create this file)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run verification
RUN python verify_installation.py

CMD ["python", "-m", "src.gui"]
```

Build and run:
```bash
docker build -t radar-annotator .
docker run -v $(pwd)/data:/app/data radar-annotator
```

## Conda Environment (All Platforms)

For users preferring Conda:

```bash
# Create environment
conda create -n radar-annotator python=3.10
conda activate radar-annotator

# Install dependencies
pip install -r requirements.txt

# Verify
python verify_installation.py

# Run
python -m src.gui
```

Create `environment.yml`:
```yaml
name: radar-annotator
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy>=1.24
  - pandas>=2.0
  - scipy>=1.10
  - matplotlib>=3.7
  - scikit-learn>=1.3
  - pip
  - pip:
    - PyQt6>=6.5
    - pyqtgraph>=0.13
    - xgboost>=1.7
    - tensorflow>=2.13
    - openpyxl>=3.1
    - joblib>=1.3
```

Install with:
```bash
conda env create -f environment.yml
conda activate radar-annotator
```

## Network Deployment (Server Mode)

For remote access or multiple users:

1. **Setup on Server**
   ```bash
   # On server
   cd /opt/radar-annotator
   pip install -r requirements.txt
   ```

2. **Create Service (Linux)**
   ```bash
   # /etc/systemd/system/radar-annotator.service
   [Unit]
   Description=Radar Annotation Service
   After=network.target
   
   [Service]
   Type=simple
   User=radar
   WorkingDirectory=/opt/radar-annotator
   ExecStart=/usr/bin/python3 -m src.gui
   Restart=on-failure
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable:
   ```bash
   sudo systemctl enable radar-annotator
   sudo systemctl start radar-annotator
   ```

3. **Access Remotely**
   - Use X11 forwarding: `ssh -X user@server`
   - Or use VNC/RDP for GUI access

## Production Deployment Checklist

- [ ] Python 3.10+ installed
- [ ] All dependencies installed from requirements.txt
- [ ] verify_installation.py passes
- [ ] Default config generated (config/default_config.json)
- [ ] Directories created (data/, output/)
- [ ] Write permissions for data/ and output/
- [ ] GUI launches successfully
- [ ] Demo runs without errors
- [ ] Unit tests pass (pytest tests/)

## Performance Tuning

### For Large Datasets
```json
// config/default_config.json
{
  "ml_params": {
    "xgboost": {
      "n_jobs": -1,  // Use all CPU cores
      "tree_method": "hist"  // Faster training
    }
  }
}
```

### For Limited Memory
- Process data in chunks
- Use XGBoost instead of LSTM
- Limit visualization to first 10,000 points

### For GPU Acceleration (TensorFlow)
```bash
pip install tensorflow[and-cuda]
```

## Backup and Data Management

### Backup Critical Files
```bash
# Backup config
cp -r config/ config_backup/

# Backup models
cp -r output/models/ models_backup/

# Backup processed data
tar -czf data_backup.tar.gz data/ output/
```

### Data Organization
```
project/
├── data/
│   ├── raw/           # Original binary files
│   ├── simulations/   # Generated test data
│   └── processed/     # Extracted CSVs
├── output/
│   ├── models/        # Trained ML models
│   ├── reports/       # Generated reports
│   └── labeled/       # Labeled datasets
```

## Security Considerations

- Application runs locally by default (no network exposure)
- Binary files are parsed with struct (safe)
- No external data transmission
- User data stays on local machine

For enterprise deployment:
- Set up firewall rules
- Use encrypted storage for sensitive data
- Implement access controls
- Regular security updates

## Support and Maintenance

### Update Application
```bash
# Pull latest changes (if using git)
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Verify
python verify_installation.py
```

### Logs Location
- Application logs: Check terminal output
- Error logs: Python exceptions in console
- ML training logs: output/models/

### Getting Help
1. Check README.md
2. Review QUICK_START.md
3. Run verify_installation.py
4. Check logs for error messages

---

**Deployment Time Estimate**
- Basic installation: 10-15 minutes
- With verification: 20 minutes
- Docker deployment: 30 minutes
- Production setup: 1-2 hours
