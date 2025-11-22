# Quick Start: High Volume Training in GUI

## 🚀 Get Started in 3 Steps

### Prerequisites

Make sure you have the required packages installed:

```bash
pip install PyQt6 pyqtgraph pandas numpy tensorflow xgboost scikit-learn matplotlib
```

Or use the installation scripts:
```bash
# Linux/Mac
bash install_gui.sh

# Windows
install_gui.bat
```

---

## Launch the GUI

```bash
python -m src.gui
```

Or double-click `run.bat` (Windows)

---

## Complete Workflow Example

### Option A: Generate New High-Volume Dataset

**Step 1: Open High Volume Training Panel**
- In the GUI, select **"🚀 High Volume Training"** from the left menu

**Step 2: Generate Dataset**
- Set **Number of Tracks**: 200 (default is good)
- Set **Duration**: 10 minutes (default is good)
- Click **"Generate Dataset"**
- Wait ~30 seconds for 200 tracks
- ✓ Status shows: "Dataset generated: data/high_volume_simulation.csv"

**Step 3: Apply Auto-Labeling**
- Click **"Apply Auto-Labeling"**
- Wait ~20 seconds for labeling
- ✓ Status shows labeled data path and annotation summary

**Step 4: Train Models**
- Check which models to train:
  - ✓ **🧠 Transformer** (recommended)
  - ✓ **🔁 LSTM** (recommended)
  - ☐ **🚀 XGBoost** (optional)
- Click **"Train Selected Models"**
- Wait 5-15 minutes depending on models selected
- ✓ Results appear in summary table

**Step 5: View Results**
- Review the **Results Summary** table
- See which model performed best (🏆 indicator)
- Click **"📄 Generate Report"** for detailed HTML report
- Click **"💾 Export Results"** to save metrics as JSON

**Total Time: ~10-20 minutes** for complete workflow

---

### Option B: Use Existing High-Volume Data

If you already have high-volume data files:

**Step 1: Open High Volume Training Panel**
- Select **"🚀 High Volume Training"**

**Step 2: Load Existing Data**
- Click **"Or Select Existing CSV File"**
- Navigate to `data/` folder
- Select `high_volume_simulation_labeled.csv` (if already labeled)
- OR select `high_volume_simulation.csv` (if unlabeled)

**Step 3: Skip or Apply Labeling**
- If data is already labeled: Skip to Step 4
- If data is unlabeled: Click **"Apply Auto-Labeling"**

**Step 4: Train Models**
- Select models (Transformer + LSTM recommended)
- Click **"Train Selected Models"**
- Wait for training to complete

**Step 5: View Results**
- Review results table
- Generate report
- Export metrics

**Total Time: ~5-15 minutes** (no generation needed)

---

## Expected Results

### Typical Performance Metrics

For 200 tracks, 10 minutes each:

| Model | Train Acc | Test Acc | F1 Score | Time |
|-------|-----------|----------|----------|------|
| Transformer | 0.95-0.98 | 0.93-0.96 | 0.92-0.95 | 3-5 min |
| LSTM | 0.92-0.95 | 0.90-0.93 | 0.89-0.92 | 2-4 min |
| XGBoost | 0.88-0.92 | 0.85-0.90 | 0.84-0.89 | 1-2 min |

**Note**: Actual results depend on hardware, dataset diversity, and random initialization

---

## Screenshots Reference

### Panel Overview
```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Generate High-Volume Dataset                   │
│   Number of Tracks: [200    ▼]                         │
│   Duration (min):   [10.0   ▼]                         │
│   [Generate Dataset]                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 2: Apply Auto-Labeling                            │
│   [Or Select Existing CSV File]                        │
│   [Apply Auto-Labeling]                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 3: Train Models                                    │
│   [✓ 🧠 Transformer] [✓ 🔁 LSTM] [☐ 🚀 XGBoost]       │
│   [Train Selected Models]                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Results Summary                                         │
│ ┌───────────────────────────────────────────────────┐  │
│ │Model       │Train Acc│Test Acc│F1 Score│Time(s) │  │
│ ├────────────┼─────────┼────────┼────────┼────────┤  │
│ │TRANSFORMER │0.9654   │0.9521  │0.9489  │267.34  │  │
│ │LSTM        │0.9342   │0.9187  │0.9145  │189.21  │  │
│ └───────────────────────────────────────────────────┘  │
│   [📄 Generate Report] [💾 Export Results]            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Status Log:                                             │
│ ============================================================
│ Generating dataset: 200 tracks, 10.0 min each...
│ ✓ Dataset generated: data/high_volume_simulation.csv
│   Total tracks: 200
│   Total records: 120,000
│   Duration: 600.0s
│ 
│ ============================================================
│ Applying auto-labeling...
│ ✓ Auto-labeling completed
│   Valid records: 119,980/120,000
│   Unique annotations: 27
│ 
│ ============================================================
│ Training 2 model(s)...
│ ✓ Training completed for all models!
│ 
│   TRANSFORMER:
│     Train Acc: 0.9654
│     Test Acc:  0.9521
│     F1 Score:  0.9489
│     Time:      267.34s
│ 
│   LSTM:
│     Train Acc: 0.9342
│     Test Acc:  0.9187
│     F1 Score:  0.9145
│     Time:      189.21s
│ 
│ 🏆 Best model: TRANSFORMER
└─────────────────────────────────────────────────────────┘
```

---

## Tips for Best Results

### 1. Dataset Generation
- **Start small** (50 tracks) to test workflow
- **Scale up** (200-500 tracks) for production models
- **Longer duration** = more data points per track
- **More tracks** = better generalization

### 2. Model Selection
- **Transformer**: Best overall, handles complex patterns
- **LSTM**: Good balance of speed and accuracy
- **XGBoost**: Fastest, good for quick baselines

### 3. Time Management
- Generation: ~0.15s per track
- Labeling: ~0.1s per track
- Training:
  - Transformer: ~3-5 min (200 tracks)
  - LSTM: ~2-4 min (200 tracks)
  - XGBoost: ~1-2 min (200 tracks)

### 4. Hardware Recommendations
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, GPU (CUDA)
- **Optimal**: 32GB RAM, RTX 3060+ GPU

---

## Common Workflows

### Research & Experimentation
```
Generate (50 tracks, 5 min) → Label → Train Transformer → Analyze
                                           ↓
                                    Adjust parameters
                                           ↓
                                    Re-train until satisfied
```

### Production Model Training
```
Generate (500 tracks, 15 min) → Label → Train All Models → Compare
                                             ↓
                                      Select Best Model
                                             ↓
                                      Generate Report
                                             ↓
                                      Deploy Model
```

### Quick Baseline
```
Use Existing Data → Train XGBoost → Get Quick Results
```

### Comprehensive Analysis
```
Generate Large Dataset → Label → Train All 3 Models → Generate Report
                                        ↓
                                  Compare Performance
                                        ↓
                                  Export Results
                                        ↓
                                  Publish Findings
```

---

## Troubleshooting

### "GUI doesn't start"
```bash
# Check PyQt6 installation
pip install PyQt6 pyqtgraph

# Try running directly
python -m src.gui
```

### "Generation is slow"
- **Normal**: 200 tracks takes ~30 seconds
- **Speed up**: Reduce number of tracks or duration
- **Hardware**: Generation is CPU-bound

### "Training takes forever"
- **Check**: GPU is being used (if available)
- **Reduce**: Number of epochs in config
- **Alternative**: Train one model at a time

### "Out of memory"
- **Reduce**: Number of tracks
- **Reduce**: Batch size in config
- **Increase**: System RAM or use GPU

### "Models show poor accuracy"
- **Check**: Label diversity (should have 10+ unique annotations)
- **Increase**: Dataset size
- **Adjust**: Auto-labeling thresholds in AutoLabeling panel

---

## Next Steps

After completing your first high-volume training:

1. **📉 Visualization Panel**
   - View your data interactively
   - Explore trajectories in PPI view
   - Analyze time series patterns

2. **📈 Report Panel**
   - Generate detailed analysis reports
   - View confusion matrices
   - Examine classification reports

3. **🏷️ AutoLabeling Panel**
   - Fine-tune labeling parameters
   - Create custom annotation rules
   - Improve label diversity

4. **🤖 AI Tagging Panel**
   - Train individual models with custom parameters
   - Experiment with different architectures
   - Test on specific datasets

---

## Success Checklist

- [ ] GUI launches successfully
- [ ] Generated high-volume dataset (or loaded existing)
- [ ] Applied auto-labeling with good diversity (10+ annotations)
- [ ] Trained at least one model successfully
- [ ] Reviewed results in summary table
- [ ] Generated HTML report
- [ ] Exported results to JSON
- [ ] Understood model performance metrics

---

## Getting Help

1. **Status Log**: Check panel's status log for detailed errors
2. **Console Output**: Run GUI from terminal to see debug output
3. **Documentation**: Read `HIGH_VOLUME_GUI_INTEGRATION.md` for details
4. **Report Issues**: Check error messages and stack traces

---

## Files Generated

After successful workflow:

```
data/
├── high_volume_simulation.csv          (Raw data)
└── high_volume_simulation_labeled.csv  (Labeled data)

output/
├── models/
│   ├── transformer_highvolume/
│   │   ├── transformer_model.h5
│   │   ├── transformer_model_metadata.pkl
│   │   └── transformer_metrics.json
│   ├── lstm_highvolume/
│   │   ├── lstm_model.h5
│   │   ├── lstm_model_metadata.pkl
│   │   └── lstm_metrics.json
│   └── xgboost_highvolume/
│       ├── xgboost_model.pkl
│       └── xgboost_metrics.json
└── high_volume_training_report.html
```

---

**Ready to start?** Launch the GUI and select "🚀 High Volume Training"!

```bash
python -m src.gui
```

**Questions?** Check the detailed documentation in `HIGH_VOLUME_GUI_INTEGRATION.md`
