---
# AI Engine Architecture (Project)

```mermaid
flowchart LR
  %% UI / Orchestration
  subgraph UI[Desktop UI]
    GUI[src/gui.py\nPyQt6 GUI]
  end

  %% Data + feature/label pipelines
  subgraph Data[Data & Label Pipeline (Python)]
    BIN[(Radar binary\n*.bin)] --> DE[src/data_engine.py\nextract_binary_to_dataframe()]
    CSV[(Tabular\n*.csv / *.xlsx)] --> DL[src/data_engine.py\nload_dataframe()]

    DE --> DF[(pandas DataFrame\ntrackid,time,x..az)]
    DL --> DF

    DF --> FE[src/autolabel_engine.py\ncompute_motion_features()]
    FE --> RU[src/autolabel_engine.py\napply_rules_and_flags()]
    RU --> ANN[(Annotation\ncomposite tags + valid_features)]

    ANN --> LT[src/label_transformer.py\nLabelTransformer]
    LT --> MLDS[(Training-ready dataset\n(single label or multi-label))]
  end

  %% Training / evaluation
  subgraph Train[Model Training & Evaluation (Python)]
    ENG[src/ai_engine.py\nXGBoost/RF + LSTM/Transformer]\n
    MLDS --> ENG
    ENG --> PKL[(Tree models\njoblib/pickle *.pkl)]
    ENG --> H5[(Neural nets\nKeras *.h5)]

    ENG --> META[(Metadata\nscaler/classes/features\n+ metrics JSON)]
  end

  %% Export formats
  subgraph Export[Packaging for Deployment]
    PKL --> ONNX[export_models_to_onnx.py\n*.pkl → *.onnx]
    H5 --> TFL[convert_model_to_tflite.py\nKeras → *.tflite]
    META --> MJSON[model_metadata.json]
  end

  %% C++ runtime
  subgraph Cpp[C++ Real-time Inference]
    RT[cpp_inference/\nradar_tagger_multioutput\n(TFLite / ONNX Runtime)]
    TFL --> RT
    ONNX --> RT
    MJSON --> RT

    RT --> OUT[(Predicted 11 tags\n+ aggregated composite label\n+ latency/throughput)]
  end

  %% GUI entry points
  GUI -->|Extract / Load| Data
  GUI -->|Auto-label + visualize| Data
  GUI -->|Train / Evaluate| Train
  GUI -->|Convert / Build / Run| Export
  GUI --> Cpp
```

## What this slide is saying

- **Python is the control plane**: the GUI (`src/gui.py`) orchestrates ingestion, labeling, training, evaluation, and deployment.
- **Label model is multi-output**: the system predicts **11 binary tags** (direction / vertical motion / path shape / maneuver intensity / speed) and then assembles an aggregated composite label.
- **Two deployment tracks**:
  - **Neural nets**: Keras (`*.h5`) → **TFLite** (`*.tflite`) via `convert_model_to_tflite.py` → C++ (`cpp_inference/`).
  - **Tree models**: pickled models (`*.pkl`) → **ONNX** via `export_models_to_onnx.py` (see `cpp_inference/ONNX_EXPORT_GUIDE.md`) → C++.
- **Metadata is first-class**: `model_metadata.json` carries feature layout + scaling + label info so the C++ runtime can reproduce the Python preprocessing assumptions.

---
