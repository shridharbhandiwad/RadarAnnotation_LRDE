# QThread Crash Fix - Model Conversion

## Problem
The application was crashing during XGBoost model to TFLite conversion with the error:
```
QThread: Destroyed while thread '' is still running
```

## Root Cause
In the C++ Deployment Panel (`src/gui.py`), three methods were creating `WorkerThread` objects as local variables:
1. `convert_model()` - line 2096
2. `build_cpp_app()` - line 2205  
3. `evaluate_model()` - line 2286

When these methods completed execution, the local `worker` variable went out of scope and was garbage collected by Python **while the thread was still running**. This caused Qt to raise the "QThread: Destroyed while thread is still running" error and crash the application.

## Solution
Changed all three instances from local variable to instance variable:

**Before:**
```python
worker = WorkerThread(task_function)
worker.finished.connect(self.on_complete)
worker.error.connect(self.on_error)
worker.start()
```

**After:**
```python
self.worker = WorkerThread(task_function)
self.worker.finished.connect(self.on_complete)
self.worker.error.connect(self.on_error)
self.worker.start()
```

## Changes Made
1. **Line 2096** (`convert_model` method): Changed `worker =` to `self.worker =`
2. **Line 2205** (`build_cpp_app` method): Changed `worker =` to `self.worker =`
3. **Line 2286** (`evaluate_model` method): Changed `worker =` to `self.worker =`

## Why This Works
By storing the thread as an instance variable (`self.worker`), the thread object stays alive for the lifetime of the panel object, preventing premature garbage collection. The thread can complete its work safely and clean up properly when finished.

## Verification
- All other WorkerThread usages in the codebase already use `self.worker` correctly
- Python syntax validation passed
- No other instances of this bug pattern were found

## Impact
This fix resolves the crash during:
- Model conversion to TFLite
- C++ application builds
- Model evaluation runs

The application will now properly wait for background threads to complete before allowing cleanup.
