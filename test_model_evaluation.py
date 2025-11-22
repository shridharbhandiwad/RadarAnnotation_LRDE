"""Test script for model evaluation functionality"""
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import ai_engine, sim_engine, autolabel_engine

def test_model_evaluation():
    """Test the model evaluation feature end-to-end"""
    print("=" * 80)
    print("Testing Model Evaluation Feature")
    print("=" * 80)
    
    # Step 1: Generate test data
    print("\n[1/5] Generating test simulation data...")
    test_data_path = "data/test_evaluation_sim.csv"
    csv_path = sim_engine.create_large_training_dataset(
        output_path=test_data_path,
        n_tracks=20,
        duration_min=2.0
    )
    print(f"✓ Generated test data: {csv_path}")
    
    # Step 2: Auto-label the data
    print("\n[2/5] Auto-labeling test data...")
    df = pd.read_csv(csv_path)
    df = autolabel_engine.compute_motion_features(df)
    df = autolabel_engine.apply_rules_and_flags(df)
    labeled_path = "data/test_evaluation_labeled.csv"
    df.to_csv(labeled_path, index=False)
    print(f"✓ Labeled data saved: {labeled_path}")
    
    # Step 3: Train a simple model
    print("\n[3/5] Training a test model (Random Forest)...")
    model, metrics = ai_engine.train_model(
        'random_forest',
        labeled_path,
        'output/test_model',
        auto_transform=True
    )
    model_path = "output/test_model/random_forest_model.pkl"
    print(f"✓ Model trained and saved: {model_path}")
    print(f"  Train accuracy: {metrics['train'].get('train_accuracy', 0):.4f}")
    print(f"  Test accuracy: {metrics['test'].get('accuracy', 0):.4f}")
    
    # Step 4: Generate new unlabeled data for prediction
    print("\n[4/5] Generating new unlabeled data...")
    new_data_path = "data/test_evaluation_new.csv"
    new_csv = sim_engine.create_large_training_dataset(
        output_path=new_data_path,
        n_tracks=5,
        duration_min=1.0
    )
    print(f"✓ Generated new data: {new_csv}")
    
    # Step 5: Use the model to predict labels
    print("\n[5/5] Evaluating model on new data...")
    output_path = "data/test_evaluation_predicted.csv"
    df_predicted = ai_engine.predict_and_label(
        model_path,
        new_csv,
        output_path
    )
    print(f"✓ Predictions completed: {output_path}")
    
    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS SUMMARY")
    print("=" * 80)
    annotation_counts = df_predicted['Annotation'].value_counts()
    total = len(df_predicted)
    print(f"\nTotal records: {total:,}")
    print(f"Unique labels: {len(annotation_counts)}")
    print("\nTop 5 predicted labels:")
    for i, (label, count) in enumerate(annotation_counts.head(5).items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {i+1}. {label}: {count:,} ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nYou can now use the GUI 'Model Evaluation' panel to:")
    print("  1. Select a trained model (e.g., output/test_model/random_forest_model.pkl)")
    print("  2. Select input data (any CSV with trajectory data)")
    print("  3. Click 'Predict and Auto-Label' to generate predictions")
    print("\nGenerated files:")
    print(f"  - Test model: {model_path}")
    print(f"  - Sample unlabeled data: {new_csv}")
    print(f"  - Predicted results: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_model_evaluation()
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
