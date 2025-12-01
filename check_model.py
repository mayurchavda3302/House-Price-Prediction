import joblib
import os

# Path to the model
MODEL_PATH = "house_price_model.joblib"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("❌ Error: Model file not found!")
    print("Please train the model first by running 'python train_improved_model.py'")
else:
    try:
        # Try to load the model
        model = joblib.load(MODEL_PATH)
        print("✅ Model is trained and ready to use!")
        
        # Print model information
        print("\n=== Model Information ===")
        print(f"Model type: {type(model).__name__}")
        
        # If it's a pipeline, show the steps
        if hasattr(model, 'steps'):
            print("\nPipeline steps:")
            for i, (name, step) in enumerate(model.steps, 1):
                print(f"{i}. {name}: {step.__class__.__name__}")
        
        # If it's a scikit-learn model, show parameters
        if hasattr(model, 'get_params'):
            print("\nModel parameters:")
            params = model.get_params()
            for key, value in list(params.items())[:5]:  # Show first 5 params
                print(f"- {key}: {value}")
            if len(params) > 5:
                print(f"... and {len(params) - 5} more parameters")
        
        print("\nTo use the model, you can run the FastAPI server:")
        print("uvicorn main:app --reload")
        
    except Exception as e:
        print(f"❌ Error loading the model: {str(e)}")
        print("The model file might be corrupted. Please retrain the model.")
