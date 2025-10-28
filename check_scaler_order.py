import pickle
import sys
import os

# Define the file paths for your assets
SCALER_FILE = 'scaler.pkl' 

if not os.path.exists(SCALER_FILE):
    print(f"Error: {SCALER_FILE} not found in the current directory.")
    sys.exit(1)

try:
    # Load the scaler object
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
        
    print("--- Loaded scaler.pkl successfully ---")

    # Check for the feature_names_in_ attribute (modern scikit-learn versions)
    if hasattr(scaler, 'feature_names_in_'):
        correct_order = scaler.feature_names_in_.tolist()
        print("\n✅ Found feature order (Feature_names_in_):")
        print("----------------------------------------------------------------")
        print(correct_order)
        print("----------------------------------------------------------------")
        print("\nUse this list EXACTLY for SCALING_COLS_ORDERED in server.py")

    # Fallback for older scikit-learn versions (less reliable but necessary)
    # This often requires having the original training DataFrame ready, but we'll print a reminder.
    else:
        print("\n⚠️ Feature names are NOT stored in the scaler object (Older scikit-learn version).")
        print("You must manually ensure your SCALING_COLS_ORDERED list matches the order you used when calling scaler.fit() in your training notebook.")
        
except Exception as e:
    print(f"\nCRITICAL ERROR: Failed to load or inspect scaler.pkl. Error: {e}")
    print("Your scaler file may be corrupted or saved with a mismatched scikit-learn version.")
