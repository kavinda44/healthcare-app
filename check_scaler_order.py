import pickle
import sys
import os


SCALER_FILE = 'scaler.pkl' 

if not os.path.exists(SCALER_FILE):
    print(f"Error: {SCALER_FILE} not found in the current directory.")
    sys.exit(1)

try:
   
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
        
    print("--- Loaded scaler.pkl successfully ---")

    
    if hasattr(scaler, 'feature_names_in_'):
        correct_order = scaler.feature_names_in_.tolist()
        print("\n✅ Found feature order (Feature_names_in_):")
        print("----------------------------------------------------------------")
        print(correct_order)
        print("----------------------------------------------------------------")
        print("\nUse this list EXACTLY for SCALING_COLS_ORDERED in server.py")

  
    else:
        print("\n⚠️ Feature names are NOT stored in the scaler object (Older scikit-learn version).")
        print("You must manually ensure your SCALING_COLS_ORDERED list matches the order you used when calling scaler.fit() in your training notebook.")
        
except Exception as e:
    print(f"\nCRITICAL ERROR: Failed to load or inspect scaler.pkl. Error: {e}")
    print("Your scaler file may be corrupted or saved with a mismatched scikit-learn version.")
