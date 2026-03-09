import xgboost as xgb
import os

model_path = "/home/mem4/.octoprint/data/octoprint_inference/xgboost_vibration_model.json"
if os.path.exists(model_path):
    bst = xgb.Booster()
    bst.load_model(model_path)
    if bst.feature_names:
        print("Feature Names in Model:")
        print(bst.feature_names)
        print(f"Total Features: {len(bst.feature_names)}")
    else:
        print("No feature names found in model.")
else:
    print(f"Model not found at {model_path}")
