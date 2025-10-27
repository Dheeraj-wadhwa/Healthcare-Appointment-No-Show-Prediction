import pandas as pd
import joblib
import os

# === Load the trained model pipeline ===
model_path = "no_show_pipeline.joblib"

if not os.path.exists(model_path):
    print("❌ Model file not found. Please run your training script (project.py) first.")
    exit()

print("✅ Model loaded successfully.")
pipeline = joblib.load(model_path)

# === Load new appointment data ===
file_path = r"C:\Users\lenovo\Desktop\Python\PROJECT\appointments.csv"
new_df = pd.read_csv(file_path)
print(f"📂 Loaded {len(new_df)} new appointments.")

# ✅ Normalize column names (fix 'No_show' → 'no_show')
new_df.columns = [col.lower() for col in new_df.columns]

# === Drop any target column if present ===
# Ensure the column exists
if "no_show" not in new_df.columns:
    new_df["no_show"] = 0
    print("✅ Added 'no_show' column with default = 0")


# === Make predictions ===
try:
    preds = pipeline.predict_proba(new_df)[:, 1]
    new_df["No_Show_Probability"] = preds

    # Categorize risk level
    def risk_level(p):
        if p > 0.7:
            return "High"
        elif p > 0.4:
            return "Medium"
        else:
            return "Low"

    new_df["Risk_Level"] = new_df["No_Show_Probability"].apply(risk_level)

    new_df.to_csv("predicted_appointments.csv", index=False)
    print("✅ Predictions complete. Results saved to 'predicted_appointments.csv'.")
    print("\n🔍 Sample output:")
    print(new_df[["No_Show_Probability", "Risk_Level"]].head())

except Exception as e:
    print(f"⚠️ Error during prediction: {e}")
