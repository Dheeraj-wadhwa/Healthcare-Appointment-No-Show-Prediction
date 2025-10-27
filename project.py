import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.utils import resample

# ✅ Step 1: Load dataset
print("📂 Loading dataset...")
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Python\PROJECT\appointments.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("✅ Columns found:", list(df.columns))

# ✅ Step 2: Identify target column automatically
target_col = None
for col in df.columns:
    if "no_show" in col or "no-show" in col or "noshow" in col:
        target_col = col
        break

if not target_col:
    raise KeyError("❌ Could not find a 'No-show' column in your CSV file.")

# ✅ Step 3: Convert target to binary
df[target_col] = df[target_col].map({"Yes": 1, "No": 0, "Y": 1, "N": 0, 1: 1, 0: 0})
df = df.dropna(subset=[target_col])

# ✅ Step 4: Basic cleaning
date_cols = [c for c in df.columns if "date" in c]
for c in date_cols:
    df[c] = pd.to_datetime(df[c], errors="coerce")

if "scheduledday" in df.columns and "appointmentday" in df.columns:
    df["waiting_days"] = (df["appointmentday"] - df["scheduledday"]).dt.days

df = df.dropna(subset=[target_col])
df = df.select_dtypes(include=["number", "object", "bool"]).copy()

# ✅ Step 5: Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# ✅ Step 6: Balance the dataset
df_majority = df[df[target_col] == 0]
df_minority = df[df[target_col] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
X = df_balanced.drop(columns=[target_col])
y = df_balanced[target_col]

# ✅ Step 7: Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=150, random_state=42))
])

# ✅ Step 8: Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# ✅ Step 9: Evaluation
y_pred = pipeline.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Step 10: Save pipeline
joblib.dump(pipeline, "no_show_pipeline.joblib")
print("\n✅ Pipeline saved as 'no_show_pipeline.joblib'")

# ✅ Step 11: Save predictions for Power BI
pred_df = X_test.copy()
pred_df["Actual_NoShow"] = y_test.values
pred_df["Predicted_NoShow"] = y_pred
pred_df["No_Show_Probability"] = pipeline.predict_proba(X_test)[:, 1]
pred_df.to_csv("no_show_predictions.csv", index=False)
print("📁 Predictions saved as 'no_show_predictions.csv'")
print("✅ Training complete.")
