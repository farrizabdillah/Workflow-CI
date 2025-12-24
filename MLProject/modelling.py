import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# ===============================
# Path Dataset & Database
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "bank_marketing_preprocessed.csv")

# Menggunakan SQLite agar tidak terjadi error folder mlruns korup di Windows/CI
db_path = os.path.join(SCRIPT_DIR, "mlflow.db")
tracking_uri = f"sqlite:///{db_path}"

print(f"[INFO] Memuat data dari: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ===============================
# Feature & Target
# ===============================
TARGET_COL = "y"
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# MLflow Setup
# ===============================
if not os.getenv("MLFLOW_TRACKING_URI"):
    db_path = os.path.join(SCRIPT_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

mlflow.set_experiment("RandomForest_BankMarketing_Basic_Farriz")

with mlflow.start_run(nested=True):

    print("[INFO] Training RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # MLflow Logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    # Logging model
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    print(f"[INFO] Accuracy: {acc:.4f}")