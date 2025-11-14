import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_FILE = "car_data.csv"
MODEL_FILE = "car_price_model.joblib"
RANDOM_STATE = 42

def find_target_column(df):
    candidates = [c for c in df.columns if "selling" in c.lower() or "price" in c.lower() and "selling" in c.lower()]
    if "Selling_Price" in df.columns:
        return "Selling_Price"
    for c in df.columns:
        if c.lower().strip() == "selling_price":
            return c
    for c in df.columns:
        low = c.lower()
        if "selling" in low and "price" in low:
            return c
    possible_price = [c for c in df.columns if "price" in c.lower()]
    if len(possible_price) == 1:
        return possible_price[0]
    raise KeyError("Could not find target column (Selling_Price). Please check your CSV columns: " + ", ".join(df.columns))

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def preprocess(df, target_col):
    df = df.copy()
    if target_col not in df.columns:
        raise KeyError(f"Target column missing after load: {target_col}")

    print("Columns detected:", list(df.columns))
    print("First rows:\n", df.head(3).to_string(index=False))

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    to_drop = []
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        cname_lower = c.lower()
        if ("name" in cname_lower or "model" in cname_lower or "id" in cname_lower) and nunique > 50:
            to_drop.append(c)
        if nunique > 200:
            to_drop.append(c)
    if to_drop:
        print("Dropping high-cardinality columns:", to_drop)
        df = df.drop(columns=to_drop)
        cat_cols = [c for c in cat_cols if c not in to_drop]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    for c in num_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)

    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode()
            fillv = mode[0] if len(mode) > 0 else ""
            df[c] = df[c].fillna(fillv)

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    print("Feature columns used:", list(X.columns))
    print("Sample of prepared features:\n", X.head(2).to_string(index=False))
    return X, y, encoders

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("Evaluation on test set:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2 : {r2:.3f}")

    return model, X_test, y_test, preds

def save_artifacts(model, encoders, feature_cols):
    joblib.dump({
        "model": model,
        "encoders": encoders,
        "feature_columns": feature_cols
    }, MODEL_FILE)
    print("Saved model and encoders to", MODEL_FILE)

def main():
    df = load_data(DATA_FILE)
    try:
        target = find_target_column(df)
    except KeyError as e:
        print(str(e))
        print("Available columns:", df.columns.tolist())
        return

    X, y, encoders = preprocess(df, target)
    model, X_test, y_test, preds = train_and_evaluate(X, y)
    save_artifacts(model, encoders, list(X.columns))

    example = X_test.iloc[0:1]
    pred_val = model.predict(example)[0]
    print("Example input:\n", example.to_string(index=False))
    print(f"Predicted {target}: {pred_val:.3f}  |  Actual: {y_test.iloc[0]:.3f}")

if __name__ == "__main__":
    main()
