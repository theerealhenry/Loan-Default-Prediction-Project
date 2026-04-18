# =========================================================
# PREDICTION CLI ENTRYPOINT (PRODUCTION READY)
# =========================================================

import argparse
import pandas as pd
import os
import sys
import yaml

# Add src to path
sys.path.append(os.path.abspath("."))

from src.modeling.predict import predict_to_submission



# =========================================================
#LOAD INPUT DATA
# =========================================================

def load_input_data(path):
    print(f"\n📂 Loading input data from: {path}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")

    print("Shape:", df.shape)

    return df


# =========================================================
#SAVE OUTPUT
# =========================================================

def save_output(df, path):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"\n💾 Predictions saved to: {path}")


# =========================================================
# MAIN FUNCTION
# =========================================================

def run_prediction(input_path, output_path, model_dir="models", id_col="ID"):

    print("\n🔮 STARTING PREDICTION PIPELINE")
    print("=" * 60)

    # =========================
    # LOAD DATA
    # =========================

    df = load_input_data(input_path)


    # =========================
    # RUN PREDICTION
    # =========================
    # Load config
    with open("configs/config_final.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    submission = predict_to_submission(
        input_df=df,
        config=config,
        id_col=id_col,
        model_dir=model_dir
    )

    print("\n📊 Prediction Summary:")
    print(submission["Target"].value_counts())

    # =========================
    # SAVE OUTPUT
    # =========================

    save_output(submission, output_path)

    print("\n✅ Prediction pipeline complete.")


# =========================================================
# CLI ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model inference")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (CSV or Parquet)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/submission.csv",
        help="Path to save predictions"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory with saved model artifacts"
    )

    parser.add_argument(
        "--id_col",
        type=str,
        default="ID",
        help="ID column name"
    )

    args = parser.parse_args()

    run_prediction(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
        id_col=args.id_col
    )