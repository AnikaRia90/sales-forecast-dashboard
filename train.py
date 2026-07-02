"""
Full training pipeline:
    1. Data ingestion (all 5 Kaggle files)
    2. Data cleaning
    3. Model evaluation (holdout)
    4. Model training (full dataset)
    5. Save model

Run from the project root:
    python train.py
"""

import os
from app.data_pipeline import run_pipeline
from app.forecast_model import (
    train_model,
    evaluate_model,
    save_model,
    MODEL_PATH,
)


def main():
    print("=" * 50)
    print("   AI Sales Forecasting - Training Pipeline")
    print("=" * 50)

    # Step 1 + 2: Ingestion and cleaning
    print("\nStep 1: Data ingestion + cleaning...")
    df = run_pipeline()

    # Step 3: Evaluate on holdout
    print("\nStep 2: Evaluating model (30-day holdout)...")
    mae, mape = evaluate_model(df)
    print(f"  MAE  (Mean Absolute Error):            {mae:,.2f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    # Step 4: Train final model on full dataset
    print("\nStep 3: Training final model on full dataset...")
    model = train_model(df)

    # Step 5: Save
    print("\nStep 4: Saving model...")
    save_model(model, MODEL_PATH)
    print(f"  Model saved -> {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("Training complete.")
    print("Next: uvicorn app.main:app --reload")
    print("=" * 50)


if __name__ == "__main__":
    main()
