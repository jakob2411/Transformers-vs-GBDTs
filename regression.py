import argparse
import numpy as np
import pandas as pd

# Import from pipeline module
from pipeline import (
    SEED,
    setup_seeds,
    create_tabpfn_regressor,
    create_lightgbm_regressor,
    create_xgboost_regressor,
    create_catboost_regressor,
    create_ft_transformer_regressor,
    run_regression_cv,
    run_ft_transformer_regression_cv,
    export_cv_results,
    print_cv_summary,
    load_openml_dataset,
)

# Dataset choices (Name -> OpenML ID)
DATASETS_REG = {
    "Diamonds": 42225,
    "BikeSharingDemand": 42712,
    "BrazilianHouses": 42688,
    "AirlinesDelay": 42721,
    "MedicalCharges": 42720,
    "CaliforniaHousing": 43939,
    "YachtHydrodynamics": 42370,
    "EnergyEfficiency": 1472,
    "Superconductivity": 44964,
    "ProteinStructure_CASP": 44963,
}

# Default dataset selection
DATASET_NAME = "BrazilianHouses"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regression CV runner with optional model selection and caching"
    )
    parser.add_argument(
        "--openml-id", 
        type=int, 
        default=None, 
        help="OpenML dataset ID to load (overrides DATASET_NAME)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models to run (TabPFN,LightGBM,XGBoost,CatBoost,FTTransformer) or 'all'.",
    )
    parser.add_argument(
        "--preserve-categorical",
        action="store_true",
        help="Preserve categorical features (passed to models that support it)",
    )
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=10, 
        help="Number of CV folds (default: 10)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name from DATASETS_REG mapping (e.g. Diamonds). Overrides --openml-id.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="OpenML task ID to load (overrides --dataset/openml-id)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke test (reduced folds and smaller models).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of estimators/iterations for boosting models (overrides defaults).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs for FT-Transformer (overrides default).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for tree-based models (default: -1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply smoke-mode defaults if requested
    if args.smoke:
        args.cv_folds = 2
        if args.n_estimators is None:
            args.n_estimators = 10
        if args.max_epochs is None:
            args.max_epochs = 1

    setup_seeds()

    print("="*60)
    print("Regression CV Comparison")
    print("="*60)

    # Determine dataset selection priority: --dataset > --openml-id > default
    if args.dataset is not None:
        selected_key = args.dataset
        selected_openml_id = None
    elif args.openml_id is not None:
        selected_key = None
        selected_openml_id = args.openml_id
    else:
        selected_key = DATASET_NAME
        selected_openml_id = None

    X, y, dataset, selected_task_id, target_name, selected_dataset_label = load_openml_dataset(
        DATASETS_REG,
        key=selected_key,
        openml_id=selected_openml_id,
        task_id=args.task_id,
    )

    real_name = dataset.name if dataset is not None else "Unknown"
    selected_openml_id = dataset.dataset_id if dataset is not None else selected_openml_id
    dataset_name = real_name

    print(f"\nSuccessfully loaded:")
    print(f"  - Key: {selected_dataset_label}")
    print(f"  - OpenML Name: {real_name}")
    print(f"  - OpenML ID: {selected_openml_id}")
    print(f"Samples: {len(X)}, Features: {len(X.columns)}")
    print(f"CV Folds: {args.cv_folds}")
    print(f"Preserve Categorical: {args.preserve_categorical}")

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors="coerce")
    y = y.to_numpy()

    # Prepare model hyperparameters
    n_estimators = args.n_estimators if args.n_estimators is not None else 100
    max_epochs = args.max_epochs if args.max_epochs is not None else 20
    n_jobs = args.n_jobs

    available_factories = {
        "TabPFN": create_tabpfn_regressor,
        "LightGBM": lambda: create_lightgbm_regressor(n_estimators=n_estimators, n_jobs=n_jobs),
        "XGBoost": lambda: create_xgboost_regressor(n_estimators=n_estimators, n_jobs=n_jobs),
        "CatBoost": lambda: create_catboost_regressor(iterations=n_estimators),
        "FTTransformer": lambda: create_ft_transformer_regressor(max_epochs=max_epochs),
    }

    if args.models.lower() == "all":
        selected_models = list(available_factories.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",")]
        invalid_models = [m for m in selected_models if m not in available_factories]
        if invalid_models:
            print(f"\nWarning: Unknown models will be skipped: {invalid_models}")
            selected_models = [m for m in selected_models if m in available_factories]

    print(f"\nSelected models: {', '.join(selected_models)}")

    # Run cross-validation
    cv_results = {}
    for model_name in selected_models:
        print("\n" + "="*60)
        print(f"Running CV for: {model_name}")
        print("="*60)
        
        if model_name == "FTTransformer":
            cv_results[model_name] = run_ft_transformer_regression_cv(
                X,
                y,
                n_splits=args.cv_folds,
                max_epochs=max_epochs,
                random_state=SEED
            )
        else:
            model_factory = available_factories[model_name]
            cv_results[model_name] = run_regression_cv(
                model_factory,
                X,
                y,
                n_splits=args.cv_folds,
                model_name=model_name,
                random_state=SEED
            )
        
        print_cv_summary(cv_results[model_name], model_name)

    # Display overall comparison
    print("\n" + "="*60)
    print("OVERALL COMPARISON - ALL MODELS")
    print("="*60)

    comparison_data = []
    for model_name, results in cv_results.items():
        avg = results['averages']
        comparison_data.append({
            'Model': model_name,
            'RMSE': f"{avg['rmse_mean']:.4f} ± {avg['rmse_std']:.4f}",
            'MAE': f"{avg['mae_mean']:.4f} ± {avg['mae_std']:.4f}",
            'R²': f"{avg['r2_mean']:.4f} ± {avg['r2_std']:.4f}",
            'Train Time (s)': f"{avg['train_time_mean']:.2f} ± {avg['train_time_std']:.2f}",
            'Predict Time (s)': f"{avg['predict_time_mean']:.2f} ± {avg['predict_time_std']:.2f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    print("\n" + "-"*60)
    print("BEST MODEL PER METRIC:")
    print("-"*60)

    model_names = list(cv_results.keys())
    rmse_means = [cv_results[m]['averages']['rmse_mean'] for m in model_names]
    mae_means = [cv_results[m]['averages']['mae_mean'] for m in model_names]
    r2_means = [cv_results[m]['averages']['r2_mean'] for m in model_names]
    train_means = [cv_results[m]['averages']['train_time_mean'] for m in model_names]
    predict_means = [cv_results[m]['averages']['predict_time_mean'] for m in model_names]

    best_rmse_idx = np.argmin(rmse_means)
    best_mae_idx = np.argmin(mae_means)
    best_r2_idx = np.argmax(r2_means)
    best_train_idx = np.argmin(train_means)
    best_predict_idx = np.argmin(predict_means)

    print(f"Lowest RMSE:          {model_names[best_rmse_idx]} ({rmse_means[best_rmse_idx]:.4f})")
    print(f"Lowest MAE:           {model_names[best_mae_idx]} ({mae_means[best_mae_idx]:.4f})")
    print(f"Highest R²:           {model_names[best_r2_idx]} ({r2_means[best_r2_idx]:.4f})")
    print(f"Fastest Training:     {model_names[best_train_idx]} ({train_means[best_train_idx]:.2f}s)")
    print(f"Fastest Prediction:   {model_names[best_predict_idx]} ({predict_means[best_predict_idx]:.2f}s)")
    print("="*60)

    # Export results
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)

    safe_dataset_name = str(selected_dataset_label).replace(" ", "_").replace("/", "_")
    filename = f"regression_{safe_dataset_name}_openml{selected_openml_id}_results"
    summary_df = export_cv_results(cv_results, filename)

    print("\nCross-validation completed successfully!")
    print(f"Compared {len(model_names)} models with {args.cv_folds}-fold CV on {dataset_name} dataset")
    print("="*60)