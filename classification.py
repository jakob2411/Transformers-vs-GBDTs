import argparse
import numpy as np
import pandas as pd

# Import from pipeline module
from pipeline import (
    setup_seeds,
    create_tabpfn_classifier,
    create_tabflex_classifier,
    create_lightgbm_classifier,
    create_xgboost_classifier,
    create_catboost_classifier,
    create_ft_transformer_classifier,
    run_classification_cv,
    encode_categorical_target,
    export_cv_results,
    print_cv_summary,
    load_openml_dataset,
)

DATASETS = {
    "Credit-G": 31,
    "Electricity": 151,
    "Covertype": 150,
    "Nursery": 26,
    "Adult": 1590,
    "HiggsBoson": 23512,
    "BankMarketing": 1461,
    "Gesture_Wiimote": 4538,
    "PhishingWebsites": 4534,
    "TelcoCustomerChurn": 42178,
}

# Default dataset selection (set to one of the keys above)
DATASET_NAME = "Credit-G"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CV runner with optional model selection and categorical preservation"
    )
    parser.add_argument(
        "--openml-id", 
        type=int, 
        default=31, 
        help="OpenML dataset ID to load (default: 31)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models to run (TabPFN,TabFlex,LightGBM,XGBoost,CatBoost,FTTransformer) or 'all'",
    )
    parser.add_argument(
        "--preserve-categorical",
        action="store_true",
        help="Preserve categorical features for CatBoost and LightGBM (native handling)",
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
        help="Dataset name from DATASETS mapping (e.g. Credit-G, Adult). Overrides --openml-id if present.",
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
    print("Classification CV Comparison")
    print("="*60)
    # Determine dataset selection priority: --dataset > --openml-id > default
    if args.dataset is not None:
        selected_key = args.dataset
        selected_openml_id = None
    elif args.openml_id != 31:
        selected_key = None
        selected_openml_id = args.openml_id
    else:
        selected_key = DATASET_NAME
        selected_openml_id = None

    X, y, dataset, selected_task_id, target_name, selected_dataset_label = load_openml_dataset(
        DATASETS,
        key=selected_key,
        openml_id=selected_openml_id,
        task_id=args.task_id,
    )

    selected_openml_id = getattr(dataset, 'dataset_id', None)
    dataset_name = dataset.name if dataset is not None else selected_dataset_label

    if dataset is not None and getattr(dataset, 'name', None) is not None:
        print(f"Selected dataset key: {selected_dataset_label} -> OpenML name: {dataset.name} (ID: {selected_openml_id})")
    else:
        print(f"Selected dataset: {selected_dataset_label} (OpenML ID: {selected_openml_id})")
    print(f"CV Folds: {args.cv_folds}")
    print(f"Preserve Categorical: {args.preserve_categorical}")
    print(f"\nDataset: {dataset_name}")
    print(f"Samples: {len(X)}, Features: {len(X.columns)}")

    y = encode_categorical_target(y)

    # Prepare model hyperparameters
    n_estimators = args.n_estimators if args.n_estimators is not None else 100
    max_epochs = args.max_epochs if args.max_epochs is not None else 20
    n_jobs = args.n_jobs

    available_factories = {
        "TabPFN": create_tabpfn_classifier,
        "TabFlex": create_tabflex_classifier,
        "LightGBM": lambda: create_lightgbm_classifier(n_estimators=n_estimators, n_jobs=n_jobs),
        "XGBoost": lambda: create_xgboost_classifier(n_estimators=n_estimators, n_jobs=n_jobs),
        "CatBoost": lambda: create_catboost_classifier(iterations=n_estimators),
        "FTTransformer": lambda: create_ft_transformer_classifier(max_epochs=max_epochs),
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
        
        model_factory = available_factories[model_name]
        cv_results[model_name] = run_classification_cv(
            model_factory,
            X,
            y,
            n_splits=args.cv_folds,
            preserve_categorical=args.preserve_categorical
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
            'Accuracy': f"{avg['accuracy_mean']:.4f} ± {avg['accuracy_std']:.4f}",
            'Balanced Acc': f"{avg['balanced_accuracy_mean']:.4f} ± {avg['balanced_accuracy_std']:.4f}",
            'F1-Score': f"{avg['f1_mean']:.4f} ± {avg['f1_std']:.4f}",
            'ROC-AUC': f"{avg['roc_auc_mean']:.4f} ± {avg['roc_auc_std']:.4f}",
            'Log Loss': f"{avg['log_loss_mean']:.4f} ± {avg['log_loss_std']:.4f}",
            'Train Time (s)': f"{avg['train_time_mean']:.2f} ± {avg['train_time_std']:.2f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    print("\n" + "-"*60)
    print("BEST MODEL PER METRIC:")
    print("-"*60)

    model_names = list(cv_results.keys())
    accuracy_means = [cv_results[m]['averages']['accuracy_mean'] for m in model_names]
    bal_acc_means = [cv_results[m]['averages']['balanced_accuracy_mean'] for m in model_names]
    f1_means = [cv_results[m]['averages']['f1_mean'] for m in model_names]
    roc_auc_means = [cv_results[m]['averages']['roc_auc_mean'] for m in model_names]
    train_means = [cv_results[m]['averages']['train_time_mean'] for m in model_names]

    best_acc_idx = np.argmax(accuracy_means)
    best_bal_acc_idx = np.argmax(bal_acc_means)
    best_f1_idx = np.argmax(f1_means)
    best_roc_idx = np.argmax([x for x in roc_auc_means if not np.isnan(x)] or [0])
    best_train_idx = np.argmin(train_means)

    print(f"Highest Accuracy:        {model_names[best_acc_idx]} ({accuracy_means[best_acc_idx]:.4f})")
    print(f"Highest Balanced Acc:    {model_names[best_bal_acc_idx]} ({bal_acc_means[best_bal_acc_idx]:.4f})")
    print(f"Highest F1-Score:        {model_names[best_f1_idx]} ({f1_means[best_f1_idx]:.4f})")
    if not all(np.isnan(roc_auc_means)):
        print(f"Highest ROC-AUC:         {model_names[best_roc_idx]} ({roc_auc_means[best_roc_idx]:.4f})")
    print(f"Fastest Training:        {model_names[best_train_idx]} ({train_means[best_train_idx]:.2f}s)")
    print("="*60)

    # Export results
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)

    safe_dataset_name = str(selected_dataset_label or dataset_name).replace(" ", "_").replace("/", "_")
    filename = f"classification_{safe_dataset_name}_openml{selected_openml_id}_results"
    summary_df = export_cv_results(cv_results, filename)

    print(f"\nCross-validation completed successfully!")
    print(f"Compared {len(model_names)} models with {args.cv_folds}-fold CV on {dataset_name} dataset")
    print("="*60)