import random
import numpy as np
import pandas as pd
import torch
import time
from pathlib import Path
import openml
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier, TabPFNRegressor
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig


# Global seed for reproducibility
SEED = 2411


def safe_filename_component(value):
    """Sanitize a value for use in filenames."""
    return str(value).strip().replace(" ", "_").replace("/", "_")


def setup_seeds(seed=None):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    if seed is None:
        seed = SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"All random seeds set to: {seed}")


def load_openml_dataset(mapping, key=None, openml_id=None, task_id=None, cache_dir="data_cache"):
    """
    Load an OpenML dataset or task with parquet caching for faster subsequent loads.
    Returns (X, y, dataset, task_id, target, label).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    selected_label = None
    dataset = None
    selected_task_id = None
    target_name = None

    # Resolve what to load based on provided parameters
    if key is not None and key in mapping:
        val = mapping[key]
        selected_label = key
        # Try loading as a dataset first, then fall back to task if that fails
        try:
            dataset = openml.datasets.get_dataset(val)
            target_name = dataset.default_target_attribute
        except Exception:
            task = openml.tasks.get_task(val)
            dataset = task.get_dataset()
            selected_task_id = val
            target_name = task.target_name
    elif task_id is not None:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        selected_task_id = task_id
        target_name = task.target_name
        selected_label = f"Task_{task_id}"
    elif openml_id is not None:
        dataset = openml.datasets.get_dataset(openml_id)
        target_name = dataset.default_target_attribute
        selected_label = f"Dataset_{openml_id}"
    else:
        raise ValueError("No dataset key, openml_id or task_id provided")

    # Try loading from cache first
    dataset_id = getattr(dataset, "dataset_id", None)
    cache_file = cache_path / f"{dataset_id}.parquet" if dataset_id is not None else None

    if cache_file is not None and cache_file.exists():
        df_all = pd.read_parquet(cache_file)
        if "__target__" not in df_all.columns:
            raise RuntimeError(f"Cached file {cache_file} missing '__target__' column")
        X = df_all.drop(columns=["__target__"]).reset_index(drop=True)
        y = df_all["__target__"].reset_index(drop=True)
    else:
        # Load from OpenML and cache for next time
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
        
        if cache_file is not None:
            try:
                df_all = X.copy()
                df_all["__target__"] = y
                df_all.to_parquet(cache_file)
            except Exception:
                pass  # Caching failure is not critical

    return X, y, dataset, selected_task_id, target_name, selected_label

class FTTransformerWrapper:
    """Wrapper to make pytorch-tabular's FT-Transformer work like a scikit-learn model."""
    
    def __init__(self, max_epochs=20, seed=None, task="classification", accelerator="auto"):
        self.max_epochs = max_epochs
        self.seed = seed if seed is not None else SEED
        self.task = task
        self.accelerator = accelerator
        self.model = None
        self.is_tabular_model = True
    
    def fit(self, X, y):
        """Train the FT-Transformer model."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        train_df = X.reset_index(drop=True).copy()
        target_col = "target"
        
        if isinstance(y, pd.Series):
            train_df[target_col] = y.reset_index(drop=True)
        else:
            train_df[target_col] = pd.Series(y).reset_index(drop=True)
        
        # Identify categorical and continuous columns
        cat_cols = [
            c for c in train_df.columns
            if pd.api.types.is_object_dtype(train_df[c]) or 
               isinstance(train_df[c].dtype, pd.CategoricalDtype)
        ]
        cat_cols = [c for c in cat_cols if c != target_col]
        cont_cols = [c for c in train_df.columns if c not in cat_cols + [target_col]]
        
        # Calculate batch size to avoid single-sample batches (which cause issues with BatchNorm)
        n_samples = len(train_df)
        batch_size = max(64, min(512, n_samples // 100))
        batch_size = min(batch_size, max(1, n_samples))

        if n_samples > 1:
            batch_size = max(2, batch_size)
            while batch_size > 2 and (n_samples % batch_size == 1):
                batch_size -= 1

        data_config = DataConfig(
            target=[target_col],
            continuous_cols=cont_cols,
            categorical_cols=cat_cols
        )

        # Choose accelerator: prefer MPS on macOS, otherwise CPU
        if self.accelerator == "auto":
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                acc = "mps"
            else:
                acc = "cpu"
        else:
            acc = self.accelerator

        trainer_config = TrainerConfig(
            max_epochs=self.max_epochs,
            accelerator=acc,
            devices=1,
            deterministic=True,
            checkpoints_save_top_k=1,
            early_stopping="valid_loss",
            early_stopping_patience=5,
            checkpoints=None,
            load_best=True,
            batch_size=batch_size,
            trainer_kwargs={"enable_progress_bar": False}
        )
        
        optimizer_config = OptimizerConfig()
        model_config = FTTransformerConfig(
            task=self.task,
            learning_rate=1e-3,
            metrics=["accuracy"] if self.task == "classification" else ["mean_squared_error"],
            seed=self.seed
        )
        
        self.model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )

        # Suppress FutureWarning from pytorch-tabular
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"A value is trying to be set on a copy of a DataFrame or Series through chained assignment.*",
            )
            self.model.fit(train=train_df)
        return self
    
    def predict(self, X):
        """Predict class labels or regression values."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        test_df = X.reset_index(drop=True)
        preds_df = self.model.predict(test_df)
        
        if self.task == "classification":
            pred_col = next((c for c in preds_df.columns if c.endswith("_prediction")), preds_df.columns[0])
            return preds_df[pred_col].to_numpy()
        else:
            # For regression, prefer columns ending with '_prediction'
            pred_col = next((c for c in preds_df.columns if c.endswith("_prediction")), None)
            if pred_col is not None:
                return pd.to_numeric(preds_df[pred_col]).to_numpy()
            # Fall back to first numeric column
            for c in preds_df.columns:
                if pd.api.types.is_numeric_dtype(preds_df[c].dtype):
                    return pd.to_numeric(preds_df[c]).to_numpy()
            return pd.to_numeric(preds_df.iloc[:, 0]).to_numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities (classification only)."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        test_df = X.reset_index(drop=True)
        preds_df = self.model.predict(test_df)
        
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification")

        # Find probability columns (different pytorch-tabular versions name them differently)
        prob_cols = [
            c for c in preds_df.columns
            if ("probability" in c.lower()) or (c.split("_")[-1].isdigit())
        ]

        if prob_cols:
            # Sort by class index if columns encode class as last token
            try:
                if all(c.split("_")[-1].isdigit() for c in prob_cols):
                    prob_cols = sorted(prob_cols, key=lambda c: int(c.split("_")[-1]))
            except Exception:
                pass
            return preds_df[prob_cols].to_numpy()
        else:
            # Fall back to one-hot encoding from predictions
            preds = self.predict(X)
            preds_arr = np.asarray(preds)
            classes, inv = np.unique(preds_arr, return_inverse=True)
            n_classes = len(classes)
            proba = np.zeros((len(preds_arr), n_classes), dtype=float)
            proba[np.arange(len(preds_arr)), inv] = 1.0
            return proba


def create_tabpfn_classifier():
    """Create a TabPFN classifier with version-specific parameter handling."""
    try:
        return TabPFNClassifier(device="cpu", N_ensemble_configurations=4, ignore_pretraining_limits=True)
    except TypeError:
        try:
            return TabPFNClassifier(device="cpu", n_ensemble_configurations=4, ignore_pretraining_limits=True)
        except TypeError:
            try:
                return TabPFNClassifier(device="cpu", n_estimators=4, ignore_pretraining_limits=True)
            except TypeError:
                try:
                    return TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
                except TypeError:
                    try:
                        return TabPFNClassifier(ignore_pretraining_limits=True)
                    except TypeError:
                        return TabPFNClassifier()


def create_tabflex_classifier(**kwargs):
    """Create a TabFlex classifier from ticl package, forcing CPU usage."""
    try:
        from ticl.prediction.tabflex import TabFlex
        from ticl.prediction.tabpfn import TabPFNClassifier
    except Exception as e:
        raise ImportError("TabFlex (ticl) is not installed. Install with `pip install git+https://github.com/microsoft/ticl.git`") from e
    
    class TabFlexCPU(TabFlex):
        """TabFlex variant that forces CPU to avoid GPU issues."""
        def __init__(self, **kwargs):
            # Don't call super().__init__() to avoid eager model instantiation
            self.tabflexh1k = None
            self.tabflexl100 = None
            self.tabflexs100 = None
            self.model = None
            
        def _ensure_models_initialized(self):
            """Initialize the underlying TabFlex models on CPU."""
            if self.tabflexh1k is None:
                from ticl.prediction.tabflex import fetch_model
                fetch_model('ssm_tabpfn_b4_maxnumclasses100_modellinear_attention_numfeatures1000_n1024_validdatanew_warm_08_23_2024_19_25_40_epoch_3140.cpkt')
                fetch_model('ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50_epoch_110.cpkt')
                fetch_model('ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44_epoch_3110.cpkt')
                
                self.tabflexh1k = TabPFNClassifier(
                    device='cpu',
                    model_string='ssm_tabpfn_b4_maxnumclasses100_modellinear_attention_numfeatures1000_n1024_validdatanew_warm_08_23_2024_19_25_40',
                    N_ensemble_configurations=3,
                    epoch='3140',
                )
                self.tabflexl100 = TabPFNClassifier(
                    device='cpu',
                    model_string='ssm_tabpfn_b4_largedatasetTrue_modellinear_attention_nsamples50000_08_01_2024_22_05_50',
                    N_ensemble_configurations=1,
                    epoch='110',
                )
                self.tabflexs100 = TabPFNClassifier(
                    device='cpu',
                    model_string='ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44',
                    N_ensemble_configurations=3,
                    epoch='3110',
                )
        
        def fit(self, X, y):
            self._ensure_models_initialized()
            N, D = X.shape
            
            if N >= 3000 and D <= 100:
                self.model = self.tabflexl100
            elif D > 100 or (D/N >= 0.2 and N >= 3000):
                if D <= 1000:
                    self.model = self.tabflexh1k
                else:
                    self.model = self.tabflexh1k
                    self.model.dimension_reduction = 'random_proj'
                    self.model.fit(X, y, overwrite_warning=True)
                    return self
            else:
                self.model = self.tabflexs100
            
            self.model.fit(X, y, overwrite_warning=True)
            return self
        
        def predict(self, X):
            """Make predictions using the selected underlying model."""
            self._ensure_models_initialized()
            if self.model is None:
                raise RuntimeError("TabFlexCPU: model not initialized. Call fit() first.")
            if hasattr(self.model, "predict"):
                return self.model.predict(X)
            if hasattr(super(), "predict"):
                return super().predict(X)
            raise NotImplementedError("Underlying model does not implement predict()")

        def predict_proba(self, X):
            """Return class probabilities, or one-hot encoding if not available."""
            import numpy as _np

            self._ensure_models_initialized()
            if self.model is None:
                raise RuntimeError("TabFlexCPU: model not initialized. Call fit() first.")

            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)

            # Fall back to one-hot probabilities from predictions
            if hasattr(self.model, "predict"):
                preds = _np.asarray(self.model.predict(X))
                classes, inv = _np.unique(preds, return_inverse=True)
                proba = _np.zeros((len(preds), len(classes)), dtype=float)
                proba[_np.arange(len(preds)), inv] = 1.0
                return proba

            raise NotImplementedError("Underlying model does not implement predict_proba() or predict()")
    
    return TabFlexCPU(**kwargs)


def create_lightgbm_classifier(n_estimators=100, n_jobs=-1, learning_rate=0.1):
    """Create a LightGBM classifier."""
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=SEED,
        verbosity=-1,
        n_jobs=n_jobs,
    )


def create_xgboost_classifier(n_estimators=100, n_jobs=1, learning_rate=0.1):
    """Create an XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=n_jobs,
    )


def create_catboost_classifier(iterations=100, learning_rate=0.1):
    """Create a CatBoost classifier."""
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        random_state=SEED,
        verbose=0
    )


def create_ft_transformer_classifier(max_epochs=20, accelerator="auto"):
    """Create an FT-Transformer for classification."""
    return FTTransformerWrapper(max_epochs=max_epochs, seed=SEED, task="classification", accelerator=accelerator)


def create_ft_transformer_regressor(max_epochs=20, accelerator="auto"):
    """Create an FT-Transformer for regression."""
    return FTTransformerWrapper(max_epochs=max_epochs, seed=SEED, task="regression", accelerator=accelerator)


def create_tabpfn_regressor():
    """Create a TabPFN regressor with version-specific parameter handling."""
    try:
        return TabPFNRegressor(
            device="cpu",
            n_estimators=4,
            ignore_pretraining_limits=True
        )
    except TypeError:
        try:
            return TabPFNRegressor(
                device="cpu",
                ignore_pretraining_limits=True
            )
        except TypeError:
            try:
                return TabPFNRegressor(ignore_pretraining_limits=True)
            except TypeError:
                return TabPFNRegressor()


def create_tabflex_regressor(**kwargs):
    """TabFlex does not support regression."""
    raise NotImplementedError("TabFlex does not provide a regressor. Use a different model for regression.")


def create_xgboost_regressor(n_estimators=100, n_jobs=1, learning_rate=0.1):
    """Create an XGBoost regressor."""
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=SEED,
        verbosity=0,
        n_jobs=n_jobs,
    )


def create_lightgbm_regressor(n_estimators=100, n_jobs=-1, learning_rate=0.1):
    """Create a LightGBM regressor."""
    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=SEED,
        verbosity=-1,
        n_jobs=n_jobs,
    )


def create_catboost_regressor(iterations=100, learning_rate=0.1):
    """Create a CatBoost regressor."""
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        random_state=SEED,
        verbose=0
    )


def run_classification_cv(estimator_factory, X_df, y_arr, n_splits=10, 
                         random_state=None, preserve_categorical=False):
    """
    Run stratified k-fold cross-validation for a classifier.
    Returns dict with 'averages' and 'folds' keys containing performance metrics.
    """
    if random_state is None:
        random_state = SEED
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    acc_list, bal_acc_list, prec_list, rec_list, f1_list = [], [], [], [], []
    roc_auc_list, logloss_list = [], []
    train_times, predict_times = [], []
    n_train_list, n_test_list = [], []
    fold_list = []
    
    print(f"\nRunning {n_splits}-fold stratified CV...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y_arr), 1):
        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_te_df = X_df.iloc[test_idx].reset_index(drop=True)
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        
        # Create the model instance
        est = estimator_factory()

        # Identify categorical columns
        cat_cols = [c for c in X_tr_df.columns if pd.api.types.is_object_dtype(X_tr_df[c]) or isinstance(X_tr_df[c].dtype, pd.CategoricalDtype)]

        # Decide how to prepare inputs based on model type
        if not preserve_categorical and not getattr(est, 'is_tabular_model', False) and not isinstance(est, CatBoostClassifier):
            # Encode categorical columns as integers, then convert to numpy
            X_tr_encoded = X_tr_df.copy()
            X_te_encoded = X_te_df.copy()
            
            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                X_tr_filled = X_tr_df[col].astype(str).replace('nan', '__missing__')
                X_te_filled = X_te_df[col].astype(str).replace('nan', '__missing__')
                
                X_tr_encoded[col] = le.fit_transform(X_tr_filled)
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                X_te_encoded[col] = X_te_filled.map(mapping).fillna(-1).astype(int)
                encoders[col] = le
            
            # Convert to numeric and fill missing values
            X_tr_numeric = X_tr_encoded.apply(lambda col: pd.to_numeric(col, errors='coerce'))
            medians = X_tr_numeric.median(skipna=True).fillna(0)
            X_tr_filled = X_tr_numeric.fillna(medians)

            X_te_numeric = X_te_encoded.apply(lambda col: pd.to_numeric(col, errors='coerce'))
            X_te_filled = X_te_numeric.fillna(medians)

            X_tr = X_tr_filled.to_numpy()
            X_te = X_te_filled.to_numpy()
            fit_kwargs = {}
        else:
            # Keep as DataFrames for models that accept pandas input
            X_tr = X_tr_df.copy()
            X_te = X_te_df.copy()
            
            if isinstance(est, CatBoostClassifier) and len(cat_cols) > 0:
                # CatBoost needs categorical columns as strings without NaN
                for col in cat_cols:
                    X_tr[col] = X_tr[col].astype(str).replace('nan', '__missing__')
                    X_te[col] = X_te[col].astype(str).replace('nan', '__missing__')
                fit_kwargs = {'cat_features': cat_cols}
            else:
                fit_kwargs = {}

        # Train the model
        t0 = time.perf_counter()
        try:
            est.fit(X_tr, y_tr, **fit_kwargs)
        except TypeError:
            # Some models may not accept extra kwargs
            est.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = est.predict(X_te)
        predict_time = time.perf_counter() - t0
        
        # Calculate metrics
        acc = accuracy_score(y_te, y_pred)
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC and log loss require probability predictions
        try:
            y_proba = est.predict_proba(X_te)
            n_classes = y_proba.shape[1]
            
            if n_classes == 2:
                roc_auc = roc_auc_score(y_te, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_te, y_proba, multi_class='ovr', average='weighted')
            
            logloss = log_loss(y_te, y_proba)
        except Exception:
            roc_auc = np.nan
            logloss = np.nan
        
        # Store metrics
        acc_list.append(acc)
        bal_acc_list.append(bal_acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)
        logloss_list.append(logloss)
        train_times.append(train_time)
        predict_times.append(predict_time)
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        n_train_list.append(n_train)
        n_test_list.append(n_test)

        fold_list.append({
            "fold": fold_idx,
            "n_train": n_train,
            "n_test": n_test,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "train_time": train_time,
            "predict_time": predict_time
        })
        
        print(f"  Fold {fold_idx:2d}: Acc={acc:.4f}, BalAcc={bal_acc:.4f}, F1={f1:.4f}")
    
    # Calculate averages
    averages = {
        "accuracy_mean": np.mean(acc_list),
        "accuracy_std": np.std(acc_list),
        "balanced_accuracy_mean": np.mean(bal_acc_list),
        "balanced_accuracy_std": np.std(bal_acc_list),
        "precision_mean": np.mean(prec_list),
        "precision_std": np.std(prec_list),
        "recall_mean": np.mean(rec_list),
        "recall_std": np.std(rec_list),
        "f1_mean": np.mean(f1_list),
        "f1_std": np.std(f1_list),
        "roc_auc_mean": np.nanmean(roc_auc_list),
        "roc_auc_std": np.nanstd(roc_auc_list),
        "log_loss_mean": np.nanmean(logloss_list),
        "log_loss_std": np.nanstd(logloss_list),
        "train_time_mean": np.mean(train_times),
        "train_time_std": np.std(train_times),
        "predict_time_mean": np.mean(predict_times),
        "predict_time_std": np.std(predict_times),
        "n_train_mean": np.mean(n_train_list) if len(n_train_list) else 0,
        "n_train_std": np.std(n_train_list) if len(n_train_list) else 0,
        "n_test_mean": np.mean(n_test_list) if len(n_test_list) else 0,
        "n_test_std": np.std(n_test_list) if len(n_test_list) else 0,
    }
    
    return {"averages": averages, "folds": fold_list}


def run_regression_cv(estimator_factory, X_df, y_arr, n_splits=10, 
                     random_state=None, model_name="Model"):
    """
    Run k-fold cross-validation for a regressor.
    Returns dict with 'averages' and 'folds' keys containing performance metrics.
    """
    if random_state is None:
        random_state = SEED
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    rmse_list, mae_list, r2_list = [], [], []
    train_times, predict_times = [], []
    n_train_list, n_test_list = [], []
    fold_list = []
    
    print(f"\nRunning {n_splits}-fold CV for {model_name}...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_df), 1):
        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_te_df = X_df.iloc[test_idx].reset_index(drop=True)
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        
        # Encode categorical columns as integers
        cat_cols = [c for c in X_tr_df.columns if pd.api.types.is_object_dtype(X_tr_df[c]) or isinstance(X_tr_df[c].dtype, pd.CategoricalDtype)]
        
        X_tr_encoded = X_tr_df.copy()
        X_te_encoded = X_te_df.copy()
        
        for col in cat_cols:
            le = LabelEncoder()
            X_tr_filled = X_tr_df[col].astype(str).replace('nan', '__missing__')
            X_te_filled = X_te_df[col].astype(str).replace('nan', '__missing__')
            
            X_tr_encoded[col] = le.fit_transform(X_tr_filled)
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            X_te_encoded[col] = X_te_filled.map(mapping).fillna(-1).astype(int)
        
        # Convert to numeric and fill missing values using training medians
        X_tr_numeric = X_tr_encoded.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        medians = X_tr_numeric.median(skipna=True).fillna(0)
        X_tr_filled = X_tr_numeric.fillna(medians)

        X_te_numeric = X_te_encoded.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        X_te_filled = X_te_numeric.fillna(medians)

        X_tr = X_tr_filled.to_numpy()
        X_te = X_te_filled.to_numpy()
        
        est = estimator_factory()
        
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = est.predict(X_te)
        predict_time = time.perf_counter() - t0
        
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        train_times.append(train_time)
        predict_times.append(predict_time)
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        n_train_list.append(n_train)
        n_test_list.append(n_test)

        fold_list.append({
            "fold": fold_idx,
            "n_train": n_train,
            "n_test": n_test,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_time": train_time,
            "predict_time": predict_time
        })
        
        print(f"  Fold {fold_idx:2d}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # Calculate averages
    averages = {
        "rmse_mean": np.mean(rmse_list),
        "rmse_std": np.std(rmse_list),
        "mae_mean": np.mean(mae_list),
        "mae_std": np.std(mae_list),
        "r2_mean": np.mean(r2_list),
        "r2_std": np.std(r2_list),
        "train_time_mean": np.mean(train_times),
        "train_time_std": np.std(train_times),
        "predict_time_mean": np.mean(predict_times),
        "predict_time_std": np.std(predict_times),
        "n_train_mean": np.mean(n_train_list) if len(n_train_list) else 0,
        "n_train_std": np.std(n_train_list) if len(n_train_list) else 0,
        "n_test_mean": np.mean(n_test_list) if len(n_test_list) else 0,
        "n_test_std": np.std(n_test_list) if len(n_test_list) else 0,
    }
    
    return {"averages": averages, "folds": fold_list}


def run_ft_transformer_regression_cv(X_df, y_arr, n_splits=10, random_state=None, max_epochs=10, accelerator="auto"):
    """
    Run k-fold cross-validation for FT-Transformer regression.
    Keeps data as pandas DataFrames for compatibility with pytorch-tabular.
    Returns dict with 'averages' and 'folds' keys.
    """
    if random_state is None:
        random_state = SEED
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    rmse_list, mae_list, r2_list = [], [], []
    train_times, predict_times = [], []
    n_train_list, n_test_list = [], []
    fold_list = []
    
    print(f"\nRunning {n_splits}-fold CV for FT-Transformer...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_df), 1):
        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True).copy()
        X_te_df = X_df.iloc[test_idx].reset_index(drop=True).copy()
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        
        target_col = "target"
        X_tr_df[target_col] = pd.Series(y_tr).astype(float).reset_index(drop=True)
        
        # Identify categorical and continuous columns
        cat_cols = [
            c for c in X_tr_df.columns
            if pd.api.types.is_object_dtype(X_tr_df[c]) or 
               isinstance(X_tr_df[c].dtype, pd.CategoricalDtype)
        ]
        cat_cols = [c for c in cat_cols if c != target_col]
        cont_cols = [c for c in X_tr_df.columns if c not in cat_cols + [target_col]]

        # Ensure continuous columns are numeric and fill missing values
        if len(cont_cols) > 0:
            X_tr_df[cont_cols] = X_tr_df[cont_cols].apply(lambda col: pd.to_numeric(col, errors='coerce')).astype(float)
            medians = X_tr_df[cont_cols].median(skipna=True).fillna(0.0)
            X_tr_df[cont_cols] = X_tr_df[cont_cols].fillna(medians)

            for c in cont_cols:
                if c in X_te_df.columns:
                    X_te_df[c] = pd.to_numeric(X_te_df[c], errors='coerce').astype(float)
                    X_te_df[c] = X_te_df[c].fillna(medians[c])

        # Ensure categorical columns are strings without NaN
        for c in cat_cols:
            if c in X_tr_df.columns:
                X_tr_df[c] = X_tr_df[c].astype(str).replace('nan', '__missing__')
            if c in X_te_df.columns:
                X_te_df[c] = X_te_df[c].astype(str).replace('nan', '__missing__')
        
        data_config = DataConfig(
            target=[target_col],
            continuous_cols=cont_cols,
            categorical_cols=cat_cols
        )
        
        # Choose accelerator
        if accelerator == "auto":
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                acc = "mps"
            else:
                acc = "cpu"
        else:
            acc = accelerator

        trainer_config = TrainerConfig(
            max_epochs=max_epochs,
            accelerator=acc,
            devices=1,
            deterministic=True,
            checkpoints_save_top_k=1,
            early_stopping="valid_loss",
            early_stopping_patience=5,
            checkpoints=None,
            load_best=True,
            trainer_kwargs={"log_every_n_steps": 50, "enable_progress_bar": False}
        )
        
        optimizer_config = OptimizerConfig()
        model_config = FTTransformerConfig(
            task="regression",
            learning_rate=1e-3,
            seed=SEED
        )
        
        ft_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )
        
        t0 = time.perf_counter()
        ft_model.fit(train=X_tr_df)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        preds_df = ft_model.predict(X_te_df)
        predict_time = time.perf_counter() - t0
        
        # Extract predictions and handle NaN values
        pred_col = next((c for c in preds_df.columns if c.endswith("_prediction")), preds_df.columns[0])
        y_pred = pd.to_numeric(preds_df[pred_col], errors="coerce").to_numpy()

        if np.isnan(y_pred).any():
            if np.all(np.isnan(y_pred)):
                # Use training median if all predictions are NaN
                try:
                    train_median = float(np.nanmedian(y_tr))
                    if not np.isfinite(train_median):
                        train_median = 0.0
                except Exception:
                    train_median = 0.0
                y_pred = np.full_like(y_pred, fill_value=train_median, dtype=float)
            else:
                # Fill individual NaN values with median of valid predictions
                valid_median = np.nanmedian(y_pred)
                if not np.isfinite(valid_median):
                    try:
                        valid_median = float(np.nanmedian(y_tr))
                    except Exception:
                        valid_median = 0.0
                y_pred = np.where(np.isnan(y_pred), valid_median, y_pred)
        
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        train_times.append(train_time)
        predict_times.append(predict_time)
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        n_train_list.append(n_train)
        n_test_list.append(n_test)

        fold_list.append({
            "fold": fold_idx,
            "n_train": n_train,
            "n_test": n_test,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_time": train_time,
            "predict_time": predict_time
        })
        
        print(f"  Fold {fold_idx:2d}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # Calculate averages
    averages = {
        "rmse_mean": np.mean(rmse_list),
        "rmse_std": np.std(rmse_list),
        "mae_mean": np.mean(mae_list),
        "mae_std": np.std(mae_list),
        "r2_mean": np.mean(r2_list),
        "r2_std": np.std(r2_list),
        "train_time_mean": np.mean(train_times),
        "train_time_std": np.std(train_times),
        "predict_time_mean": np.mean(predict_times),
        "predict_time_std": np.std(predict_times),
        "n_train_mean": np.mean(n_train_list) if len(n_train_list) else 0,
        "n_train_std": np.std(n_train_list) if len(n_train_list) else 0,
        "n_test_mean": np.mean(n_test_list) if len(n_test_list) else 0,
        "n_test_std": np.std(n_test_list) if len(n_test_list) else 0,
    }
    
    return {"averages": averages, "folds": fold_list}


def preprocess_numeric_data(X, y=None):
    """Convert features to numeric and fill missing values. Optionally encode y for classification."""
    X_numeric = X.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    medians = X_numeric.median(skipna=True).fillna(0)
    X_numeric = X_numeric.fillna(medians)
    
    if y is not None:
        if not pd.api.types.is_numeric_dtype(y):
            if y.dtype == 'object' or hasattr(y, 'cat'):
                y_numeric = LabelEncoder().fit_transform(y)
            else:
                y_numeric = pd.to_numeric(y, errors="coerce").astype(float).to_numpy()
        else:
            y_numeric = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        return X_numeric, y_numeric
    
    return X_numeric


def encode_categorical_target(y):
    """Encode non-numeric target labels to integers."""
    if not pd.api.types.is_numeric_dtype(y):
        return LabelEncoder().fit_transform(y)
    return y


def export_results_to_excel(results_df, filename, output_dir="outputs", sheet_name="results"):
    """Write a DataFrame to an Excel file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    try:
        xlsx_file = out_dir / filename
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved Excel workbook to: {xlsx_file}")
        return xlsx_file
    except Exception as e:
        print(f"Could not write Excel file (openpyxl may be missing): {e}")
        return None


def export_cv_results(cv_results_dict, base_filename, output_dir="outputs"):
    """Export CV results (summary and per-fold metrics) to an Excel workbook."""
    # Create summary DataFrame
    summary_data = []
    for model_name, results in cv_results_dict.items():
        row = {"Model": model_name}
        row.update(results["averages"])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)

    # Build per-fold DataFrame
    folds_frames = []
    for model_name, results in cv_results_dict.items():
        folds = results.get("folds", [])
        if folds:
            df_folds = pd.DataFrame(folds)
            df_folds.insert(0, "Model", model_name)
            folds_frames.append(df_folds)

    if folds_frames:
        folds_df = pd.concat(folds_frames, ignore_index=True)
    else:
        folds_df = pd.DataFrame()

    # Export to Excel
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    xlsx_file = out_dir / f"{base_filename}.xlsx"
    try:
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            if not folds_df.empty:
                folds_df.to_excel(writer, sheet_name="folds", index=False)
        print(f"Saved Excel workbook to: {xlsx_file}")
    except Exception as e:
        print(f"Could not write Excel file (openpyxl may be missing): {e}")

    return summary_df


def print_cv_summary(cv_results, model_name="Model"):
    """Print a summary of cross-validation results."""
    print(f"\n{'='*60}")
    print(f"{model_name} - Cross-Validation Summary")
    print(f"{'='*60}")
    
    averages = cv_results["averages"]
    
    for key, value in averages.items():
        if not key.endswith("_std"):
            metric_name = key.replace("_mean", "")
            mean_val = value
            std_key = f"{metric_name}_std"
            std_val = averages.get(std_key, 0)
            
            print(f"  {metric_name:20s}: {mean_val:8.4f} ± {std_val:.4f}")
    
    print(f"{'='*60}")


def get_model_factory(model_name, task="classification"):
    """Get a model factory function by name for the given task."""
    model_name = model_name.lower()
    
    if task == "classification":
        factories = {
            "tabpfn": create_tabpfn_classifier,
            "tabflex": create_tabflex_classifier,
            "lightgbm": create_lightgbm_classifier,
            "xgboost": create_xgboost_classifier,
            "catboost": create_catboost_classifier,
            "fttransformer": create_ft_transformer_classifier,
            "ft-transformer": create_ft_transformer_classifier,
        }
    else:
        factories = {
            "tabpfn": create_tabpfn_regressor,
            "lightgbm": create_lightgbm_regressor,
            "xgboost": create_xgboost_regressor,
            "catboost": create_catboost_regressor,
        }
    
    return factories.get(model_name)