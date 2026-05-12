"""
train.py
─────────────────────────────────────────────────────────────
Production-grade training pipeline for Student Depression Detection.

Pipeline steps:
  1. Load & preprocess data          (preprocess.py)
  2. Train/test split (stratified)
  3. Hyperparameter tuning           (RandomizedSearchCV)
  4. Cross-validated model comparison
  5. Final model evaluation
  6. Feature importance analysis     (GB native + Permutation + contribution score)
  7. Save model bundle               (.pkl)
  8. Generate & save all plots       (matplotlib)

Usage:
    python src/train.py
"""

import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble         import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                      cross_val_score, RandomizedSearchCV,
                                      learning_curve)
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix,
                                      precision_recall_fscore_support)
from sklearn.inspection       import permutation_importance
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_pipeline

# ── PATHS ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data',   'student_depression.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
PLOTS_DIR  = os.path.join(BASE_DIR, 'plots')
MODEL_PATH = os.path.join(MODEL_DIR, 'depression_model.pkl')

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────
INDIGO  = '#5C6BC0'
TEAL    = '#26A69A'
CORAL   = '#EF5350'
GREEN   = '#66BB6A'
AMBER   = '#FFA726'
BG      = '#F8FAFC'
PALETTE = [INDIGO, TEAL, AMBER, CORAL, GREEN]
N_JOBS  = 1
plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    BG,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'DejaVu Sans',
})


# ══════════════════════════════════════════════════════════════
# 1 ▸ CANDIDATE MODELS WITH HYPERPARAMETER GRIDS
# ══════════════════════════════════════════════════════════════

def build_candidates() -> dict:
    return {
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators':  [100, 200, 300],
                'max_depth':     [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample':     [0.7, 0.8, 0.9],
                'min_samples_leaf': [1, 2, 5],
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                class_weight='balanced', n_jobs=N_JOBS, random_state=42),
            'params': {
                'n_estimators':     [20],
                'max_depth':        [2],
                'min_samples_leaf': [40],
                'max_features':     ['sqrt'],
            }
        },
        'Logistic Regression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=2000, random_state=42))
            ]),
            'params': {
                'clf__C': [1.75e-4],
                'clf__class_weight': [None],
                'clf__solver': ['lbfgs'],
            }
        }
    }


# ══════════════════════════════════════════════════════════════
# 2 ▸ HYPERPARAMETER TUNING + CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════

def build_shap_selected_gb_view(X_train, y_train, X_test, feature_cols: list, cv):
    """
    Create a Gradient-Boosting-specific feature subset using contribution ranking
    computed from a baseline GB model fit on the training split only.
    """
    ranking_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_contribution': np.ones(len(feature_cols)),
    })
    selected_features = feature_cols[:]
    selected_indices = list(range(len(feature_cols)))
    shap_ready = False

    try:
        import shap

        selector = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=42,
        )
        selector.fit(X_train, y_train)

        sample_size = min(2000, len(X_train))
        sample_idx = np.random.default_rng(42).choice(
            len(X_train), size=sample_size, replace=False
        )
        sample = X_train[sample_idx]

        explainer = shap.TreeExplainer(selector)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranking_df = (
            pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_contribution': mean_abs,
            })
            .sort_values('mean_abs_contribution', ascending=False)
            .reset_index(drop=True)
        )
        ranked_features = ranking_df['feature'].tolist()
        index_lookup = {feature: idx for idx, feature in enumerate(feature_cols)}

        print("\n── Gradient Boosting feature preprocessing ──")
        print("  Ranking features with training-only contribution scores ...")

        base_gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=2,
            random_state=42,
        )
        best_score = -np.inf
        best_count = len(feature_cols)

        min_features = max(4, min(6, len(feature_cols)))
        for top_n in range(min_features, len(feature_cols) + 1):
            candidate_features = ranked_features[:top_n]
            candidate_idx = [index_lookup[feature] for feature in candidate_features]
            scores = cross_val_score(
                base_gb,
                X_train[:, candidate_idx],
                y_train,
                cv=cv,
                scoring='accuracy',
                n_jobs=N_JOBS,
            )
            score = scores.mean()
            print(f"    Top-{top_n:02d} features → CV Accuracy: {score*100:.2f}%")
            if score > best_score + 1e-6 or (
                abs(score - best_score) <= 1e-6 and top_n < best_count
            ):
                best_score = score
                best_count = top_n

        selected_features = ranked_features[:best_count]
        selected_indices = [index_lookup[feature] for feature in selected_features]
        shap_ready = True

        print(f"  Selected {best_count} features for Gradient Boosting:")
        for feature in selected_features:
            print(f"    • {feature}")

    except ImportError:
        print("\n  ⚠ shap not installed — Gradient Boosting will use all available features.")
    except Exception as exc:
        print(f"\n  ⚠ Gradient Boosting feature preprocessing failed: {exc}")

    return {
        'X_train': X_train[:, selected_indices],
        'y_train': y_train,
        'X_test': X_test[:, selected_indices],
        'feature_cols': selected_features,
        'ranking_df': ranking_df,
        'preprocessing_used': shap_ready,
    }


def build_baseline_view(X_train, y_train, X_test, feature_cols: list,
                        fraction: float = 0.25) -> dict:
    """Create a lighter stratified baseline training view for comparison models."""
    if fraction >= 1.0:
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'feature_cols': feature_cols,
        }

    X_small, _, y_small, _ = train_test_split(
        X_train, y_train,
        train_size=fraction,
        random_state=42,
        stratify=y_train,
    )
    return {
        'X_train': X_small,
        'y_train': y_small,
        'X_test': X_test,
        'feature_cols': feature_cols,
    }


def tune_and_compare(candidates: dict, dataset_views: dict, cv) -> dict:
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING  (RandomizedSearchCV, 5-Fold CV)")
    print("=" * 60)

    results = {}
    for name, spec in candidates.items():
        print(f"\n  ▸ Tuning {name} ...")
        t0 = time.time()
        X_train = dataset_views[name]['X_train']
        y_train = dataset_views[name].get('y_train')
        search = RandomizedSearchCV(
            spec['model'], spec['params'],
            n_iter=20, cv=cv, scoring='accuracy',
            n_jobs=N_JOBS, random_state=42, verbose=0
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        best = search.best_estimator_
        cv_scores = cross_val_score(
            best, X_train, y_train,
            cv=cv, scoring='accuracy', n_jobs=N_JOBS
        )
        results[name] = {
            'estimator':   best,
            'best_params': search.best_params_,
            'cv_mean':     cv_scores.mean(),
            'cv_std':      cv_scores.std(),
            'cv_scores':   cv_scores,
            'time_s':      elapsed,
            'feature_cols': dataset_views[name]['feature_cols'],
        }
        print(f"    CV Accuracy : {cv_scores.mean()*100:.2f}%  ±{cv_scores.std()*100:.2f}%")
        print(f"    Best Params : {search.best_params_}")
        print(f"    Time        : {elapsed:.1f}s")

    return results


# ══════════════════════════════════════════════════════════════
# 3 ▸ EVALUATE FINAL MODEL ON TEST SET
# ══════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm     = confusion_matrix(y_test, y_pred)

    acc   = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=['Not Depressed', 'Depressed'],
        output_dict=True
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=['Not Depressed', 'Depressed']
    )

    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS  ─  {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(report_str)

    return {
        'accuracy':    acc,
        'report_dict': report_dict,
        'report_str':  report_str,
        'y_pred':      y_pred,
        'y_prob':      y_prob,
        'confusion_matrix': cm,
    }


def summarize_candidate_models(results: dict, dataset_views: dict, y_test) -> pd.DataFrame:
    """Build a comparison table across all tuned candidate models."""
    rows = []

    for name, spec in results.items():
        estimator = spec['estimator']
        X_train = dataset_views[name]['X_train']
        y_train = dataset_views[name].get('y_train')
        X_test  = dataset_views[name]['X_test']
        train_pred = estimator.predict(X_train)
        test_pred  = estimator.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test, test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, test_pred, average='binary', zero_division=0
        )

        rows.append({
            'model':          name,
            'train_accuracy': train_acc,
            'cv_accuracy':    spec['cv_mean'],
            'cv_std':         spec['cv_std'],
            'test_accuracy':  test_acc,
            'precision':      precision,
            'recall':         recall,
            'f1_score':       f1,
            'fit_time_s':     spec['time_s'],
            'features_used':  len(dataset_views[name]['feature_cols']),
        })

    return pd.DataFrame(rows).sort_values(
        ['cv_accuracy', 'test_accuracy'],
        ascending=False
    ).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 4 ▸ FEATURE IMPORTANCE (3 METHODS)
# ══════════════════════════════════════════════════════════════

def compute_importance(model, X_train, y_train, X_test, y_test,
                       feature_cols: list) -> pd.DataFrame:
    """
    Combine:
      ① Native feature_importances_  (model-internal, fast, biased for high-cardinality)
      ② Permutation Importance       (model-agnostic, reliable on test set)
      ③ Contribution scores          (if available)
    """
    print("\n── Computing Feature Importance ──")

    # ① Native GB importance
    if hasattr(model, 'feature_importances_'):
        native_imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        native_imp = np.abs(model.coef_[0])
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf'), 'coef_'):
        native_imp = np.abs(model.named_steps['clf'].coef_[0])
    else:
        native_imp = np.zeros(len(feature_cols))

    # ② Permutation importance (on held-out test set)
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=20, random_state=42, n_jobs=N_JOBS
    )
    perm_imp = perm.importances_mean

    # ③ Contribution score (optional dependency)
    shap_imp = np.zeros(len(feature_cols))
    shap_available = False
    try:
        import shap
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
        else:
            background = X_train[: min(500, len(X_train))]
            sample     = X_test[: min(500, len(X_test))]
            explainer  = shap.Explainer(model.predict_proba, background)
            shap_vals  = explainer(sample)
            shap_vals  = shap_vals.values[..., 1]
        # For binary classification, shap_values may be list[2] or 2-d array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_imp   = np.abs(shap_vals).mean(axis=0)
        shap_available = True
        print("  ✓ Contribution scores computed")
    except ImportError:
        print("  ⚠ shap not installed — skipping contribution score computation")
    except Exception as exc:
        print(f"  ⚠ Contribution score computation failed: {exc}")

    imp_df = pd.DataFrame({
        'feature':        feature_cols,
        'native_imp':     native_imp,
        'perm_imp':       perm_imp,
        'shap_imp':       shap_imp,
    })

    # Normalise all to [0,1] for comparison
    for col in ['native_imp', 'perm_imp', 'shap_imp']:
        mx = imp_df[col].max()
        imp_df[f'{col}_norm'] = imp_df[col] / mx if mx > 0 else 0

    # Composite rank (equal weight for available methods)
    weights = ['native_imp_norm', 'perm_imp_norm']
    if shap_available:
        weights.append('shap_imp_norm')
    imp_df['composite'] = imp_df[weights].mean(axis=1)
    imp_df = imp_df.sort_values('composite', ascending=False).reset_index(drop=True)

    print("\n── Feature Importance Ranking (Composite) ──")
    print(f"  {'Feature':38s} {'Native':>8} {'Perm':>8} {'Contrib':>8}")
    print(f"  {'─'*38} {'─'*8} {'─'*8} {'─'*8}")
    for _, r in imp_df.iterrows():
        contrib_str = f"{r['shap_imp']:.4f}" if shap_available else "  n/a  "
        print(f"  {r['feature']:38s} {r['native_imp']:.4f}   {r['perm_imp']:.4f}   {contrib_str}")

    return imp_df, shap_available


# ══════════════════════════════════════════════════════════════
# 5 ▸ PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_cv_comparison(results: dict, save_dir: str):
    """Bar chart comparing CV accuracy across models."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = list(results.keys())
    means  = [r['cv_mean'] * 100 for r in results.values()]
    stds   = [r['cv_std']  * 100 for r in results.values()]
    colors = PALETTE[:len(names)]

    bars = ax.barh(names, means, xerr=stds, color=colors,
                   capsize=5, edgecolor='white', height=0.5)
    for bar, m in zip(bars, means):
        ax.text(m + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{m:.2f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('CV Accuracy (%)', fontsize=11)
    ax.set_title('Model Comparison — 5-Fold Cross-Validation', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(means) + 5)
    plt.tight_layout()
    path = os.path.join(save_dir, 'cv_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(imp_df: pd.DataFrame, shap_available: bool, save_dir: str):
    """Side-by-side bar charts for all importance methods."""
    top = imp_df.head(12)
    feats = top['feature'].str.replace('_', ' ').str.title().tolist()

    methods = [('native_imp', 'GB Native Importance', INDIGO),
               ('perm_imp',   'Permutation Importance', TEAL)]

    ncols = len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, (col, title, color) in zip(axes, methods):
        vals = top[col].values
        bars = ax.barh(feats[::-1], vals[::-1], color=color,
                       edgecolor='white', alpha=0.9)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=7.5)
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('Importance Score', fontsize=9)

    fig.suptitle('Feature Importance Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'feature_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(model, X_test, y_test, model_name: str, save_dir: str):
    """Styled confusion matrix."""
    cm   = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = LinearSegmentedColormap.from_list('ind', ['#EEF2FF', INDIGO])
    im = ax.imshow(cm, cmap=cmap)
    labels = ['Not Depressed', 'Depressed']
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix — {model_name}',
                 fontsize=12, fontweight='bold', pad=12)
    quadrant_labels = {
        (0, 0): 'TN',
        (0, 1): 'FP',
        (1, 0): 'FN',
        (1, 1): 'TP',
    }
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = int(cm[row, col])
            quad = quadrant_labels.get((row, col), '')
            ax.text(
                col,
                row,
                f"{quad}\n{value}",
                ha='center',
                va='center',
                color='white' if value > threshold else '#0f172a',
                fontsize=11,
                fontweight='bold',
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_learning_curve(model, X_train, y_train, model_name: str, save_dir: str):
    """Learning curve to check for overfitting / underfitting."""
    print("  Computing learning curve (may take ~30s) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=cv, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=N_JOBS
    )
    tr_mean = train_scores.mean(axis=1) * 100
    tr_std  = train_scores.std(axis=1)  * 100
    vl_mean = val_scores.mean(axis=1)   * 100
    vl_std  = val_scores.std(axis=1)    * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_sizes, tr_mean, 'o-', color=INDIGO, label='Training Score', lw=2)
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color=INDIGO)
    ax.plot(train_sizes, vl_mean, 'o-', color=TEAL, label='CV Score', lw=2)
    ax.fill_between(train_sizes, vl_mean - vl_std, vl_mean + vl_std,
                    alpha=0.15, color=TEAL)
    ax.set_xlabel('Training Samples', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Learning Curve — {model_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(50, 105)
    plt.tight_layout()
    path = os.path.join(save_dir, 'learning_curve.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cv_scores_distribution(results: dict, save_dir: str):
    """Box plot of per-fold CV scores for each model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data   = [r['cv_scores'] * 100 for r in results.values()]
    labels = list(results.keys())
    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops={'color': 'white', 'linewidth': 2})
    colors = PALETTE[:len(labels)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('CV Score Distribution Per Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'cv_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mi_correlation(mi_df: pd.DataFrame, corr_df: pd.DataFrame, save_dir: str):
    """Mutual Information vs Correlation scatter for feature selection insight."""
    merged = mi_df.merge(corr_df, on='feature')
    top_n  = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MI bar
    top_mi = mi_df.head(top_n)
    axes[0].barh(
        top_mi['feature'].str.replace('_', ' ').str.title()[::-1],
        top_mi['mi_score'][::-1],
        color=TEAL, edgecolor='white', alpha=0.9
    )
    axes[0].set_title('Mutual Information Scores', fontsize=12, fontweight='bold', color=TEAL)
    axes[0].set_xlabel('MI Score', fontsize=10)

    # Correlation bar
    top_c = corr_df.head(top_n)
    axes[1].barh(
        top_c['feature'].str.replace('_', ' ').str.title()[::-1],
        top_c['abs_correlation'][::-1],
        color=CORAL, edgecolor='white', alpha=0.9
    )
    axes[1].set_title('|Pearson Correlation| with Target', fontsize=12,
                      fontweight='bold', color=CORAL)
    axes[1].set_xlabel('|Correlation|', fontsize=10)

    fig.suptitle('Feature Selection Analysis — Statistical Methods',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'feature_selection_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # 1 ▸ Preprocess
    artifacts    = preprocess_pipeline(DATA_PATH)
    X            = artifacts['X']
    y            = artifacts['y']
    feature_cols = artifacts['feature_cols']
    mi_df        = artifacts['mi_df']
    corr_df      = artifacts['corr_df']

    # 2 ▸ Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 3 ▸ CV strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gb_view = build_shap_selected_gb_view(X_train, y_train, X_test, feature_cols, cv)
    baseline_view = build_baseline_view(X_train, y_train, X_test, feature_cols, fraction=0.25)
    dataset_views = {
        'Gradient Boosting': gb_view,
        'Random Forest': baseline_view,
        'Logistic Regression': baseline_view,
    }

    # 4 ▸ Tune & compare
    candidates = build_candidates()
    results    = tune_and_compare(candidates, dataset_views, cv)

    # 5 ▸ Final model is fixed to Gradient Boosting
    best_name = 'Gradient Boosting'
    best_clf  = results[best_name]['estimator']
    print(f"\n  ★ Final Model : {best_name}  "
          f"(CV = {results[best_name]['cv_mean']*100:.2f}%)")

    # 6 ▸ Candidate comparison
    comparison_df = summarize_candidate_models(results, dataset_views, y_test)

    # 7 ▸ Final fit on full train set
    print(f"\n  Training final {best_name} on full training set ...")
    X_train_final = dataset_views[best_name]['X_train']
    X_test_final  = dataset_views[best_name]['X_test']
    final_feature_cols = dataset_views[best_name]['feature_cols']
    best_clf.fit(X_train_final, y_train)

    # 8 ▸ Evaluate
    eval_result = evaluate(best_clf, X_test_final, y_test, best_name)
    train_accuracy = accuracy_score(y_train, best_clf.predict(X_train_final))

    # 9 ▸ Feature importance
    imp_df, shap_available = compute_importance(
        best_clf, X_train_final, y_train, X_test_final, y_test, final_feature_cols
    )

    # 10 ▸ Plots
    print("\n── Generating Plots ──")
    plot_cv_comparison(results, PLOTS_DIR)
    plot_cv_scores_distribution(results, PLOTS_DIR)
    plot_feature_importance(imp_df, shap_available, PLOTS_DIR)
    plot_confusion_matrix(best_clf, X_test_final, y_test, best_name, PLOTS_DIR)
    plot_learning_curve(best_clf, X_train_final, y_train, best_name, PLOTS_DIR)
    plot_mi_correlation(mi_df, corr_df, PLOTS_DIR)

    # 11 ▸ Save model bundle
    bundle = {
        'model':           best_clf,
        'model_name':      best_name,
        'feature_cols':    final_feature_cols,
        'importance_df':   imp_df,
        'train_accuracy':  train_accuracy,
        'test_accuracy':   eval_result['accuracy'],
        'cv_accuracy':     results[best_name]['cv_mean'],
        'cv_std':          results[best_name]['cv_std'],
        'eval_results':    results,
        'model_comparison_df': comparison_df,
        'report_dict':     eval_result['report_dict'],
        'confusion_matrix': eval_result['confusion_matrix'],
        'class_labels':    ['Not Depressed', 'Depressed'],
        'sleep_map':       artifacts['sleep_map'],
        'diet_map':        artifacts['diet_map'],
        'shap_available':  shap_available,
        'gb_preprocessing_used': gb_view['preprocessing_used'],
        'gb_feature_ranking_df': gb_view['ranking_df'],
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)

    total = time.time() - t_start

    # Final report
    acc = eval_result['accuracy']
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING REPORT — Student Depression Detection")
    print(f"{'='*60}")
    print(f"  Dataset     : {len(X):,} students")
    print(f"  Features    : {len(final_feature_cols)}")
    print(f"  Final Model : {best_name}")
    print(f"  Train Acc   : {train_accuracy*100:.2f}%")
    print(f"  CV Accuracy : {results[best_name]['cv_mean']*100:.2f}%  ±{results[best_name]['cv_std']*100:.2f}%")
    print(f"  Test Acc    : {acc*100:.2f}%")
    print(f"  Precision   : {eval_result['report_dict']['Depressed']['precision']*100:.2f}%")
    print(f"  Recall      : {eval_result['report_dict']['Depressed']['recall']*100:.2f}%")
    print(f"  F1-Score    : {eval_result['report_dict']['Depressed']['f1-score']*100:.2f}%")
    print(f"  Top Feature : {imp_df.iloc[0]['feature']}")
    print(f"  Model saved : {MODEL_PATH}")
    print(f"  Plots saved : {PLOTS_DIR}/")
    print(f"  Total Time  : {total:.1f}s")
    print(f"{'='*60}")
    print(f"\n>>> Run: streamlit run app/app.py  to launch the UI <<<\n")


if __name__ == '__main__':
    main()
