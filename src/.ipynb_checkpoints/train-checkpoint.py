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
  6. Feature importance analysis     (GB native + Permutation + SHAP)
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
from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                      cross_val_score, RandomizedSearchCV,
                                      learning_curve)
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection       import permutation_importance

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_pipeline

# ── OPTIONAL: XGBoost ─────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


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
    candidates = {
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
                class_weight='balanced', n_jobs=-1, random_state=42),
            'params': {
                'n_estimators':    [200, 300, 400],
                'max_depth':       [8, 10, 12, None],
                'min_samples_leaf':[1, 2, 4],
                'max_features':    ['sqrt', 'log2'],
            }
        },
    }
    if XGBOOST_AVAILABLE:
        candidates['XGBoost'] = {
            'model': XGBClassifier(
                eval_metric='logloss', use_label_encoder=False,
                n_jobs=-1, random_state=42),
            'params': {
                'n_estimators':  [100, 200, 300],
                'max_depth':     [3, 4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample':     [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 1.0],
            }
        }
    return candidates


# ══════════════════════════════════════════════════════════════
# 2 ▸ HYPERPARAMETER TUNING + CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════

def tune_and_compare(X_train, y_train, candidates: dict, cv) -> dict:
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING  (RandomizedSearchCV, 5-Fold CV)")
    print("=" * 60)

    results = {}
    for name, spec in candidates.items():
        print(f"\n  ▸ Tuning {name} ...")
        t0 = time.time()
        search = RandomizedSearchCV(
            spec['model'], spec['params'],
            n_iter=20, cv=cv, scoring='accuracy',
            n_jobs=-1, random_state=42, verbose=0
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        best = search.best_estimator_
        cv_scores = cross_val_score(best, X_train, y_train,
                                    cv=cv, scoring='accuracy', n_jobs=-1)
        results[name] = {
            'estimator':   best,
            'best_params': search.best_params_,
            'cv_mean':     cv_scores.mean(),
            'cv_std':      cv_scores.std(),
            'cv_scores':   cv_scores,
            'time_s':      elapsed,
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
    }


# ══════════════════════════════════════════════════════════════
# 4 ▸ FEATURE IMPORTANCE (3 METHODS)
# ══════════════════════════════════════════════════════════════

def compute_importance(model, X_train, y_train, X_test, y_test,
                       feature_cols: list) -> pd.DataFrame:
    """
    Combine:
      ① Native feature_importances_  (model-internal, fast, biased for high-cardinality)
      ② Permutation Importance       (model-agnostic, reliable on test set)
      ③ SHAP values                  (if shap is installed; gold standard)
    """
    print("\n── Computing Feature Importance (3 methods) ──")

    # ① Native GB importance
    native_imp = model.feature_importances_

    # ② Permutation importance (on held-out test set)
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=20, random_state=42, n_jobs=-1
    )
    perm_imp = perm.importances_mean

    # ③ SHAP (optional dependency)
    shap_imp = np.zeros(len(feature_cols))
    shap_available = False
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test)
        # For binary classification, shap_values may be list[2] or 2-d array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_imp   = np.abs(shap_vals).mean(axis=0)
        shap_available = True
        print("  ✓ SHAP values computed")
    except ImportError:
        print("  ⚠ shap not installed — skipping SHAP (pip install shap)")
    except Exception as exc:
        print(f"  ⚠ SHAP failed: {exc}")

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
    print(f"  {'Feature':38s} {'Native':>8} {'Perm':>8} {'SHAP':>8}")
    print(f"  {'─'*38} {'─'*8} {'─'*8} {'─'*8}")
    for _, r in imp_df.iterrows():
        shap_str = f"{r['shap_imp']:.4f}" if shap_available else "  n/a  "
        print(f"  {r['feature']:38s} {r['native_imp']:.4f}   {r['perm_imp']:.4f}   {shap_str}")

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
    colors = [INDIGO, TEAL, AMBER][:len(names)]

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
    if shap_available:
        methods.append(('shap_imp', 'SHAP Mean |φ|', CORAL))

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

    fig.suptitle('Feature Importance Analysis — Three Methods Compared',
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Not Depressed', 'Depressed'])
    cmap = LinearSegmentedColormap.from_list('ind', ['#EEF2FF', INDIGO])
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(f'Confusion Matrix — {model_name}',
                 fontsize=12, fontweight='bold', pad=12)
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
        n_jobs=-1
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
    colors = [INDIGO, TEAL, AMBER][:len(labels)]
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

    # 4 ▸ Tune & compare
    candidates = build_candidates()
    results    = tune_and_compare(X_train, y_train, candidates, cv)

    # 5 ▸ Select best model
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best_clf  = results[best_name]['estimator']
    print(f"\n  ★ Best Model : {best_name}  "
          f"(CV = {results[best_name]['cv_mean']*100:.2f}%)")

    # 6 ▸ Final fit on full train set
    print(f"\n  Training final {best_name} on full training set ...")
    best_clf.fit(X_train, y_train)

    # 7 ▸ Evaluate
    eval_result = evaluate(best_clf, X_test, y_test, best_name)

    # 8 ▸ Feature importance
    imp_df, shap_available = compute_importance(
        best_clf, X_train, y_train, X_test, y_test, feature_cols
    )

    # 9 ▸ Plots
    print("\n── Generating Plots ──")
    plot_cv_comparison(results, PLOTS_DIR)
    plot_cv_scores_distribution(results, PLOTS_DIR)
    plot_feature_importance(imp_df, shap_available, PLOTS_DIR)
    plot_confusion_matrix(best_clf, X_test, y_test, best_name, PLOTS_DIR)
    plot_learning_curve(best_clf, X_train, y_train, best_name, PLOTS_DIR)
    plot_mi_correlation(mi_df, corr_df, PLOTS_DIR)

    # 10 ▸ Save model bundle
    bundle = {
        'model':           best_clf,
        'model_name':      best_name,
        'feature_cols':    feature_cols,
        'importance_df':   imp_df,
        'test_accuracy':   eval_result['accuracy'],
        'cv_accuracy':     results[best_name]['cv_mean'],
        'eval_results':    results,
        'report_dict':     eval_result['report_dict'],
        'sleep_map':       artifacts['sleep_map'],
        'diet_map':        artifacts['diet_map'],
        'shap_available':  shap_available,
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
    print(f"  Features    : {len(feature_cols)}")
    print(f"  Best Model  : {best_name}")
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
