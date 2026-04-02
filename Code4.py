# =============================================================================
#  CS4053E – Data Mining | Assignment 2
#  Music Genre Classification – Feature Extraction & Classification
#  Group 4 | NIT Calicut | Winter 2025-26
#
#  Members:
#    Atluri Venkata Sai Vignesh Chowdary  B230857CS
#    Adimulam Yaswanth Veera Nagesh       B230755CS
#    Munukuntla Rithvik Reddy             B231114CS
#    Teja Chinthala                       B230267CS
#    Kukkala Sai Dinesh Reddy             B231035CS
#
#  Dataset : GTZAN features_30_sec.csv
#  Run on  : Google Colab / Kaggle / local Python 3.8+
#
#  HOW TO RUN
#  ----------
#  Option A – Google Colab (recommended):
#    1. Upload this .py file to Colab.
#    2. Upload features_30_sec.csv to Colab (or mount Google Drive).
#    3. Run:  !python cs4053e_assignment2_group04.py
#       OR open as notebook: File → Upload notebook → select this .py
#
#  Option B – Kaggle:
#    1. Add GTZAN dataset to the notebook (andradaolteanu/gtzan-dataset-music-genre-classification).
#    2. Run:  !python cs4053e_assignment2_group04.py
#
#  Option C – Local:
#    1. Place features_30_sec.csv in the same folder as this script.
#    2. pip install scikit-learn pandas numpy matplotlib
#    3. python cs4053e_assignment2_group04.py
#
#  Outputs saved to ./output_04/ :
#    pca_scree_04.png, selectkbest_scores_04.png, accuracy_bar_04.png,
#    grouped_metrics_04.png, cv_vs_test_04.png, confusion_matrix_04.png,
#    metrics_detailed_04.csv, metrics_summary_04.csv, best_params_04.csv
# =============================================================================

# ── Step 1 – Import Libraries ─────────────────────────────────────────────────
import json
import textwrap
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE = 42
TEAM         = '04'
OUTPUT_DIR   = Path('output_04')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('All libraries imported successfully!')

# ── Step 2 – Load Dataset ─────────────────────────────────────────────────────
# Search order: Kaggle input path → current directory → Google Drive mount
_search_paths = [
    Path('/kaggle/input'),                          # Kaggle
    Path('/content/drive/MyDrive'),                 # Google Drive (Colab)
    Path('/content'),                               # Colab upload
    Path('.'),                                      # local / same folder
]

CSV_PATH = None
for base in _search_paths:
    found = list(base.glob('**/features_30_sec.csv'))
    if found:
        CSV_PATH = found[0]
        break

if CSV_PATH is None:
    raise FileNotFoundError(
        'features_30_sec.csv not found.\n'
        'Please place features_30_sec.csv in the same folder as this script,\n'
        'or upload it to Colab / add the Kaggle dataset.'
    )

print('Dataset path  :', CSV_PATH)
df = pd.read_csv(CSV_PATH)
print(f'Shape         : {df.shape}')
print(f'Genres        : {sorted(df["label"].unique())}')
print(f'Samples/genre :\n{df["label"].value_counts().sort_index().to_string()}')

# ── Step 3 – Preprocessing ───────────────────────────────────────────────────
drop_cols = [c for c in ['filename', 'length', 'label'] if c in df.columns]
X      = df.drop(columns=drop_cols)
y_text = df['label']

le = LabelEncoder()
y  = le.fit_transform(y_text)

print(f'Feature matrix : {X.shape}')
print(f'Classes        : {list(le.classes_)}')
print(f'Missing values : {int(X.isnull().sum().sum())}')

if X.isnull().sum().sum() > 0:
    raise ValueError('Missing values detected – clean dataset before continuing.')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f'Train : {X_train_raw.shape[0]} samples')
print(f'Test  : {X_test_raw.shape[0]} samples')

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)
print(f'Mean (train, after scaling) : {X_train_scaled.mean():.6f}')
print(f'Std  (train, after scaling) : {X_train_scaled.std():.6f}')
print('Standardisation complete.')

# ── Step 4 – Feature Extraction Method 1: PCA ────────────────────────────────
pca         = PCA(n_components=0.90, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print(f'Original features       : {X_train_scaled.shape[1]}')
print(f'PCA components retained : {X_train_pca.shape[1]}')
print(f'Variance explained      : {np.sum(pca.explained_variance_ratio_)*100:.2f}%')

# Scree plot
indiv  = pca.explained_variance_ratio_ * 100
cumvar = np.cumsum(indiv)

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(range(1, len(indiv)+1), indiv, color='steelblue', alpha=0.75,
        label='Individual variance')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Individual Explained Variance (%)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(range(1, len(cumvar)+1), cumvar, color='crimson', marker='o',
         markersize=3, linewidth=1.8, label='Cumulative variance')
ax2.axhline(90, color='orange', linestyle='--', linewidth=1.2, label='90% threshold')
ax2.set_ylabel('Cumulative Explained Variance (%)', color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')
ax2.set_ylim(0, 105)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='center right', fontsize=9)
plt.title('PCA – Individual & Cumulative Explained Variance')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'pca_scree_{TEAM}.png', dpi=150)
plt.show()
print('PCA scree plot saved.')

# ── Step 5 – Feature Extraction Method 2: SelectKBest (ANOVA F-test) ─────────
# Fit with k=20 for visualisation only – k is tuned inside each GridSearch
sel_vis = SelectKBest(score_func=f_classif, k=20)
sel_vis.fit(X_train_scaled, y_train)

feat_names  = X.columns.tolist()
all_scores  = sel_vis.scores_
top20_idx   = np.argsort(all_scores)[::-1][:20]
top20_names = [feat_names[i] for i in top20_idx]
top20_scr   = all_scores[top20_idx]

print('Top 20 features by ANOVA F-score:')
for rank, (fname, fscore) in enumerate(zip(top20_names, top20_scr), 1):
    print(f'  {rank:2d}. {fname:<35s}  F = {fscore:.1f}')

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(20), top20_scr, color='darkorange', edgecolor='black', linewidth=0.5)
ax.set_xticks(range(20))
ax.set_xticklabels(top20_names, rotation=75, ha='right', fontsize=8)
ax.set_ylabel('ANOVA F-score')
ax.set_title('SelectKBest – Top 20 Features by ANOVA F-score')
for bar, s in zip(bars, top20_scr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{s:.0f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'selectkbest_scores_{TEAM}.png', dpi=150)
plt.show()
print('F-score chart saved.')

# ── Step 6 – Grid Search Setup ───────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate(model, X_te, y_te):
    """Run model on test set; return metrics dict (values as %) + y_pred."""
    y_pred = model.predict(X_te)
    return {
        'accuracy'  : accuracy_score(y_te, y_pred) * 100,
        'precision' : precision_score(y_te, y_pred, average='macro', zero_division=0) * 100,
        'recall'    : recall_score(y_te, y_pred,    average='macro', zero_division=0) * 100,
        'f1'        : f1_score(y_te, y_pred,        average='macro', zero_division=0) * 100,
        'y_pred'    : y_pred,
    }

def clean_params(params):
    """Strip Pipeline step prefixes (clf__, skb__) from parameter names."""
    return {k.split('__')[-1]: v for k, v in params.items()}

results = {}   # (clf_name, feat_method) -> metrics dict
print('Setup complete. Starting experiments...')

# ── Step 7 – Classifier 1: Decision Tree ─────────────────────────────────────
dt_params = {
    'criterion'         : ['gini', 'entropy'],
    'max_depth'         : [5, 10, 15, 20, None],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf'  : [1, 2, 4],
}

print('=== Decision Tree + PCA ===')
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_params, cv=cv, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train_pca, y_train)
m = evaluate(gs.best_estimator_, X_test_pca, y_test)
results[('Decision Tree', 'PCA')] = {
    **m, 'best_params': gs.best_params_, 'cv_acc': gs.best_score_ * 100
}
print(f'  Best params : {gs.best_params_}')
print(f'  CV  Acc     : {gs.best_score_*100:.2f}%')
print(f'  Test Acc    : {m["accuracy"]:.2f}%  |  Prec: {m["precision"]:.2f}%  '
      f'|  Rec: {m["recall"]:.2f}%  |  F1: {m["f1"]:.2f}%')

print('=== Decision Tree + SelectKBest ===')
pipe = Pipeline([
    ('skb', SelectKBest(score_func=f_classif)),
    ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE)),
])
gs = GridSearchCV(
    pipe,
    {
        'skb__k'                 : [10, 20, 30, 40],
        'clf__criterion'         : ['gini', 'entropy'],
        'clf__max_depth'         : [5, 10, 15, 20, None],
        'clf__min_samples_split' : [2, 5, 10],
        'clf__min_samples_leaf'  : [1, 2, 4],
    },
    cv=cv, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train_scaled, y_train)
m = evaluate(gs.best_estimator_, X_test_scaled, y_test)
results[('Decision Tree', 'SelectKBest')] = {
    **m, 'best_params': clean_params(gs.best_params_), 'cv_acc': gs.best_score_ * 100
}
print(f'  Best params : {clean_params(gs.best_params_)}')
print(f'  CV  Acc     : {gs.best_score_*100:.2f}%')
print(f'  Test Acc    : {m["accuracy"]:.2f}%  |  Prec: {m["precision"]:.2f}%  '
      f'|  Rec: {m["recall"]:.2f}%  |  F1: {m["f1"]:.2f}%')

# ── Step 8 – Classifier 2: K-Nearest Neighbours ──────────────────────────────
knn_params = {
    'n_neighbors' : [3, 5, 7, 9, 11],
    'weights'     : ['uniform', 'distance'],
    'metric'      : ['euclidean', 'manhattan'],
}

print('=== KNN + PCA ===')
gs = GridSearchCV(
    KNeighborsClassifier(),
    knn_params, cv=cv, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train_pca, y_train)
m = evaluate(gs.best_estimator_, X_test_pca, y_test)
results[('KNN', 'PCA')] = {
    **m, 'best_params': gs.best_params_, 'cv_acc': gs.best_score_ * 100
}
print(f'  Best params : {gs.best_params_}')
print(f'  CV  Acc     : {gs.best_score_*100:.2f}%')
print(f'  Test Acc    : {m["accuracy"]:.2f}%  |  Prec: {m["precision"]:.2f}%  '
      f'|  Rec: {m["recall"]:.2f}%  |  F1: {m["f1"]:.2f}%')

print('=== KNN + SelectKBest ===')
pipe = Pipeline([
    ('skb', SelectKBest(score_func=f_classif)),
    ('clf', KNeighborsClassifier()),
])
gs = GridSearchCV(
    pipe,
    {
        'skb__k'          : [10, 20, 30, 40],
        'clf__n_neighbors': [3, 5, 7, 9, 11],
        'clf__weights'    : ['uniform', 'distance'],
        'clf__metric'     : ['euclidean', 'manhattan'],
    },
    cv=cv, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train_scaled, y_train)
m = evaluate(gs.best_estimator_, X_test_scaled, y_test)
results[('KNN', 'SelectKBest')] = {
    **m, 'best_params': clean_params(gs.best_params_), 'cv_acc': gs.best_score_ * 100
}
print(f'  Best params : {clean_params(gs.best_params_)}')
print(f'  CV  Acc     : {gs.best_score_*100:.2f}%')
print(f'  Test Acc    : {m["accuracy"]:.2f}%  |  Prec: {m["precision"]:.2f}%  '
      f'|  Rec: {m["recall"]:.2f}%  |  F1: {m["f1"]:.2f}%')

# ── Step 9 – Classifier 3: Gaussian Naïve Bayes ──────────────────────────────
nb_params = {'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}

print('=== Naive Bayes + PCA ===')
gs = GridSearchCV(
    GaussianNB(),
    nb_params, cv=cv, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train_pca, y_train)
m = evaluate(gs.best_estimator_, X_test_pca, y_test)
results[('Naive Bayes', 'PCA')] = {
    **m, 'best_params': gs.best_params_, 'cv_acc': gs.best_score_ * 100
}
print(f'  Best params : {gs.best_params_}')
print(f'  CV  Acc     : {gs.best_score_*100:.2f}%')
print(f'  Test Acc    : {m["accuracy"]:.2f}%  |  Prec: {m["precision"]:.2f}%  '
      f'|  Rec: {m["recall"]:.2f}%  |  F1: {m["f1"]:.2f}%')

print('=== Naive Bayes + SelectKBest ===')
pipe = Pipeline([
    ('skb', SelectKBest(score_func=f_classif)),
    ('clf', GaussianNB()),
])
gs = GridSearchCV(
    pipe,
    {
        'skb__k'             : [10, 20, 30, 40],
        'clf__var_smoothing' : [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    },
    cv=cv, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train_scaled, y_train)
m = evaluate(gs.best_estimator_, X_test_scaled, y_test)
results[('Naive Bayes', 'SelectKBest')] = {
    **m, 'best_params': clean_params(gs.best_params_), 'cv_acc': gs.best_score_ * 100
}
print(f'  Best params : {clean_params(gs.best_params_)}')
print(f'  CV  Acc     : {gs.best_score_*100:.2f}%')
print(f'  Test Acc    : {m["accuracy"]:.2f}%  |  Prec: {m["precision"]:.2f}%  '
      f'|  Rec: {m["recall"]:.2f}%  |  F1: {m["f1"]:.2f}%')

# ── Step 10 – Full Results Summary ───────────────────────────────────────────
SEP = '=' * 105
print(SEP)
print('COMPLETE RESULTS SUMMARY  –  CS4053E Data Mining Assignment 2  –  Group 04')
print(SEP)
print(f'{"Classifier":<16} {"Feature Method":<16} {"CV Acc":>9} {"Test Acc":>10}'
      f' {"Precision":>11} {"Recall":>9} {"F1":>9}')
print('-' * 105)
for (clf, feat), v in results.items():
    print(f'{clf:<16} {feat:<16} {v["cv_acc"]:>8.2f}% {v["accuracy"]:>9.2f}%'
          f' {v["precision"]:>10.2f}% {v["recall"]:>8.2f}% {v["f1"]:>8.2f}%')
print(SEP)

best_key = max(results, key=lambda k: results[k]['accuracy'])
best     = results[best_key]
print(f'\n★  Best Model   : {best_key[0]} + {best_key[1]}')
print(f'   Best Params  : {best["best_params"]}')
print(f'   Accuracy     : {best["accuracy"]:.2f}%')
print(f'   Precision    : {best["precision"]:.2f}%  (macro-avg)')
print(f'   Recall       : {best["recall"]:.2f}%  (macro-avg)')
print(f'   F1-Score     : {best["f1"]:.2f}%  (macro-avg)')

# ── Step 11 – Classification Report (Best Model) ─────────────────────────────
print(f'\nDetailed Classification Report – {best_key[0]} + {best_key[1]} (Best Model)\n')
print(classification_report(
    y_test,
    results[best_key]['y_pred'],
    target_names=le.classes_
))

# ── Step 12 – Confusion Matrix (Best Model) ──────────────────────────────────
cm = confusion_matrix(y_test, results[best_key]['y_pred'])

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title(f'Confusion Matrix – {best_key[0]} + {best_key[1]} (Best Model)',
             fontsize=13, pad=12)
ax.set_xlabel('Predicted Genre', fontsize=11)
ax.set_ylabel('True Genre', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'confusion_matrix_{TEAM}.png', dpi=150)
plt.show()

print('\nPer-genre accuracy:')
for idx, genre in enumerate(le.classes_):
    correct = cm[idx, idx]
    total   = cm[idx].sum()
    print(f'  {genre:<12s}: {correct}/{total}  ({correct/total*100:.1f}%)')
print('\nConfusion matrix saved.')

# ── Step 13 – Visualisations ─────────────────────────────────────────────────
# Plot 1: Test Accuracy comparison
bar_labels = [f'{clf}\n+{feat}' for (clf, feat) in results]
bar_accs   = [v['accuracy'] for v in results.values()]
bar_colors = ['#4C72B0' if 'PCA' in lbl else '#DD8452' for lbl in bar_labels]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(bar_labels, bar_accs, color=bar_colors, edgecolor='black',
              linewidth=0.5, width=0.5)
for bar, a in zip(bars, bar_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{a:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(10, color='gray', linestyle=':', linewidth=1, alpha=0.6,
           label='Random baseline (10%)')
ax.set_ylim(0, 88)
ax.set_ylabel('Test Accuracy (%)', fontsize=11)
ax.set_title('Test Accuracy by Classifier & Feature Extraction Method', fontsize=12)
ax.legend(handles=[
    mpatches.Patch(color='#4C72B0', label='PCA'),
    mpatches.Patch(color='#DD8452', label='SelectKBest'),
], fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'accuracy_bar_{TEAM}.png', dpi=150)
plt.show()
print('Accuracy bar chart saved.')

# Plot 2: Grouped Accuracy / Precision / Recall / F1
grp_labels = [f'{clf}\n+{feat[:3]}' for (clf, feat) in results]
g_accs  = [v['accuracy']  for v in results.values()]
g_precs = [v['precision'] for v in results.values()]
g_recs  = [v['recall']    for v in results.values()]
g_f1s   = [v['f1']        for v in results.values()]

x, w = np.arange(len(grp_labels)), 0.20

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - 1.5*w, g_accs,  w, label='Accuracy',  color='#4C72B0',
       edgecolor='black', linewidth=0.5)
ax.bar(x - 0.5*w, g_precs, w, label='Precision', color='#55A868',
       edgecolor='black', linewidth=0.5)
ax.bar(x + 0.5*w, g_recs,  w, label='Recall',    color='#C44E52',
       edgecolor='black', linewidth=0.5)
ax.bar(x + 1.5*w, g_f1s,   w, label='F1-Score',  color='#8172B2',
       edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(grp_labels, fontsize=9)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_ylim(0, 88)
ax.set_title('Accuracy / Precision / Recall / F1 per Model', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'grouped_metrics_{TEAM}.png', dpi=150)
plt.show()
print('Grouped metrics chart saved.')

# Plot 3: CV Accuracy vs Test Accuracy
cv_accs    = [v['cv_acc']   for v in results.values()]
test_accs  = [v['accuracy'] for v in results.values()]
clf_labels = [f'{clf}\n+{feat[:3]}' for (clf, feat) in results]

x, w = np.arange(len(clf_labels)), 0.35

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - w/2, cv_accs,   w, label='CV Accuracy (5-fold)',
            color='#4C72B0', alpha=0.85, edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + w/2, test_accs, w, label='Test Accuracy (holdout)',
            color='#DD8452', alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, val in [(b, v) for bars, vals in [(b1, cv_accs), (b2, test_accs)]
                 for b, v in zip(bars, vals)]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(clf_labels, fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 85)
ax.set_title('CV Accuracy vs Test Accuracy - Generalisation Check', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'cv_vs_test_{TEAM}.png', dpi=150)
plt.show()
print('CV vs Test accuracy chart saved.')

# ── Step 14 – Save CSV Output Files ──────────────────────────────────────────
rows = []
for (clf, feat), v in results.items():
    rows.append({
        'feature_method'       : feat,
        'classifier'           : clf,
        'cv_best_accuracy'     : round(v['cv_acc']    / 100, 4),
        'test_accuracy'        : round(v['accuracy']  / 100, 4),
        'test_precision_macro' : round(v['precision'] / 100, 4),
        'test_recall_macro'    : round(v['recall']    / 100, 4),
        'test_f1_macro'        : round(v['f1']        / 100, 4),
        'best_params'          : json.dumps(v['best_params'], sort_keys=True, default=str),
    })

results_df = pd.DataFrame(rows)

summary_df = (
    results_df[['feature_method', 'classifier', 'test_accuracy',
                'test_precision_macro', 'test_recall_macro', 'test_f1_macro']]
    .sort_values(['feature_method', 'test_accuracy'], ascending=[True, False])
    .reset_index(drop=True)
)

params_df = results_df[['feature_method', 'classifier', 'best_params']].copy()

results_df.to_csv(OUTPUT_DIR / f'metrics_detailed_{TEAM}.csv', index=False)
summary_df.to_csv(OUTPUT_DIR / f'metrics_summary_{TEAM}.csv',  index=False)
params_df.to_csv( OUTPUT_DIR / f'best_params_{TEAM}.csv',      index=False)

print('CSV files saved.')
print('\n-- Performance Summary --')
print(summary_df.to_string(index=False))
print('\n-- Best Parameters --')
print(params_df.to_string(index=False))

# ── Done ──────────────────────────────────────────────────────────────────────
print('\n' + '='*55)
print('  ALL OUTPUTS SAVED TO:', OUTPUT_DIR.resolve())
print('='*55)
print(f'  pca_scree_{TEAM}.png')
print(f'  selectkbest_scores_{TEAM}.png')
print(f'  accuracy_bar_{TEAM}.png')
print(f'  grouped_metrics_{TEAM}.png')
print(f'  cv_vs_test_{TEAM}.png')
print(f'  confusion_matrix_{TEAM}.png')
print(f'  metrics_detailed_{TEAM}.csv')
print(f'  metrics_summary_{TEAM}.csv')
print(f'  best_params_{TEAM}.csv')
print('='*55)
