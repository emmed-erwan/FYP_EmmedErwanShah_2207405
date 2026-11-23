from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, classification_report, ConfusionMatrixDisplay
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'rf_results'
OUT.mkdir(exist_ok=True)

def load_class(folder, stem, label):
    X = pd.read_csv(ROOT / folder / f'{stem}_ml_non_normalized.csv', header=None)
    m = pd.read_csv(ROOT / folder / f'{stem}_source_mapping.csv')
    assert len(X) == len(m), f'{stem}: mapping length mismatch'
    assert X.shape[1] == 202, f'{stem}: expected 202 features'
    assert np.isfinite(X.to_numpy(dtype=float)).all(), f'{stem}: missing/nonfinite features'
    assert (m['Video_Label'].str.lower().str.replace('-', '') == ('fleeing' if label else 'nonfleeing')).all()
    return X, pd.Series(label, index=range(len(X))), m['Source_Video'].astype(str).reset_index(drop=True).map(lambda s: f'{label}:{s}')

xf, yf, gf = load_class('fleeing_master', 'fleeing', 1)
xn, yn, gn = load_class('non_fleeing_master', 'non_fleeing', 0)
X = pd.concat([xf, xn], ignore_index=True)
y = pd.concat([yf, yn], ignore_index=True)
groups = pd.concat([gf, gn], ignore_index=True)

# Stratify at SOURCE VIDEO level, not at permutation-row level.
video_table = pd.DataFrame({'group': groups, 'label': y}).drop_duplicates('group')
assert video_table.groupby('group')['label'].nunique().max() == 1
train_groups, test_groups = train_test_split(video_table['group'], test_size=0.20, random_state=42, stratify=video_table['label'])
train_mask = groups.isin(train_groups)
test_mask = groups.isin(test_groups)
assert not (set(groups[train_mask]) & set(groups[test_mask]))

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced', min_samples_leaf=2)
model.fit(X.loc[train_mask], y.loc[train_mask])
pred = model.predict(X.loc[test_mask])
truth = y.loc[test_mask]

metrics = {
    'Accuracy': accuracy_score(truth, pred),
    'Balanced Accuracy': balanced_accuracy_score(truth, pred),
    'F1 (Fleeing)': f1_score(truth, pred, pos_label=1),
    'Fleeing Recall': recall_score(truth, pred, pos_label=1),
}
print('Train source videos:', len(train_groups), 'Test source videos:', len(test_groups))
print('Train rows:', train_mask.sum(), 'Test rows:', test_mask.sum())
for k,v in metrics.items(): print(f'{k}: {v:.4f}')
print('\n', classification_report(truth, pred, target_names=['Non-Fleeing', 'Fleeing']))
pd.Series(metrics, name='Score').to_csv(OUT/'metrics.csv')
results = pd.DataFrame({'Source_Video': groups.loc[test_mask].values, 'True_Label': truth.values, 'Predicted_Label': pred})
results.to_csv(OUT/'test_predictions.csv', index=False)
ConfusionMatrixDisplay.from_predictions(truth, pred, labels=[0,1], display_labels=['Non-Fleeing','Fleeing'], cmap='Blues')
plt.tight_layout(); plt.savefig(OUT/'confusion_matrix.png', dpi=180);plt.close()
joblib.dump(model, OUT/'random_forest_model.joblib')
print('Saved outputs in:', OUT)
