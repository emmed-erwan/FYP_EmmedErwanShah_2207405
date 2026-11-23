"""Compare Random Forest with 4, 8, and 24 fish-order training permutations.
Run from the folder containing fleeing_master/ and non_fleeing_master/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'rf_permutation_comparison'
OUT.mkdir(parents=True, exist_ok=True)

def load_class(folder, stem, label):
    x = pd.read_csv(ROOT / folder / f'{stem}_ml_non_normalized.csv', header=None)
    mapping = pd.read_csv(ROOT / folder / f'{stem}_source_mapping.csv')
    if len(x) != len(mapping) or x.shape[1] != 202:
        raise ValueError(f'{stem}: feature/mapping row count or feature count mismatch')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError(f'{stem}: nonfinite features')
    group = mapping['Source_Video'].astype(str).map(lambda s: f'{label}:{s}')
    if (mapping.groupby('Source_Video').size() != 24).any():
        raise ValueError(f'{stem}: expected 24 permutations per source video')
    if mapping['Permutation_ID'].isna().any():
        raise ValueError(f'{stem}: missing permutation IDs')
    return x, pd.Series(label, index=range(len(x))), group.reset_index(drop=True), mapping['Permutation_ID'].reset_index(drop=True)

xf, yf, gf, pf = load_class('fleeing_master', 'fleeing', 1)
xn, yn, gn, pn = load_class('non_fleeing_master', 'non_fleeing', 0)
X = pd.concat([xf, xn], ignore_index=True)
y = pd.concat([yf, yn], ignore_index=True)
groups = pd.concat([gf, gn], ignore_index=True)
permutation_ids = pd.concat([pf, pn], ignore_index=True)
video_table = pd.DataFrame({'group': groups, 'label': y}).drop_duplicates('group')
train_groups, test_groups = train_test_split(
    video_table['group'], test_size=0.20, random_state=42, stratify=video_table['label']
)
train_mask = groups.isin(train_groups).to_numpy()
test_mask = groups.isin(test_groups).to_numpy()
assert not set(groups[train_mask]).intersection(set(groups[test_mask]))
assert test_mask.sum() == len(test_groups) * 24

# Evaluate each model on the SAME held-out videos and all their 24 permutations.
X_test = X.iloc[np.flatnonzero(test_mask)]
y_test = y.iloc[np.flatnonzero(test_mask)].to_numpy()
g_test = groups.iloc[np.flatnonzero(test_mask)].to_numpy()

def metric_values(actual, predicted):
    return {
        'Accuracy': accuracy_score(actual, predicted),
        'Balanced_Accuracy': balanced_accuracy_score(actual, predicted),
        'F1_Fleeing': f1_score(actual, predicted, pos_label=1, zero_division=0),
        'Fleeing_Recall': recall_score(actual, predicted, pos_label=1, zero_division=0),
    }

summaries = []
video_predictions = []
for n_perms in (4, 8, 24):
    seeds = (42, 123, 2026) if n_perms < 24 else (42,)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen = []
        for group in train_groups:
            indices = np.flatnonzero(train_mask & (groups.to_numpy() == group))
            if len(indices) != 24:
                raise ValueError(f'{group}: expected 24 training permutations, found {len(indices)}')
            if n_perms == 24:
                selected = indices
            else:
                selected = rng.choice(indices, size=n_perms, replace=False)
            chosen.extend(selected)
        chosen = np.array(chosen, dtype=int)
        # Fixed RF settings, as in the baseline script.
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight='balanced', min_samples_leaf=2
        )
        model.fit(X.iloc[chosen], y.iloc[chosen])
        probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        row_pred = model.predict(X_test)
        row_metrics = metric_values(y_test, row_pred)
        rows = pd.DataFrame({
            'Source_Video': g_test, 'Actual': y_test,
            'Fleeing_Probability': probabilities,
        })
        video = rows.groupby('Source_Video', as_index=False).agg(
            Actual=('Actual', 'first'),
            Mean_Fleeing_Probability=('Fleeing_Probability', 'mean'),
            Permutation_Count=('Fleeing_Probability', 'size'),
        )
        assert (video['Permutation_Count'] == 24).all()
        video['Predicted'] = (video['Mean_Fleeing_Probability'] >= 0.5).astype(int)
        video_metrics = metric_values(video['Actual'], video['Predicted'])
        for _, r in video.iterrows():
            video_predictions.append({
                'Training_Permutations': n_perms, 'Seed': seed,
                'Source_Video': r['Source_Video'], 'Actual': int(r['Actual']),
                'Mean_Fleeing_Probability': r['Mean_Fleeing_Probability'],
                'Predicted': int(r['Predicted'])
            })
        result = {'Training_Permutations': n_perms, 'Seed': seed,
                  'Train_Rows': len(chosen), 'Test_Rows': len(y_test),
                  'Test_Videos': len(video)}
        result.update({f'Row_{k}': v for k, v in row_metrics.items()})
        result.update({f'Video_{k}': v for k, v in video_metrics.items()})
        summaries.append(result)
        print(f'{n_perms:>2} permutations | seed {seed:>4} | '
              f'video accuracy {video_metrics["Accuracy"]:.4f} | '
              f'video balanced accuracy {video_metrics["Balanced_Accuracy"]:.4f} | '
              f'video F1 {video_metrics["F1_Fleeing"]:.4f} | '
              f'video fleeing recall {video_metrics["Fleeing_Recall"]:.4f}', flush=True)
        if seed == 42:
            joblib.dump(model, OUT / f'random_forest_{n_perms}perms.joblib')
            cm = confusion_matrix(video['Actual'], video['Predicted'], labels=[0, 1])
            pd.DataFrame(cm, index=['Actual_Non_Fleeing', 'Actual_Fleeing'],
                         columns=['Pred_Non_Fleeing', 'Pred_Fleeing']).to_csv(
                             OUT / f'video_confusion_matrix_{n_perms}perms.csv')

results = pd.DataFrame(summaries)
results.to_csv(OUT / 'all_runs.csv', index=False)
pd.DataFrame(video_predictions).to_csv(OUT / 'video_predictions.csv', index=False)
metric_cols = [c for c in results.columns if c.startswith('Video_') or c.startswith('Row_')]
summary = results.groupby('Training_Permutations')[metric_cols].agg(['mean', 'std'])
summary.to_csv(OUT / 'comparison_summary.csv')
print('\nMean video-level metrics (across random permutation selections):')
print(results.groupby('Training_Permutations')[[
    'Video_Accuracy', 'Video_Balanced_Accuracy', 'Video_F1_Fleeing', 'Video_Fleeing_Recall'
]].mean().round(4).to_string())
print('\nSaved files in:', OUT)
