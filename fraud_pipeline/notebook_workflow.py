"""Portable helpers for the e1155885 notebook and its checked-in release assets."""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'model_assets/lgbm_masked_ensemble'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_release():
    release = json.loads((ASSETS / 'release.json').read_text())
    for name, expected in release['files'].items():
        if sha256(ASSETS / name) != expected:
            raise ValueError(f'Release asset changed: {name}')
    return release


def check_data(directory, release):
    for name, expected in release['data_sha256'].items():
        path = Path(directory) / name
        if not path.exists():
            raise FileNotFoundError(f'Place the competition file at {path}')
        if sha256(path) != expected:
            raise ValueError(f'Competition input differs from the retained release: {name}')


def masked_training_arrays(x, y, config, seed):
    """Reproduce the original two-view masking and per-class sample weights."""
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Training requires observed binary labels')
    columns = [c for c in x if c != 'time_step']
    frames = [x.reset_index(drop=True)]
    for view in range(config['copies']):
        rng = np.random.default_rng(seed + 100003 * view)
        values = x[columns].to_numpy(copy=True)
        values[rng.random(values.shape) < config['rate']] = np.nan
        augmented = x.reset_index(drop=True).copy()
        augmented[columns] = values
        frames.append(augmented)
    class_weights = np.where(y == 1, config['positive_weight'], 1.)
    mass = config['original_mass']
    weights = np.concatenate([mass * class_weights] + [
        (1 - mass) / config['copies'] * class_weights
        for _ in range(config['copies'])])
    return pd.concat(frames, ignore_index=True), np.tile(y, len(frames)), weights


def fit_core(config, fit, features, seed, threads=2):
    """Optional exact-recipe refit of one final masked-core estimator."""
    columns = features + ['time_step']
    x = fit[columns].astype('float32')
    medians = x.median().fillna(0)
    x, y, weights = masked_training_arrays(x.fillna(medians), fit.label, config, seed)
    child = int(40 * (1 + config['copies']) / 2)
    params = dict(objective='binary', metric='auc', learning_rate=.01,
                  num_leaves=config['leaves'], min_data_in_leaf=child,
                  lambda_l2=config['l2'], lambda_l1=.2,
                  feature_fraction=config['feature_fraction'], bagging_fraction=.85,
                  bagging_freq=1, seed=seed, extra_trees=config['extra_trees'],
                  num_threads=threads, deterministic=True, force_col_wise=True, verbosity=-1)
    model = lgb.train(params, lgb.Dataset(x, label=y, weight=weights),
                      num_boost_round=config['trees'])
    return dict(model=model, columns=columns, medians=medians, config=config,
                seed=seed, min_data_in_leaf=child)


def validation_table(train, release):
    """Recompute temporal AUCs from shipped out-of-time prediction snapshots."""
    rows = []
    labels = train.set_index('txId')
    for record in release['historical_forecasts']:
        cutoff = record['cutoff']
        ref = pd.read_csv(ASSETS / f'reference__{cutoff}.csv')
        actual = labels.loc[ref.txId]
        np.testing.assert_array_equal(ref[['time_step', 'label']], actual[['time_step', 'label']])
        assert ref.time_step.gt(cutoff).all()
        parts = []
        for seed in release['seeds']:
            saved = pd.read_csv(ASSETS / f'cache/long_emphasized8__seed{seed}__{cutoff}.csv')
            np.testing.assert_array_equal(ref[['txId', 'time_step', 'label']], saved[['txId', 'time_step', 'label']])
            parts.append(saved[f"trees_{release['trees']}"])
        probability = ((1 - release['weight']) * ref.probability
                       + release['weight'] * np.mean(parts, axis=0))
        score = roc_auc_score(ref.label, probability)
        assert abs(score - record['candidate_auc']) < 1e-12
        rows.append(dict(fit_through=cutoff, score_from=record['start'], score_to=record['end'],
                         anchor_auc=roc_auc_score(ref.label, ref.probability),
                         ensemble_auc=score, gain=score - roc_auc_score(ref.label, ref.probability)))
    return pd.DataFrame(rows)


def make_submission(sample, test, probability, destination):
    probability = np.asarray(probability)
    if len(probability) != len(test) or not np.isfinite(probability).all():
        raise ValueError('Missing or non-finite predictions')
    if not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Predictions must lie between zero and one')
    assert test['index'].is_unique and sample['index'].is_unique
    assert set(test['index']) == set(sample['index'])
    result = sample[['index']].copy()
    result['target'] = result['index'].map(pd.Series(probability, index=test['index']))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    reread = pd.read_csv(destination)
    assert list(reread) == ['index', 'target'] and len(reread) == len(sample)
    np.testing.assert_array_equal(reread['index'], sample['index'])
    np.testing.assert_allclose(reread.target, result.target, atol=1e-12, rtol=0)
    return reread
