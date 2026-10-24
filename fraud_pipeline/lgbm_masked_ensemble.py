"""Replay the LightGBM masked ensemble and verify its frozen release artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .data import load_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/lgbm_masked_ensemble'
DATA = ROOT / 'bt-4012-competition-2026'
SEEDS = [2026, 17, 73]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(name):
    return json.loads((OUT / name).read_text())


def eligible(result):
    return bool(result['mean_gain'] >= .0005 and result['wins'] >= 4
                and result['worst_gain'] >= -.001
                and result['disjoint_mean_gain'] >= 0
                and result['mean_period_gain'] >= 0)


def check_integrity():
    integrity = read_json('cleanup_integrity.json')
    for relative, expected in integrity['original_v16_files'].items():
        if digest(OUT / relative) != expected:
            raise ValueError(f'Original v16 artifact changed: {relative}')
    for name, expected in read_json('manifest.json')['data_sha256'].items():
        if digest(DATA / name) != expected:
            raise ValueError(f'Input data changed: {name}')
    if digest(OUT / 'dependencies/v13_anchor.csv') != integrity['anchor_sha256']:
        raise ValueError('The fixed v13 anchor changed')
    if integrity['anchor_sha256'] != read_json('manifest.json')['reference_sha256']:
        raise ValueError('The anchor does not match the original v16 manifest')
    if digest(OUT / 'original_sources.zip') != integrity['original_sources_sha256']:
        raise ValueError('Original source archive changed')


def predict(bundle, frame, trees):
    x = frame[bundle['columns']].astype('float32').fillna(bundle['medians'])
    return bundle['model'].predict(x, num_iteration=trees, num_threads=2)


def ensemble(frame, cutoff, selection, records):
    parts = []
    for component in selection['recipe']:
        for seed in SEEDS:
            key = (component['family'], seed, cutoff)
            receipt = records[key]
            # Frozen receipts retain historical paths; models now live in this release.
            path = OUT / 'models' / Path(receipt['model_path']).name
            if digest(path) != receipt['model_sha256']:
                raise ValueError(f'Model changed: {path}')
            bundle = joblib.load(path)
            parts.append(predict(bundle, frame, component['trees']))
    return np.mean(parts, axis=0)


def reconstruct(test, sample, selection, records):
    anchor = pd.read_csv(OUT / 'dependencies/v13_anchor.csv')
    np.testing.assert_array_equal(np.sort(anchor['index']), np.sort(test['index']))
    base = anchor.set_index('index').target.loc[test['index']].to_numpy()
    probability = ((1 - selection['weight']) * base
                   + selection['weight'] * ensemble(test, 35, selection, records))
    return sample[['index']].assign(target=sample['index'].map(pd.Series(probability, index=test['index'])))


def verify_models(train, test, features, records):
    for (name, seed, cutoff), receipt in records.items():
        # Resolve the filename within the renamed release directory.
        path = OUT / 'models' / Path(receipt['model_path']).name
        if digest(path) != receipt['model_sha256']:
            raise ValueError(f'Model changed: {path}')
        bundle = joblib.load(path)
        config = next(f for f in read_json('manifest.json')['families'] if f['name'] == name)
        assert bundle['config'] == config and bundle['seed'] == seed
        assert bundle['columns'] == features + ['time_step']
        assert bundle['model'].current_iteration() == config['trees']
        fit = train[train.time_step.le(cutoff)]
        pd.testing.assert_series_equal(bundle['medians'], fit[bundle['columns']].astype('float32').median().fillna(0))
        assert receipt['fitting_rows'] == len(fit) and receipt['fit_max_time'] == cutoff
        saved = pd.read_csv(OUT / f'cache/{name}__seed{seed}__{cutoff}.csv')
        score = test if cutoff == 35 else train.set_index('txId', drop=False).loc[saved.txId].reset_index(drop=True)
        keys = ['index'] if cutoff == 35 else ['txId', 'time_step', 'label']
        np.testing.assert_array_equal(saved[keys], score[keys])
        for trees in config['checkpoints']:
            np.testing.assert_allclose(saved[f'trees_{trees}'], predict(bundle, score, trees), atol=1e-12, rtol=0)


def verify_history(train, selection):
    historical = read_json('independent_verification.json')['historical_forecasts']
    for record in historical:
        cutoff = record['cutoff']
        ref = pd.read_csv(OUT / f'reference__{cutoff}.csv')
        raw = train.set_index('txId').loc[ref.txId]
        np.testing.assert_array_equal(ref[['time_step', 'label']], raw[['time_step', 'label']])
        assert ref.time_step.gt(cutoff).all()
        parts = []
        for component in selection['recipe']:
            for seed in SEEDS:
                saved = pd.read_csv(OUT / f"cache/{component['family']}__seed{seed}__{cutoff}.csv")
                np.testing.assert_array_equal(ref[['txId', 'time_step', 'label']], saved[['txId', 'time_step', 'label']])
                parts.append(saved[f"trees_{component['trees']}"])
        probability = (1 - selection['weight']) * ref.probability + selection['weight'] * np.mean(parts, axis=0)
        assert abs(roc_auc_score(ref.label, probability) - record['candidate_auc']) < 1e-12
        saved = pd.read_csv(OUT / f'historical_forecasts/predictions__{cutoff}.csv')
        np.testing.assert_array_equal(saved.txId, ref.txId)
        np.testing.assert_allclose(saved.probability, probability, atol=1e-12, rtol=0)
    return len(historical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['predict', 'verify'])
    parser.add_argument('--output', type=Path, help='Prediction CSV destination; defaults to artifacts/lgbm_masked_ensemble/submission_replayed.csv')
    args = parser.parse_args()
    check_integrity()
    train, test, _, sample, features = load_data(DATA)
    train = train[train.label.notna()].reset_index(drop=True)
    selection = read_json('selection.json')
    assert eligible(selection)
    records = {(r['family'], r['seed'], r['cutoff']): r for r in read_json('build_receipt.json')}
    expected_tasks = {(r['family'], s, c) for r in selection['recipe'] for s in SEEDS for c in [18, 19, 35]}
    assert set(records) == expected_tasks
    with threadpool_limits(limits=2):
        replay = reconstruct(test, sample, selection, records)
        original = pd.read_csv(OUT / 'submission.csv')
        assert list(original) == ['index', 'target'] and len(original) == len(sample)
        np.testing.assert_array_equal(replay['index'], original['index'])
        assert replay['index'].is_unique and np.isfinite(replay.target).all() and replay.target.between(0, 1).all()
        np.testing.assert_allclose(replay.target, original.target, atol=1e-12, rtol=0)
        if args.stage == 'verify':
            verify_models(train, test, features, records)
            windows = verify_history(train, selection)
            print(json.dumps(dict(status='passed', models_replayed=len(records), historical_windows=windows,
                rows=len(replay), submission_sha256=digest(OUT / 'submission.csv'),
                max_probability_error=float(np.abs(replay.target - original.target).max()))))
        else:
            output = args.output or OUT / 'submission_replayed.csv'
            if output.resolve() == (OUT / 'submission.csv').resolve():
                raise ValueError('Choose a destination other than the frozen original submission.csv')
            replay.to_csv(output, index=False)
            print(f'Replayed {len(replay)} predictions to {output}')


if __name__ == '__main__':
    main()
