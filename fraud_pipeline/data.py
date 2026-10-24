"""Load and audit files without inferring labels or anonymous feature meanings."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def load_data(directory):
    directory = Path(directory)
    train = pd.read_csv(directory / "train.csv", na_values=["none", "unknown"])
    test = pd.read_csv(directory / "test.csv")
    edges = pd.read_csv(directory / "txs_edgelist.csv")
    sample = pd.read_csv(directory / "sample_submission.csv")
    features = [c for c in train if c.startswith(("feat_", "feature_"))]
    if len(features) != 165 or not set(features).issubset(test.columns):
        raise ValueError("Expected 165 matching anonymous feature columns.")
    for name, frame in [("train", train), ("test", test)]:
        if frame.txId.isna().any() or frame.txId.duplicated().any():
            raise ValueError(f"{name}: missing or duplicate transaction IDs")
        if frame.time_step.isna().any():
            raise ValueError(f"{name}: missing time steps")
        if np.isinf(frame[features].to_numpy()).any():
            raise ValueError(f"{name}: infinite feature values")
    if set(train.txId) & set(test.txId):
        raise ValueError("Training and test transaction IDs overlap.")
    if not set(train.label.dropna().unique()).issubset({0, 1}):
        raise ValueError("Unexpected target encoding; expected 0, 1, or missing.")
    if train.time_step.max() >= test.time_step.min():
        raise ValueError("Expected strictly later test time steps.")
    if list(edges.columns) != ["txId1", "txId2"] or edges.isna().any().any():
        raise ValueError("Unexpected edge schema or missing endpoints.")
    if list(sample.columns) != ["index", "target"]:
        raise ValueError("Unexpected submission schema.")
    if test["index"].duplicated().any() or sample["index"].duplicated().any():
        raise ValueError("Duplicate submission indices.")
    if set(test["index"]) != set(sample["index"]):
        raise ValueError("Sample and test indices differ.")
    return train, test, edges, sample, features


def audit(directory, train, test, edges, features):
    nodes = pd.Index(pd.unique(edges.to_numpy().ravel()))
    known = pd.concat([train, test], ignore_index=True)
    times = known.set_index("txId").time_step
    source, destination = edges.txId1.map(times), edges.txId2.map(times)
    both = source.notna() & destination.notna()
    hashes = {}
    for file in sorted(Path(directory).glob("*.csv")):
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[file.name] = digest.hexdigest()
    return {
        "train_rows": len(train), "test_rows": len(test),
        "labeled_rows": int(train.label.notna().sum()),
        "unknown_rows": int(train.label.isna().sum()),
        "positive_rows": int(train.label.eq(1).sum()),
        "negative_rows": int(train.label.eq(0).sum()),
        "train_time_range": [int(train.time_step.min()), int(train.time_step.max())],
        "test_time_range": [int(test.time_step.min()), int(test.time_step.max())],
        "feature_count": len(features),
        "missing_feature_cells": int(known[features].isna().sum().sum()),
        "edge_rows": len(edges), "duplicate_edges": int(edges.duplicated().sum()),
        "self_edges": int(edges.txId1.eq(edges.txId2).sum()),
        "graph_nodes": len(nodes), "nodes_without_features": int((~nodes.isin(known.txId)).sum()),
        "known_endpoint_edges": int(both.sum()),
        "cross_time_known_edges": int((source[both] != destination[both]).sum()),
        "sha256": hashes,
    }
