"""Episode-safe chunking and train-only normalization for SO101 demonstrations."""

import numpy as np


def split_episodes(episode_ids, validation_count=5, seed=42):
    ids = np.unique(episode_ids)
    if not 0 < validation_count < len(ids):
        raise ValueError("validation_count must leave training episodes")
    val = np.random.default_rng(seed).permutation(ids)[:validation_count]
    return sorted(set(ids.tolist()) - set(val.tolist())), sorted(val.tolist())


def make_chunks(actions, episode_ids, horizon):
    actions = np.asarray(actions, dtype=np.float32)
    episode_ids = np.asarray(episode_ids)
    if horizon < 1 or len(actions) != len(episode_ids):
        raise ValueError("invalid chunk inputs")
    chunks = np.zeros((len(actions), horizon, actions.shape[1]), dtype=np.float32)
    valid = np.zeros((len(actions), horizon), dtype=bool)
    for i in range(len(actions)):
        for offset in range(horizon):
            j = i + offset
            if j >= len(actions) or episode_ids[j] != episode_ids[i]:
                break
            chunks[i, offset] = actions[j]
            valid[i, offset] = True
    return chunks, valid


def normalization(values, train_indices):
    subset = np.asarray(values, dtype=np.float32)[train_indices]
    if not len(subset) or not np.isfinite(subset).all():
        raise ValueError("normalization needs finite training values")
    return dict(
        mean=subset.mean(0).tolist(), std=np.maximum(subset.std(0), 1e-6).tolist()
    )


def normalize(values, stats):
    return (
        np.asarray(values, dtype=np.float32) - np.array(stats["mean"], dtype=np.float32)
    ) / np.array(stats["std"], dtype=np.float32)
