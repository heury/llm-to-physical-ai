import numpy as np
from scripts.runpod_so101.data_utils import (
    make_chunks,
    normalization,
    normalize,
    split_episodes,
)


def test_chunks_never_cross_episode_and_mask_end():
    a = np.arange(10, dtype=np.float32).reshape(5, 2)
    chunk, mask = make_chunks(a, [0, 0, 1, 1, 1], 3)
    np.testing.assert_array_equal(chunk[0, :2], a[:2])
    assert mask.tolist() == [
        [True, True, False],
        [True, False, False],
        [True, True, True],
        [True, True, False],
        [True, False, False],
    ]
    assert not chunk[1, 1:].any()


def test_split_is_episode_disjoint_and_complete():
    train, val = split_episodes(np.repeat(np.arange(8), 3), 2)
    assert len(val) == 2 and set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(8))


def test_normalization_ignores_validation_and_floors_constant_dimension():
    values = np.array([[1, 2], [3, 2], [10000, 7]], dtype=np.float32)
    stats = normalization(values, [0, 1])
    np.testing.assert_allclose(stats["mean"], [2, 2])
    assert stats["std"][1] > 0
    assert np.isfinite(normalize(values, stats)).all()
    np.testing.assert_allclose(normalize(values[:2], stats).mean(0), [0, 0], atol=1e-6)


def test_cleanup_guard_rejects_incomplete_training():
    import pytest
    from scripts.runpod_so101.watch import validate_completion

    settings = {"epochs": 100, "optimizer_updates_target": 5200}
    success = {
        "shared_volume_verified": True,
        "epochs": 100,
        "optimizer_updates": 5200,
        "model_sha256": "abc",
    }
    evaluation = dict(
        epochs=100,
        optimizer_updates=5200,
        strict_load=True,
        finite_weights=True,
        frozen_vlm_bitwise_unchanged=True,
        image_inference_finite=True,
        checkpoint_sha256="abc",
    )
    hashes = {"model/final_vla.pt": "abc"}
    validate_completion(success, settings, evaluation, hashes)
    for bad in [
        {"epochs": 99},
        {"optimizer_updates": 5199},
        {"strict_load": False},
        {"checkpoint_sha256": "wrong"},
    ]:
        with pytest.raises(ValueError):
            validate_completion(success, settings, {**evaluation, **bad}, hashes)


def test_download_receipt_cannot_authorize_deletion_of_another_run():
    import pytest
    from scripts.runpod_so101.watch import validate_download_receipt

    receipt = dict(pod_id="old", remote_root="/shared/old", episodes=10,
                   local_videos_verified=True, archive_sha256="abc", verified_files=93)
    validate_download_receipt(receipt, "old", "/shared/old")
    for pod, root in [("new", "/shared/old"), ("old", "/shared/new"), ("new", "/shared/new")]:
        with pytest.raises(ValueError):
            validate_download_receipt(receipt, pod, root)
    for flag, value in [("episodes", 9), ("local_videos_verified", False), ("archive_sha256", ""), ("verified_files", 0)]:
        with pytest.raises(ValueError):
            validate_download_receipt({**receipt, flag: value}, "old", "/shared/old")
