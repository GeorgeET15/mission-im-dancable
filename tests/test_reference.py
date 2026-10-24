import numpy as np
import pytest

from mission_im_dancable.reference import (
    CACHE_VERSION,
    ReferenceChoreography,
    cache_path_for,
    load_cache,
    sampled_frame_indices,
    save_cache,
)


def _choreo(video_path, n_samples=5):
    kpts = np.zeros((n_samples, 17, 3), dtype=np.float32)
    kpts[:, :, 2] = 0.9
    stat = video_path.stat()
    return ReferenceChoreography(
        video_path=video_path,
        video_fps=30.0,
        frame_count=300,
        width=640,
        height=480,
        sample_fps=10.0,
        timestamps=np.arange(n_samples) / 10.0,
        keypoints=kpts,
        model_name="yolo11l-pose.pt",
        video_size=stat.st_size,
        video_mtime_ns=stat.st_mtime_ns,
    )


def test_sampled_frame_indices():
    # 30 fps video sampled at 10 fps → every 3rd frame
    assert sampled_frame_indices(90, 30.0, 10.0) == list(range(0, 90, 3))
    # sample rate above video rate → every frame
    assert sampled_frame_indices(10, 30.0, 60.0) == list(range(10))
    assert sampled_frame_indices(0, 30.0, 10.0) == []


def test_choreography_properties(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    c = _choreo(video)
    assert c.duration == pytest.approx(10.0)
    assert c.num_samples == 5
    assert c.detected_count() == 5
    c.keypoints[2, :, 2] = 0.0  # one sample missed the person
    assert c.detected_count() == 4


def test_cache_roundtrip(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video-data")
    choreo = _choreo(video)
    cache_path = cache_path_for(video, tmp_path / "cache")
    save_cache(choreo, cache_path, "yolo11l-pose.pt", confidence=0.3)

    loaded = load_cache(cache_path, video, 10.0, "yolo11l-pose.pt", 0.3)
    assert loaded is not None
    assert loaded.video_fps == pytest.approx(30.0)
    assert np.array_equal(loaded.timestamps, choreo.timestamps)
    assert np.array_equal(loaded.keypoints, choreo.keypoints)


def test_cache_rejects_stale_inputs(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video-data")
    choreo = _choreo(video)
    cache_path = cache_path_for(video, tmp_path / "cache")
    save_cache(choreo, cache_path, "yolo11l-pose.pt", confidence=0.3)

    # different extraction settings or model → stale
    assert load_cache(cache_path, video, 15.0, "yolo11l-pose.pt", 0.3) is None
    assert load_cache(cache_path, video, 10.0, "other-model.pt", 0.3) is None
    assert load_cache(cache_path, video, 10.0, "yolo11l-pose.pt", 0.5) is None

    # video file modified → stale
    video.write_bytes(b"edited-video-data")
    assert load_cache(cache_path, video, 10.0, "yolo11l-pose.pt", 0.3) is None


def test_cache_missing_and_corrupt(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video-data")
    cache_path = tmp_path / "cache" / "nope.npz"
    assert load_cache(cache_path, video, 10.0, "m.pt", 0.3) is None

    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_bytes(b"not an npz")
    assert load_cache(cache_path, video, 10.0, "m.pt", 0.3) is None


def test_cache_version_bump_invalidates(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video-data")
    choreo = _choreo(video)
    cache_path = cache_path_for(video, tmp_path / "cache")
    save_cache(choreo, cache_path, "yolo11l-pose.pt", 0.3)

    # simulate an incompatible cache format from a previous version
    data = np.load(cache_path)
    import json

    meta = json.loads(str(data["meta"]))
    meta["cache_version"] = CACHE_VERSION + 1
    np.savez(cache_path, meta=json.dumps(meta), timestamps=data["timestamps"], keypoints=data["keypoints"])
    assert load_cache(cache_path, video, 10.0, "yolo11l-pose.pt", 0.3) is None
