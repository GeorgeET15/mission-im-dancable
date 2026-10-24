"""Deterministic tests for the MediaPipe adapter mapping and benchmark math.

No mediapipe package, model, GPU, or camera required — fake landmarks and
fake inference functions drive everything. Videos are tiny synthetic
clips written by OpenCV into tmp_path.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mission_im_dancable import benchmark, mediapipe_pose as mpp


def make_video(tmp_path, n_frames=10, size=(64, 64), name="clip.mp4"):
    path = tmp_path / name
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, size)
    assert w.isOpened()
    for i in range(n_frames):
        w.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
    w.release()
    return path


def fake_mp_landmarks(x=0.5, y=0.5, vis=0.9, pres=0.9):
    lm = SimpleNamespace(x=x, y=y, z=0.0, visibility=vis, presence=pres)
    return [SimpleNamespace(**vars(lm)) for _ in range(33)]


def test_mapping_preserves_left_right_and_converts_to_pixels():
    lms = fake_mp_landmarks()
    lms[11].x, lms[11].y = 0.25, 0.4   # LEFT_SHOULDER → image-left quarter
    lms[12].x, lms[12].y = 0.75, 0.4   # RIGHT_SHOULDER → image-right
    out = mpp.landmarks_to_coco(lms, width=200, height=100)
    assert out[5, 0] == pytest.approx(50.0)    # coco left_shoulder x
    assert out[6, 0] == pytest.approx(150.0)
    assert out[5, 2] == pytest.approx(0.9)
    assert out[0, 0] == pytest.approx(100.0)   # nose centered


def test_invalid_and_out_of_frame_landmarks_get_zero_confidence():
    lms = fake_mp_landmarks()
    lms[23].x = float("nan")
    lms[24].x, lms[24].y = -0.5, -0.1
    lms[25].x, lms[25].y = 1.02, 0.5   # extrapolated past the right edge
    lms[26].x, lms[26].y = 0.5, 1.3    # below the bottom edge
    out = mpp.landmarks_to_coco(lms, 200, 100)
    assert out[11, 2] == 0.0   # left_hip NaN → dropped
    assert out[12, 2] == 0.0   # right_hip off-frame → dropped
    assert out[13, 2] == 0.0   # left_knee x>1 normalized → dropped
    assert out[14, 2] == 0.0   # right_knee y>1 normalized → dropped
    assert out[5, 2] > 0.0     # left_shoulder intact


def test_world_landmarks_preserved_for_diagnostics():
    lms = [SimpleNamespace(x=0.1, y=-0.2, z=0.3, visibility=0.8) for _ in range(33)]
    world = mpp.world_landmarks_to_array(lms)
    assert world.shape == (33, 4)
    assert world[0, :3] == pytest.approx([0.1, -0.2, 0.3])
    assert mpp.world_landmarks_to_array(None) is None


def test_missing_model_and_package_errors_are_explicit(tmp_path):
    with pytest.raises(mpp.MediaPipeUnavailableError, match="not found"):
        mpp.MediaPipePoseTracker(tmp_path / "nope.task")
    with pytest.raises(ValueError, match="unknown variant"):
        mpp.model_path_for("medium")
    with pytest.raises(ValueError, match="unknown delegate"):
        mpp.MediaPipePoseTracker(tmp_path / "nope.task", delegate="tpu")


def test_missing_mediapipe_package_is_graceful(tmp_path, monkeypatch):
    """Importing the tracker without mediapipe raises a clear error."""
    import sys

    fake_model = tmp_path / "m.task"
    fake_model.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", None)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", None)
    with pytest.raises(mpp.MediaPipeUnavailableError, match="not installed"):
        mpp.MediaPipePoseTracker(fake_model)


def test_close_is_idempotent_and_releases_landmarker():
    class FakeLandmarker:
        closed = 0

        def close(self):
            self.closed += 1

    tracker = mpp.MediaPipePoseTracker.__new__(mpp.MediaPipePoseTracker)
    tracker._landmarker = FakeLandmarker()
    tracker._closed = False
    tracker.close()
    tracker.close()
    assert tracker._landmarker.closed == 1
    tracker._frame_index = 0
    with pytest.raises(RuntimeError, match="closed"):
        tracker.track(np.zeros((4, 4, 3), np.uint8))


def test_video_timestamps_strictly_increase(monkeypatch):
    """Worker rejects the API contract itself if timestamps repeat."""
    tracker = mpp.MediaPipePoseTracker.__new__(mpp.MediaPipePoseTracker)
    tracker._frame_index = 0
    tracker._closed = False
    ts = [tracker._timestamp_ms(30.0) for _ in range(4)]
    assert ts == sorted(ts) and len(set(ts)) == 4


def test_percentile_and_latency_stats():
    ms = [float(v) for v in range(1, 101)]
    stats = benchmark.latency_stats(ms)
    assert stats["median_ms"] == 50.5
    assert stats["p95_ms"] == pytest.approx(95.05)
    assert stats["fps"] == pytest.approx(1000 / statistics.fmean(ms), rel=1e-3)
    assert benchmark.latency_stats([])["count"] == 0


def test_missing_runs_counts_longest_gap():
    seq = [True, False, False, True, False, True, False]
    assert benchmark.missing_runs(seq) == {"missing_frames": 4,
                                           "longest_missing_streak": 2}


def test_jitter_ignores_unreliable_and_missing_frames():
    ok = np.tile(np.array([10, 20, 0.9], dtype=np.float32), (17, 1))
    # hips at 11,12 and shoulders 5,6 needed for normalization
    ok[5], ok[6], ok[11], ok[12] = (40, 10, .9), (60, 10, .9), (45, 50, .9), (55, 50, .9)
    moved = ok.copy(); moved[5, 0] += 5
    low = ok.copy(); low[:, 2] = 0.0
    jit = benchmark.temporal_jitter([ok, low, moved, ok], 0.3)
    assert jit["transition_count"] > 0
    assert jit["median_jitter"] >= 0


def test_benchmark_gracefully_reports_missing_backend(tmp_path):
    with pytest.raises(mpp.MediaPipeUnavailableError):
        benchmark.run_benchmark(tmp_path / "v.mp4", "mediapipe-heavy", 0.3,
                                model_path=tmp_path / "x.task")


def test_iter_frames_streams_and_respects_limit(tmp_path):
    video = make_video(tmp_path, n_frames=10)
    seen = [i for i, _ in benchmark.iter_frames(video)]
    assert seen == list(range(10))
    limited = [i for i, _ in benchmark.iter_frames(video, limit=4)]
    assert limited == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="cannot open"):
        list(benchmark.iter_frames(tmp_path / "nope.mp4"))


def test_run_benchmark_excludes_warmup_and_counts_misses(monkeypatch, tmp_path):
    """Fake YOLO tracker: warmup exclusion, miss handling, resource fields."""
    model = tmp_path / "yolo.pt"
    model.write_bytes(b"x")
    video = make_video(tmp_path, n_frames=10)
    calls = {"n": 0, "closed": 0}

    class FakeTracker:
        def __init__(self, path, device, conf):
            pass

        def predict(self, frame):
            calls["n"] += 1
            if calls["n"] % 3 == 0:  # every 3rd call misses
                return None
            return SimpleNamespace(
                keypoints=np.tile(np.array([10, 20, 0.9], np.float32), (17, 1)),
                inference_ms=5.0,
            )

        def close(self):
            calls["closed"] += 1

    monkeypatch.setattr(benchmark, "PoseTracker", FakeTracker)
    r = benchmark.run_benchmark(video, "yolo", 0.3, device="cpu",
                                model_path=model, warmup=4)
    assert calls["n"] == 10                 # 4 warmup + 6 measured
    assert r["frames_evaluated"] == 6
    assert r["warmup_frames"] == 4
    assert r["inference"]["count"] == 6     # misses still counted via e2e
    assert r["reliability"]["missing_frames"] == 2   # calls 6, 9 → miss
    assert r["resources"]["peak_rss_mb"] > 0
    assert r["resources"]["wall_s"] >= 0
    assert r["resources"]["gpu_peak_mb"] is None     # cpu run → no CUDA stat
    assert calls["closed"] == 1             # tracker released on exit


def test_run_benchmark_limit_and_bounded_visualization(monkeypatch, tmp_path):
    """Overlay frames must stream to the writer, not accumulate in memory."""
    video = make_video(tmp_path, n_frames=12)

    class FakeTracker:
        def __init__(self, path, device, conf):
            pass

        def predict(self, frame):
            return SimpleNamespace(
                keypoints=np.tile(np.array([10, 20, 0.9], np.float32), (17, 1)),
                inference_ms=1.0,
            )

        def close(self):
            pass

    monkeypatch.setattr(benchmark, "PoseTracker", FakeTracker)
    weights = tmp_path / "y.pt"
    weights.write_bytes(b"x")
    viz = tmp_path / "overlay.mp4"
    r = benchmark.run_benchmark(video, "yolo", 0.3, device="cpu",
                                model_path=weights, warmup=2,
                                limit=8, visualize_path=viz)
    assert r["frames_evaluated"] == 6        # limit 8 minus 2 warmup
    assert r["visualization"] == str(viz)
    cap = cv2.VideoCapture(str(viz))         # released writer → readable file
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()


def test_yolo_variant_resolves_model_file(tmp_path):
    assert benchmark.yolo_model_path("11m") == Path("models/yolo11m-pose.pt")
    assert benchmark.yolo_model_path("26l") == Path("models/yolo26l-pose.pt")
    # missing variant weights fail loudly, never substitute n
    with pytest.raises(FileNotFoundError, match="not found"):
        benchmark.run_benchmark(tmp_path / "v.mp4", "yolo-99z", 0.3,
                                device="cpu")


def test_yolo_variant_uses_resolved_path_and_records_spec(monkeypatch, tmp_path):
    video = make_video(tmp_path, n_frames=3)
    fake_weights = tmp_path / "models" / "yolo11m-pose.pt"
    fake_weights.parent.mkdir()
    fake_weights.write_bytes(b"weights")
    seen = {}

    class FakeTracker:
        def __init__(self, path, device, conf):
            seen["path"] = Path(path)

        def predict(self, frame):
            return None

        def close(self):
            pass

    monkeypatch.setattr(benchmark, "PoseTracker", FakeTracker)
    monkeypatch.setattr(benchmark, "YOLO_MODEL_DIR", fake_weights.parent)
    r = benchmark.run_benchmark(video, "yolo-11m", 0.3, device="cpu", warmup=0)
    assert seen["path"] == fake_weights
    assert r["backend"]["model_file"] == "yolo11m-pose.pt"
    assert r["backend"]["model_sha256"] == benchmark.sha256_of(fake_weights)
    # explicit --model overrides variant resolution
    other = tmp_path / "custom.pt"
    other.write_bytes(b"x")
    benchmark.run_benchmark(video, "yolo-11m", 0.3, device="cpu",
                            model_path=other, warmup=0)
    assert seen["path"] == other


def test_missing_frame_indices_recorded(monkeypatch, tmp_path):
    video = make_video(tmp_path, n_frames=6)

    class FakeTracker:
        def __init__(self, path, device, conf):
            pass

        def predict(self, frame):
            FakeTracker.n += 1
            return None if FakeTracker.n in {2, 5} else SimpleNamespace(
                keypoints=np.tile(np.array([1, 2, 0.9], np.float32), (17, 1)),
                inference_ms=1.0)

        def close(self):
            pass

    FakeTracker.n = 0
    monkeypatch.setattr(benchmark, "PoseTracker", FakeTracker)
    model = tmp_path / "y.pt"
    model.write_bytes(b"x")
    r = benchmark.run_benchmark(video, "yolo", 0.3, device="cpu",
                                model_path=model, warmup=0)
    assert r["reliability"]["missing_frame_indices"] == [1, 4]


def test_sha256_of_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc123")
    import hashlib
    assert benchmark.sha256_of(f) == hashlib.sha256(b"abc123").hexdigest()


def test_mediapipe_backend_parses_gpu_suffix(monkeypatch, tmp_path):
    video = make_video(tmp_path, n_frames=3)
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    seen = {}

    class FakeMP:
        def __init__(self, path, det, trk, delegate):
            seen["delegate"] = delegate

        def track(self, frame, fps):
            seen["fps"] = fps
            return None

        def close(self):
            pass

    monkeypatch.setattr(mpp, "MediaPipePoseTracker", FakeMP)
    r = benchmark.run_benchmark(video, "mediapipe-lite-gpu", 0.3,
                                model_path=model, warmup=0, fps=24.0)
    assert seen["delegate"] == "gpu"
    assert seen["fps"] == 24.0
    assert r["backend"]["variant"] == "lite"
    assert r["backend"]["delegate"] == "gpu"


def test_world_jitter_aggregates_displacements():
    a = np.tile(np.array([0.0, 0.0, 0.0, 0.9], np.float32), (33, 1))
    b = a.copy()
    b[:, 0] += 0.05                        # every joint moves 5 cm in x
    low_vis = a.copy()
    low_vis[:, 3] = 0.0
    stats = benchmark.world_jitter([a, low_vis, a, b], 0.3)
    assert stats["median_disp_m"] == pytest.approx(0.05, abs=1e-4)
    assert stats["transition_count"] == 33
    assert stats["mean_visibility"] > 0
    assert benchmark.world_jitter([], 0.3)["transition_count"] == 0


def _fake_result(name, mean=10.0):
    return {"backend": {"backend": name},
            "init_seconds": 1.0, "warmup_frames": 2,
            "frames_evaluated": 3,
            "inference": benchmark.latency_stats([mean, mean + 1]),
            "end_to_end": benchmark.latency_stats([12.0]),
            "reliability": {"detection_rate": 1.0,
                            "missing_frames": 0,
                            "longest_missing_streak": 0},
            "jitter": {"median_jitter": 0.01},
            "world_landmark_frames": 0,
            "_world_seq": np.zeros((2, 33, 4), np.float32)}


def test_save_results_emits_json_and_summary(tmp_path):
    import json

    out = benchmark.save_results([_fake_result("yolo")],
                                 {"path": "v.mp4", "fps": 30.0}, tmp_path)
    payload = json.loads(out.read_text())
    assert payload["results"][0]["backend"]["backend"] == "yolo"
    assert "_world_seq" not in payload["results"][0]   # stripped
    assert "yolo" in (tmp_path / "summary.txt").read_text()


def test_save_results_merges_one_backend_at_a_time(tmp_path):
    import json

    meta = {"path": "v.mp4", "fps": 30.0}
    benchmark.save_results([_fake_result("yolo")], meta, tmp_path)
    benchmark.save_results([_fake_result("mediapipe-heavy")], meta, tmp_path)
    names = {r["backend"]["backend"]
             for r in json.loads((tmp_path / "pose_benchmark.json")
                                 .read_text())["results"]}
    assert names == {"yolo", "mediapipe-heavy"}
    # re-running one backend replaces its entry, not the other
    benchmark.save_results([_fake_result("yolo", mean=99.0)], meta, tmp_path)
    results = {r["backend"]["backend"]: r
               for r in json.loads((tmp_path / "pose_benchmark.json")
                                   .read_text())["results"]}
    assert results["yolo"]["inference"]["mean_ms"] == 99.5
    assert "mediapipe-heavy" in results


def test_summarize_handles_backend_errors():
    text = benchmark.summarize([
        {"backend": {"backend": "mediapipe-heavy"}, "error": "boom"}])
    assert "ERROR" in text and "mediapipe-heavy" in text


import statistics  # noqa: E402  (used above)


def test_cli_accepts_benchmark_flags():
    from mission_im_dancable.config import parse_args
    cfg = parse_args(["--benchmark-pose", "--source", "dance_1.mp4",
                      "--pose-backends", "yolo", "--visualize-pose"])
    assert cfg.benchmark_pose and cfg.visualize_pose
    assert cfg.pose_backends == "yolo"
    assert cfg.max_frames == 30  # smoke-test default
    cfg2 = parse_args(["--benchmark-pose", "--source", "dance_1.mp4",
                       "--max-frames", "0"])
    assert cfg2.max_frames == 0
    with pytest.raises(SystemExit):
        parse_args(["--benchmark-pose"])  # needs --source
    with pytest.raises(SystemExit):
        parse_args(["--visualize-pose", "--source", "x.mp4"])  # needs benchmark
    with pytest.raises(SystemExit):
        parse_args(["--benchmark-pose", "--source", "x.mp4",
                    "--max-frames", "-1"])
