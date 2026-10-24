from pathlib import Path

import pytest

from mission_im_dancable.config import AppConfig, parse_args, resolve_device


def test_parse_args_defaults():
    cfg = parse_args([])
    assert cfg == AppConfig(
        camera_index=0,
        frame_width=1280,
        frame_height=720,
        model_path=Path("models/yolo11n-pose.pt"),
        device="auto",
        confidence=0.3,
    )


def test_parse_args_custom_values():
    cfg = parse_args(
        ["--camera", "1", "--width", "640", "--height", "480", "--device", "cpu", "--conf", "0.5"]
    )
    assert cfg.camera_index == 1
    assert cfg.frame_width == 640
    assert cfg.device == "cpu"
    assert cfg.confidence == 0.5


def test_parse_args_rejects_bad_confidence():
    with pytest.raises(SystemExit):
        parse_args(["--conf", "1.5"])


def test_parse_args_rejects_nonpositive_dimensions():
    with pytest.raises(SystemExit):
        parse_args(["--width", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--height", "-10"])


def test_resolve_device_auto_prefers_cuda():
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("auto", cuda_available=False) == "cpu"


def test_resolve_device_explicit():
    assert resolve_device("cuda", cuda_available=False) == "cuda"
    assert resolve_device("cpu", cuda_available=True) == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu", cuda_available=True)
