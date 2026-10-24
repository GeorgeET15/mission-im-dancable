import pytest

from mission_im_dancable.timing import FpsCounter


def test_first_tick_reports_zero():
    counter = FpsCounter()
    assert counter.tick(now=1000.0) == 0.0


def test_fps_from_fake_timestamps():
    counter = FpsCounter()
    for t in [0.0, 0.1, 0.2, 0.3]:
        fps = counter.tick(now=t)
    assert fps == pytest.approx(10.0)


def test_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        FpsCounter(window=0)
