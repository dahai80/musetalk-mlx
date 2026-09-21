import json

from musetalk_mlx.tools.realtime_infer import _RttProbe


def test_rtt_split_stable_vs_first_packet(tmp_path):
    # sr=16000, window=5s (80000 samples). Frames whose PTS*sr >= window are
    # "stable"; PTS*sr < window are "first_packet".
    sr = 16000
    win = 5 * sr
    probe = _RttProbe(sr, window_samples=win)
    chunk = 3200  # 200ms
    # Feed 40 chunks: pts grows 0.2s -> 8.0s. Boundary at pts=5.0s.
    for _ in range(40):
        probe.on_push(chunk)
        probe.on_publish(probe._total / sr)
    report = probe.write_report(str(tmp_path / "rtt.json"))
    n_first = report["first_packet_rtt"]["frames"]
    n_stable = report["stable_rtt"]["frames"]
    assert n_first + n_stable == 40
    # pts=0.2..4.8 (24 values) < 5.0 -> first_packet; 5.0..8.0 (16) -> stable
    assert n_first == 24
    assert n_stable == 16
    assert "scope" in report["stable_rtt"]
    assert "<=80ms" in report["stable_rtt"]["scope"]
    data = json.loads((tmp_path / "rtt.json").read_text())
    assert data["frames_total"] == 40


def test_rtt_empty_when_no_pushes(tmp_path):
    probe = _RttProbe(16000)
    probe.on_publish(1.0)  # no pushes -> dropped
    report = probe.write_report(str(tmp_path / "rtt.json"))
    assert report["frames_total"] == 0
    assert report["stable_rtt"]["frames"] == 0
    assert report["first_packet_rtt"]["frames"] == 0
