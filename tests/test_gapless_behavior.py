"""六项无缝播放修复的行为测试。

平台自带的契约测试只检查每个入口能以 ``fn(1)`` 调用并返回非 None。这里从
题目描述的实际故障出发，验证可观察的数值、状态和样本级性质。
"""

import json
import math

import pytest

import media.gapless_priming_delay as gapless


ENTRY_NAMES = [
    "encoder_delay_trim",
    "prefeed_decode",
    "cancel_old_decode",
    "reset_resampler",
    "device_clock_resume",
    "seam_sample_check",
    "load_track",
    "decode",
    "enqueue",
    "play",
    "switch_track",
    "validate_gap",
    "metrics",
    "audit",
    "fallback",
]


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_seed_entry_returns_jsonable_evidence(name):
    result = getattr(gapless, name)(1)
    assert result is not None
    # 评审/审计链路需要能直接记录这些证据。
    json.dumps(result)


# ----------------------------------------------------------- 1. 实际编码延迟

def test_trim_uses_actual_encoder_delay_not_container_value():
    report = gapless.encoder_delay_trim(1)
    assert report["fixed"] is True
    assert report["actual_delay"] == 312          # Opus pre-skip 夹具值
    assert report["container_delay"] == 2048
    assert report["voice_head_saved_samples"] == 1736
    assert report["voice_head_saved_ms"] == pytest.approx(36.166666)
    assert report["onset_preserved"] is True
    assert report["old_behavior_ate_onset"] is True


def test_track_trim_preserves_first_audible_sample():
    track = gapless.make_track({"encoder": "opus", "id": "voice"})
    fixed = gapless.encoder_delay_trim(track)
    wrong = gapless.encoder_delay_trim(
        track, delay=track["container_delay"], source="container"
    )

    assert fixed["removed_start_samples"] == track["actual_delay"]
    assert fixed["first_sample"] == pytest.approx(track["audible_samples"][0])
    assert fixed["onset_preserved"] is True
    assert wrong["onset_preserved"] is False
    assert len(wrong["samples"]) < len(fixed["samples"])


def test_direct_sample_trim_call():
    assert gapless.trim_samples([9, 9, 1, 2, 3], 2) == [1.0, 2.0, 3.0]
    result = gapless.encoder_delay_trim([9, 9, 1, 2, 3], 2)
    assert result["samples"] == [9.0, 1.0, 2.0, 3.0][1:]


# ----------------------------------------------------------- 2. 预滚提前入队

def test_prefeed_report_decodes_before_boundary():
    report = gapless.prefeed_decode(1)
    assert report["started_before_boundary"] is True
    assert report["remaining_current_when_started"] > 0
    assert report["prefeed_samples"] > 0
    assert report["first_frame_queued_before_seam"] is True
    assert report["underruns"] == 0


def test_player_prefeed_queue_is_ready_while_current_has_samples():
    player = gapless.GaplessPlayer(1)
    player.load_track({"id": "A", "encoder": "aac_lc"})
    player.enqueue(player.decode())
    player.play_frames(512)

    remaining = player.queue.depth_samples
    report = player.prefeed_next({"id": "B", "encoder": "opus"})

    assert remaining > 0
    assert report["remaining_current_when_started"] == remaining
    assert player.prefeed_queue.depth_samples > 0
    assert report["first_frame_queued_before_seam"] is True


# ----------------------------------------------------------- 3. 切歌取消旧解码

def test_cancel_report_rejects_all_stale_writes():
    report = gapless.cancel_old_decode(1)
    assert report["old_cancelled"] is True
    assert report["cancel_event_set"] is True
    assert report["old_generation"] + 1 == report["new_generation"]
    assert report["race_write_rejected"] is True
    assert report["stale_writes_applied"] == 0
    assert report["stale_chunks_rejected"] >= 1
    assert report["new_generation_accepted"] is True


def test_generation_sink_and_handle_are_thread_token_compatible():
    sink = gapless.GenerationSink(generation=3)
    handle = gapless.DecodeHandle(3, "old")
    assert not handle.cancelled
    handle.cancel()
    assert handle.cancelled

    # 即使取消和 generation 切换之间发生竞态，旧代际写入也不能落盘。
    sink.activate_generation(4)
    assert sink.write(3, [0.1, 0.2]) is False
    assert sink.write(4, [0.3, 0.4]) is True
    stats = sink.stats()
    assert stats["stale_rejected_samples"] == 2
    assert stats["stale_samples_written"] == 0
    assert stats["accepted_samples"] == 2


# ----------------------------------------------------------- 4. 重置重采样器

def test_resampler_report_clears_phase_and_history_on_rate_change():
    report = gapless.reset_resampler(1)
    assert report["rate_changed"] is True
    assert report["reset"] is True
    assert report["phase_before_switch"] != 0.0
    assert report["history_before_switch"] > 0
    assert report["phase_after_switch"] == 0.0
    assert report["history_after_switch"] == 0
    assert report["phase_discontinuity_cleared"] is True
    assert report["after_output_finite"] is True


def test_resampler_object_resets_streaming_state():
    resampler = gapless.GaplessResampler(48_000, 44_100)
    resampler.process([0.1 * i for i in range(256)])
    assert resampler.history_samples > 0

    change = resampler.set_rates(input_rate=44_100, output_rate=48_000)
    assert change["rate_changed"] is True
    assert resampler.phase == 0.0
    assert resampler.history_samples == 0
    assert resampler.reset_count == 1

    out = resampler.process([0.0, 0.25, 0.5, 0.25, 0.0])
    assert out and all(math.isfinite(x) for x in out)


# ----------------------------------------------------------- 5. 设备时钟恢复

def test_resume_uses_device_clock_not_wall_clock():
    report = gapless.device_clock_resume(1)
    assert report["wall_clock_advance_samples"] == 96_000
    assert report["device_clock_advance_during_pause_samples"] == 0
    assert report["underrun_detected"] is True
    assert report["buggy_wallclock_jump_samples"] > 96_000
    assert report["used_device_clock"] is True
    assert report["resume_jump_samples"] == 0
    assert report["no_audible_sample_jump"] is True


def test_device_clock_object_tracks_actual_frames():
    clock = gapless.DeviceClock(48_000)
    clock.advance_device(12_000)
    clock.pause()
    clock.advance_wall(2.0)
    result = clock.resume(device_frames=11_520, underrun_frames=480)

    assert result["clock_source"] == "device_sample_counter"
    assert clock.media_position == 11_520
    assert result["resume_jump_samples"] == 0


# ----------------------------------------------------------- 6. 样本级接缝校验

def test_seam_check_accepts_exact_concatenation():
    result = gapless.seam_sample_check(
        [1, 2, 3], [4, 5], expected=[1, 2, 3, 4, 5]
    )
    assert result["ok"] is True
    assert result["sample_level"] is True
    assert result["mismatch_count"] == 0
    assert result["gap_samples"] == 0


def test_seam_check_reports_first_sample_mismatch():
    result = gapless.seam_sample_check(
        [1, 2, 3], [5, 6], expected=[1, 2, 3, 4, 5, 6]
    )
    assert result["ok"] is False
    assert result["mismatch_count"] > 0
    assert result["first_mismatch"] == 3


def test_seam_check_detects_pts_gap_and_overlap():
    gap = gapless.check_seam([1.0], [2.0], pts_left_end=100, pts_right_start=102)
    overlap = gapless.check_seam([1.0], [2.0], pts_left_end=102, pts_right_start=100)

    assert gap["ok"] is False and gap["gap_samples"] == 2
    assert overlap["ok"] is False and overlap["overlap_samples"] == 2


def test_seed_seam_report_catches_bad_container_trim():
    report = gapless.seam_sample_check(1)
    assert report["exact_seam_ok"] is True
    assert report["sample_level_exact"] is True
    assert report["gap_samples"] == 0
    assert report["validator_detects_container_trim_damage"] is True
    assert report["bad_seam_mismatch_count"] > 0


# ----------------------------------------------------------- 端到端与审计指标

def test_pipeline_gapless_seam_is_sample_exact():
    result = gapless.validate_gap(1)
    assert result["ok"] is True
    assert result["gapless"] is True
    assert result["gap_samples"] == 0
    assert result["mismatch_count"] == 0
    assert result["sample_level_exact"] is True
    assert result["empty_queue_window_samples"] == 0


def test_pipeline_switch_cancels_and_resets_resampler():
    report = gapless.switch_track(1)
    assert report["old_cancelled"] is True
    assert report["race_write_rejected"] is True
    assert report["stale_writes_applied"] == 0
    assert report["resampler_reset"] is True
    assert report["resampler_phase_after"] == 0.0
    assert report["queued_samples"] > 0


def test_metrics_all_six_requirements_met():
    report = gapless.metrics(1)
    assert all(report["checks"].values()), report["checks"]
    assert report["all_requirements_met"] is True
    assert report["gap_samples"] == 0
    assert report["stale_writes_applied"] == 0
    assert report["resume_jump_samples"] == 0


def test_audit_contains_every_required_event():
    report = gapless.audit(1)
    assert report["ok"] is True
    assert report["missing_event_kinds"] == []
    assert report["event_count"] >= 8
    kinds = {event["kind"] for event in report["events"]}
    assert {
        "prefeed_decode",
        "seam_sample_check",
        "cancel_old_decode",
        "reset_resampler",
        "device_clock_resume",
    } <= kinds


def test_fallback_cases_all_safe():
    report = gapless.fallback(1)
    assert report["all_safe"] is True
    assert report["case_count"] >= 5
    assert report["safe_count"] == report["case_count"]
    assert all(case["safe"] for case in report["cases"])
