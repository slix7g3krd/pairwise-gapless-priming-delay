import pytest
import media.gapless_priming_delay as mod

CASES = [
    ("encoder_delay_trim", (1,)),
    ("prefeed_decode", (1,)),
    ("cancel_old_decode", (1,)),
    ("reset_resampler", (1,)),
    ("device_clock_resume", (1,)),
    ("seam_sample_check", (1,)),
    ("load_track", (1,)),
    ("decode", (1,)),
    ("enqueue", (1,)),
    ("play", (1,)),
    ("switch_track", (1,)),
    ("validate_gap", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
