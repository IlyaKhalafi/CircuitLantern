import logging

from circuit_lantern.services.execution import _CastNoticeFilter


def notice(dtype):
    return logging.makeLogRecord(
        {
            "msg": "MatMul8bitLt: inputs will be cast from %s to float16 during quantization",
            "args": (dtype,),
        }
    )


def test_expected_cast_notices_are_emitted_once_per_input_dtype():
    log_filter = _CastNoticeFilter()
    assert log_filter.filter(notice("torch.float32"))
    assert not log_filter.filter(notice("torch.float32"))
    assert log_filter.filter(notice("torch.bfloat16"))
    assert not log_filter.filter(notice("torch.bfloat16"))


def test_repeated_unrelated_backend_warnings_remain_visible():
    log_filter = _CastNoticeFilter()
    warning = logging.makeLogRecord({"msg": "A kernel returned non finite output", "args": ()})
    assert log_filter.filter(warning)
    assert log_filter.filter(warning)
