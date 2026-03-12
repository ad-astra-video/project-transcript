import asyncio
from pathlib import Path
import sys
from unittest.mock import AsyncMock, Mock

import numpy as np

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))


def test_compute_rms_dbfs_silence_is_negative_infinity_like():
    from pipeline.main import _compute_rms_dbfs

    samples = np.zeros((16000,), dtype=np.float32)
    rms, dbfs = _compute_rms_dbfs(samples)

    assert rms == 0.0
    assert dbfs < -200.0


def test_process_transcription_skips_low_audio_window():
    import pipeline.main
    from pipeline.main import TranscriberState, _process_transcription_async

    state = TranscriberState()
    state.buffer_rate = 16000
    state.low_audio_filter_enabled = True
    state.low_audio_min_dbfs = -45.0
    state.whisper_client = Mock()
    state.whisper_client.transcribe_audio = AsyncMock(return_value=(0, [], None, 0.0))
    state.summary_client = None

    original_state = pipeline.main.STATE
    pipeline.main.STATE = state

    try:
        async def run_test():
            silence = np.zeros((16000 * 2,), dtype=np.float32)
            await _process_transcription_async(silence, 0.0, 2.0)

        asyncio.run(run_test())

        state.whisper_client.transcribe_audio.assert_not_called()
    finally:
        pipeline.main.STATE = original_state


def test_process_diarization_skips_low_audio_window():
    import pipeline.main
    from pipeline.main import TranscriberState, _process_diarization_async

    state = TranscriberState()
    state.buffer_rate = 16000
    state.low_audio_filter_enabled = True
    state.low_audio_min_dbfs = -45.0
    state.diarization_client = Mock()
    state.diarization_client.process_audio = AsyncMock()

    original_state = pipeline.main.STATE
    pipeline.main.STATE = state

    try:
        async def run_test():
            silence = np.zeros((16000 * 10,), dtype=np.float32)
            await _process_diarization_async(silence, 0.0, 10.0)

        asyncio.run(run_test())

        state.diarization_client.process_audio.assert_not_called()
    finally:
        pipeline.main.STATE = original_state


def test_update_params_updates_and_clamps_low_audio_threshold():
    import pipeline.main
    from pipeline.main import TranscriberState, update_params

    state = TranscriberState()

    original_state = pipeline.main.STATE
    pipeline.main.STATE = state

    try:
        async def run_test():
            await update_params({"low_audio_filter_enabled": False, "low_audio_min_dbfs": 10.0})

        asyncio.run(run_test())

        assert state.low_audio_filter_enabled is False
        assert state.low_audio_min_dbfs == 0.0
    finally:
        pipeline.main.STATE = original_state
