from typing import TYPE_CHECKING

import av
import numpy as np
import pytest

import pixeltable as pxt
from pixeltable.functions.array import to_audio

from .utils import IN_CI, rerun, skip_test_if_not_installed

if TYPE_CHECKING:
    import datasets  # type: ignore[import-untyped]


class TestArray:
    @pytest.mark.parametrize(
        'format,downsample', [('wav', False), ('mp3', False), ('mp3', True), ('flac', False), ('mp4', False)]
    )
    def test_to_audio(self, format: str, downsample: bool, reset_db: None) -> None:
        # Load a sample mp3 file to an array
        audio_data, sample_rate = self._load_sample_audio('./docs/resources/10-minute tour of Pixeltable.mp3')
        assert audio_data.dtype == np.float32
        assert audio_data.ndim == 2
        assert audio_data.shape[0] == 1  # That sample is in mono

        # Use to_audio to encode it to an audio file
        t = pxt.create_table('test_mp3_to_array_and_back', {'audio_array': pxt.Array[pxt.Float]})  # type: ignore[misc]
        output_sample_rate = sample_rate // 2 if downsample else sample_rate
        t.add_computed_column(
            audio_file=to_audio(
                t.audio_array, input_sample_rate=sample_rate, format=format, output_sample_rate=output_sample_rate
            )
        )
        update_status = t.insert(audio_array=audio_data)
        assert update_status.num_rows == 1
        assert update_status.num_excs == 0

        row = t.head(1)[0]
        assert set(row.keys()) == {'audio_array', 'audio_file'}
        encoded_path = row['audio_file']
        if format == 'mp4':
            assert encoded_path.endswith('.m4a')
        else:
            assert encoded_path.endswith(f'.{format}')
        print(f'Encoded audio file: {row["audio_file"]}')

        # Read back, decode, and validate the encoded file
        with av.open(encoded_path) as container:
            audio_stream = container.streams.audio[0]
            duration_seconds = float(audio_stream.duration * audio_stream.time_base)
            # the "10 minute tour" is actually about 5 minutes long
            assert abs(duration_seconds - 300) < 10

    def _load_sample_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        with av.open(file_path) as container:
            assert len(container.streams.audio) == 1
            audio_stream = container.streams.audio[0]
            sample_rate = audio_stream.rate
            audio_frames = [frame.to_ndarray() for frame in container.decode(audio_stream)]

        audio_data = np.concatenate(audio_frames, axis=1)
        assert len(audio_data) > 0
        return audio_data, sample_rate

    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    @rerun(reruns=3, reruns_delay=15)  # Guard against connection errors downloading datasets
    def test_encode_dataset_audio(self, reset_db: None) -> None:
        """
        The point of this test case is to validate to_audio UDF on a real-world dataset.

        As an added bonus, it exercises the stereo codepath which the test above doesn't.
        """
        skip_test_if_not_installed('datasets')
        import datasets

        hf_dataset = datasets.load_dataset('Hani89/medical_asr_recording_dataset')
        t = pxt.create_table('hfds', source=hf_dataset)
        row = t.head(1)[0]
        assert set(row.keys()) == {'audio', 'sentence'}
        assert isinstance(row['audio'], dict)
        assert set(row['audio'].keys()) == {'array', 'path', 'sampling_rate'}
        assert isinstance(row['audio']['array'], np.ndarray)
        assert isinstance(row['audio']['sampling_rate'], int)

        update_status = t.add_computed_column(
            audio_file=to_audio(
                t.audio.array.astype(pxt.Array), input_sample_rate=t.audio.sampling_rate.astype(pxt.Int), format='flac'
            )
        )
        assert update_status.num_computed_values > 6000
        assert update_status.num_excs == 0
        for row in t.head(10):
            assert set(row.keys()) == {'audio', 'sentence', 'audio_file'}
            print(f'Encoded audio file: {row["audio_file"]}')
