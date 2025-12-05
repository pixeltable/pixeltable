import pathlib
import sysconfig
from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import IN_CI, rerun, skip_test_if_not_installed

if TYPE_CHECKING:
    import datasets  # type: ignore[import-untyped]


@pytest.mark.skipif(
    sysconfig.get_platform() == 'linux-aarch64', reason='libsndfile.so is missing on Linux ARM instances in CI'
)
# @rerun(reruns=3, reruns_delay=15)  # Guard against connection errors downloading datasets
class TestHfDatasets:
    def test_import_hf_dataset(self, reset_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        test_cases = [
            # { # includes a timestamp. 20MB for specific slice
            # Disbled this test case because download is failing, and its not critical.
            #     'dataset_name': 'c4',
            #     # see https://huggingface.co/datasets/allenai/c4/blob/main/realnewslike/c4-train.00000-of-00512.json.gz
            #     'dataset': datasets.load_dataset(
            #         "allenai/c4",
            #         data_dir="realnewslike",
            #         data_files="c4-train.00000-of-00512.json.gz",
            #         split='train[:1000]',
            #         cache_dir=tmp_path
            #     ),
            # },
            {  # includes an embedding (array type), common in a few RAG datasets.
                'dataset_name': 'cohere_wikipedia',
                'dataset': datasets.load_dataset(
                    'Cohere/wikipedia-2023-11-embed-multilingual-v3', data_dir='cr'
                ).select_columns(['url', 'title', 'text', 'emb']),
                # column with name `_id`` is not currently allowed by pixeltable rules,
                # so filter out that column.
                # cr subdir has a small number of rows, avoid running out of space in CI runner
                # see https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/tree/main/cr
                'schema_override': {'emb': pxt.Array[(1024,), pxt.Float]},  # type: ignore[misc]
            },
            # example of dataset dictionary with multiple splits
            {'dataset_name': 'rotten_tomatoes', 'dataset': datasets.load_dataset('rotten_tomatoes')},
            # example of dataset with a sequence of dicts
            # (commented out for now, to keep the test overhead low, and because the test itself could use attention)
            # {
            #     'dataset_name': 'hotpot_qa',
            #     'dataset': datasets.load_dataset(
            #         'hotpot_qa',
            #         'fullwiki',
            #         split='train[:1000]',
            #         trust_remote_code=True
            #     ),
            # },
        ]

        # test a column name for splits other than the default of 'split'
        split_column_name = 'my_split_col'
        for rec in test_cases:
            dataset_name = rec['dataset_name']
            hf_dataset = rec['dataset']

            tab = pxt.io.import_huggingface_dataset(
                dataset_name,
                hf_dataset,
                schema_overrides=rec.get('schema_override', None),
                column_name_for_split=split_column_name,
            )
            if isinstance(hf_dataset, datasets.Dataset):
                self._assert_hf_dataset_equal(hf_dataset, tab.select(), split_column_name)
            elif isinstance(hf_dataset, datasets.DatasetDict):
                assert tab.count() == sum(hf_dataset.num_rows.values())
                assert split_column_name in tab._get_schema()

                for dataset_name in hf_dataset:
                    query = tab.where(tab.my_split_col == dataset_name)
                    self._assert_hf_dataset_equal(hf_dataset[dataset_name], query, split_column_name)
            else:
                raise AssertionError()

    def test_insert_hf_dataset(self, reset_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        test_cases = [
            # { # includes a timestamp. 20MB for specific slice
            # Disbled this test case because download is failing, and its not critical.
            #     'dataset_name': 'c4',
            #     # see https://huggingface.co/datasets/allenai/c4/blob/main/realnewslike/c4-train.00000-of-00512.json.gz
            #     'dataset': datasets.load_dataset(
            #         "allenai/c4",
            #         data_dir="realnewslike",
            #         data_files="c4-train.00000-of-00512.json.gz",
            #         split='train[:1000]',
            #         cache_dir=tmp_path
            #     ),
            # },
            {  # includes an embedding (array type), common in a few RAG datasets.
                'dataset_name': 'cohere_wikipedia',
                'dataset': datasets.load_dataset(
                    'Cohere/wikipedia-2023-11-embed-multilingual-v3', data_dir='cr'
                ).select_columns(['url', 'title', 'text', 'emb']),
                # column with name `_id`` is not currently allowed by pixeltable rules,
                # so filter out that column.
                # cr subdir has a small number of rows, avoid running out of space in CI runner
                # see https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3/tree/main/cr
                'schema_override': {'emb': pxt.Array[(1024,), pxt.Float]},  # type: ignore[misc]
            },
            # example of dataset dictionary with multiple splits
            {'dataset_name': 'rotten_tomatoes', 'dataset': datasets.load_dataset('rotten_tomatoes')},
        ]

        # test a column name for splits other than the default of 'split'
        split_column_name = 'my_split_col'
        for rec in test_cases:
            dataset_name = rec['dataset_name']
            hf_dataset = rec['dataset']

            tab = pxt.io.import_huggingface_dataset(
                dataset_name,
                hf_dataset,
                schema_overrides=rec.get('schema_override', None),
                column_name_for_split=split_column_name,
            )
            if isinstance(hf_dataset, datasets.Dataset):
                self._assert_hf_dataset_equal(hf_dataset, tab.select(), split_column_name)
            elif isinstance(hf_dataset, datasets.DatasetDict):
                assert tab.count() == sum(hf_dataset.num_rows.values())
                assert split_column_name in tab._get_schema()

                for dataset_name in hf_dataset:
                    query = tab.where(tab.my_split_col == dataset_name)
                    self._assert_hf_dataset_equal(hf_dataset[dataset_name], query, split_column_name)
            else:
                raise AssertionError()
            len1 = tab.count()
            tab.insert(
                hf_dataset, schema_overrides=rec.get('schema_override', None), column_name_for_split=split_column_name
            )
            assert tab.count() == len1 * 2

    @classmethod
    def _assert_hf_dataset_equal(cls, hf_dataset: 'datasets.Dataset', query: pxt.Query, split_column_name: str) -> None:
        import datasets

        assert query.count() == hf_dataset.num_rows
        assert set(query.schema.keys()) == (set(hf_dataset.features.keys()) | {split_column_name})

        # immutable so we can use it as in a set
        DatasetTuple = namedtuple('DatasetTuple', ' '.join(hf_dataset.features.keys()))  # type: ignore[misc]  # noqa: PYI024
        acc_dataset: set[DatasetTuple] = set()
        for tup in hf_dataset:
            immutable_tup = {}
            for k in tup:
                if isinstance(tup[k], list):
                    immutable_tup[k] = tuple(tup[k])
                else:
                    immutable_tup[k] = tup[k]

            acc_dataset.add(DatasetTuple(**immutable_tup))

        for tup in query.collect():
            assert tup[split_column_name] in hf_dataset.split._name

            encoded_tup = {}
            for column_name, value in tup.items():
                if column_name == split_column_name:
                    continue
                feature_type = hf_dataset.features[column_name]
                if isinstance(feature_type, datasets.ClassLabel):
                    assert value in feature_type.names
                    # must use the index of the class label as the value to
                    # compare with dataset iteration output.
                    enc_value = feature_type.encode_example(value)
                elif isinstance(feature_type, datasets.Sequence):
                    assert feature_type.feature.dtype == 'float32', 'may need to add more types'
                    enc_value = tuple(float(x) for x in value)
                else:
                    enc_value = value

                encoded_tup[column_name] = enc_value

            check_tup = DatasetTuple(**encoded_tup)
            assert check_tup in acc_dataset

    def test_import_hf_dataset_with_images(self, reset_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        # Test that datasets with images load properly
        t = pxt.io.import_huggingface_dataset('mnist', datasets.load_dataset('ylecun/mnist', split='test'))
        assert t.count() == 10000
        img = t.head(1)['image'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size == (28, 28)

    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_arrays(self, reset_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        hf_dataset = datasets.load_dataset('Hani89/medical_asr_recording_dataset')
        t = pxt.create_table('hfds', source=hf_dataset)
        res = t.head(1)
        row = res[0]
        assert set(row.keys()) == {'audio', 'sentence'}
        assert isinstance(row['audio'], dict)
        assert set(row['audio'].keys()) == {'array', 'path', 'sampling_rate'}
        assert isinstance(row['audio']['array'], np.ndarray)
        assert isinstance(row['audio']['path'], str)
        assert isinstance(row['audio']['sampling_rate'], int)
        assert isinstance(row['sentence'], str)

    def test_import_list_of_dict(self, reset_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        dataset = datasets.load_dataset('natolambert/GeneralThought-430K-filtered', split='train[:100]')
        t = pxt.create_table('natolambert', source=dataset, primary_key='question_id', if_exists='replace')

        res = t.head(1)
        row = res[0]
        assert set(row.keys()) == {'question_id', 'question', 'answer'}

    def test_import_hf_dataset_invalid(self, reset_db: None) -> None:
        skip_test_if_not_installed('datasets')
        with pytest.raises(pxt.Error) as exc_info:
            pxt.io.import_huggingface_dataset('test', {})
        assert 'Unsupported data source type' in str(exc_info.value)

    def test_import_classlabel(self, reset_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        hf_dataset = datasets.load_dataset('rotten_tomatoes')
        t = pxt.create_table('test', source=hf_dataset)

        row = t.head(1)[0]
        assert set(row.keys()) == {'label', 'text'}
        assert row['label'] in ['neg', 'pos']

    def test_fast_hf_importer_with_images(self, reset_db: None) -> None:
        """Test FastHFImporter with image data."""
        skip_test_if_not_installed('datasets')
        import datasets

        from pixeltable.io.table_data_conduit import HFTableDataConduit, TableDataConduit

        # Test with mnist dataset
        hf_dataset = datasets.load_dataset('ylecun/mnist', split='test[:10]')

        tds = TableDataConduit(source=hf_dataset)
        fast_importer = HFTableDataConduit.from_tds(tds)

        schema = fast_importer.infer_schema()
        assert 'image' in schema

        batches = list(fast_importer.valid_row_batch())
        assert len(batches) > 0

        first_row = batches[0][0]
        assert 'image' in first_row
        assert isinstance(first_row['image'], PIL.Image.Image)

    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_fast_hf_importer_with_audio(self, reset_db: None) -> None:
        """Test FastHFImporter with audio data (nested Sequence in dict)."""
        skip_test_if_not_installed('datasets')
        import datasets

        from pixeltable.io.table_data_conduit import HFTableDataConduit, TableDataConduit

        hf_dataset = datasets.load_dataset('Hani89/medical_asr_recording_dataset')

        tds = TableDataConduit(source=hf_dataset)
        fast_importer = HFTableDataConduit.from_tds(tds)

        schema = fast_importer.infer_schema()
        assert 'audio' in schema
        assert 'sentence' in schema

        batches = list(fast_importer.valid_row_batch())
        assert len(batches) > 0

        first_row = batches[0][0]
        assert isinstance(first_row['audio'], dict)
        assert 'array' in first_row['audio']
        # Key assertion: array should be numpy array, not Python list
        assert isinstance(first_row['audio']['array'], np.ndarray)

    def test_import_sequence_float(self, reset_db: None) -> None:
        """Test importing dataset with Sequence of floats (embeddings).
        Uses Cohere/wikipedia-2023-11-embed-multilingual-v3 which has 1024-dim embeddings.
        """
        skip_test_if_not_installed('datasets')
        import datasets

        # Cohere Wikipedia has embeddings as Sequence(float32)
        hf_dataset = datasets.load_dataset(
            'Cohere/wikipedia-2023-11-embed-multilingual-v3', data_dir='cr', split='train[:10]'
        ).select_columns(['title', 'emb'])
        t = pxt.create_table('emb_test', source=hf_dataset, schema_overrides={'emb': pxt.Array[(1024,), pxt.Float]})

        schema = t._get_schema()
        assert schema['emb'].is_array_type()

        row = t.head(1)[0]
        assert isinstance(row['emb'], np.ndarray)
        assert row['emb'].shape == (1024,)
        assert row['emb'].dtype == np.float32

    def test_import_squad_answers(self, reset_db: None) -> None:
        """Test importing SQuAD dataset which has Sequence of structs (answers field).
        The answers field has structure: Sequence({'text': string, 'answer_start': int32})
        """
        skip_test_if_not_installed('datasets')
        import datasets

        # SQuAD has answers as Sequence({'text': Value, 'answer_start': Value})
        hf_dataset = datasets.load_dataset('squad', split='validation[:10]')
        t = pxt.create_table('squad_test', source=hf_dataset)

        schema = t._get_schema()
        # answers is Sequence of struct -> Json
        assert schema['answers'].is_json_type()
        assert schema['question'].is_string_type()
        assert schema['context'].is_string_type()

        row = t.head(1)[0]
        # answers should be a list of dicts with 'text' and 'answer_start' keys
        assert isinstance(row['answers'], list)
        if len(row['answers']) > 0:
            assert 'text' in row['answers'][0]
            assert 'answer_start' in row['answers'][0]
            assert isinstance(row['answers'][0]['text'], str)
            assert isinstance(row['answers'][0]['answer_start'], int)

    def test_import_hotpotqa_nested_struct(self, reset_db: None) -> None:
        """Test importing HotpotQA dataset which has complex nested structures.
        - supporting_facts: Sequence({'title': string, 'sent_id': int32})
        - context: Sequence({'title': string, 'sentences': Sequence(string)})
        """
        skip_test_if_not_installed('datasets')
        import datasets

        # HotpotQA has complex nested structures
        hf_dataset = datasets.load_dataset(
            'hotpotqa/hotpot_qa', 'distractor', split='validation[:10]', trust_remote_code=True
        )
        t = pxt.create_table('hotpotqa_test', source=hf_dataset)

        schema = t._get_schema()
        # supporting_facts is Sequence of struct -> Json
        assert schema['supporting_facts'].is_json_type()
        # context is Sequence of struct -> Json
        assert schema['context'].is_json_type()

        row = t.head(1)[0]
        # supporting_facts should be list of dicts with 'title' and 'sent_id'
        assert isinstance(row['supporting_facts'], list)
        if len(row['supporting_facts']) > 0:
            assert 'title' in row['supporting_facts'][0]
            assert 'sent_id' in row['supporting_facts'][0]

        # context should be a list of dicts with 'title' and 'sentences'
        assert isinstance(row['context'], list)
        if len(row['context']) > 0:
            assert 'title' in row['context'][0]
            assert 'sentences' in row['context'][0]
            # sentences should be a list of strings
            assert isinstance(row['context'][0]['sentences'], list)

    def test_import_array2d_array3d(self, reset_db: None) -> None:
        """Test importing dataset with Array2D and Array3D features.
        Uses tanganke/nyuv2 depth perception dataset which has:
        - image: Array3D(shape=(3, 288, 384), dtype='float32')
        - segmentation: Array2D(shape=(288, 384), dtype='int64')
        - depth: Array3D(shape=(1, 288, 384), dtype='float32')
        """
        skip_test_if_not_installed('datasets')
        import datasets

        hf_dataset = datasets.load_dataset('tanganke/nyuv2', split='train[:5]')
        t = pxt.create_table('nyuv2_test', source=hf_dataset)

        schema = t._get_schema()
        # Array2D and Array3D should map to ArrayType
        assert schema['image'].is_array_type()
        assert schema['segmentation'].is_array_type()
        assert schema['depth'].is_array_type()

        row = t.head(1)[0]
        # Values should be numpy arrays with correct shapes
        assert isinstance(row['image'], np.ndarray)
        assert row['image'].shape == (3, 288, 384)
        assert row['image'].dtype == np.float32

        assert isinstance(row['segmentation'], np.ndarray)
        assert row['segmentation'].shape == (288, 384)

        assert isinstance(row['depth'], np.ndarray)
        assert row['depth'].shape == (1, 288, 384)

    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_audio_decoded(self, reset_db: None) -> None:
        """Test importing dataset with decoded Audio feature.
        Uses Hani89/medical_asr_recording_dataset which has decoded audio with arrays.
        Note: Audio datasets with raw bytes (like PolyAI/minds14) are handled differently.
        """
        skip_test_if_not_installed('datasets')
        import datasets

        # This dataset has decoded audio (array, path, sampling_rate)
        hf_dataset = datasets.load_dataset('Hani89/medical_asr_recording_dataset', split='train[:5]')
        t = pxt.create_table('audio_test', source=hf_dataset)

        schema = t._get_schema()
        # Audio with decoded arrays maps to Json struct
        assert schema['audio'].is_json_type()
        assert schema['sentence'].is_string_type()

        row = t.head(1)[0]
        # Audio struct should have 'array', 'path', 'sampling_rate'
        assert isinstance(row['audio'], dict)
        assert 'array' in row['audio']
        assert 'path' in row['audio']
        assert 'sampling_rate' in row['audio']
        # array should be a numpy array
        assert isinstance(row['audio']['array'], np.ndarray)
