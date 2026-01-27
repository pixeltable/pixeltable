import pathlib
import sysconfig
from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pixeltable as pxt

from ..utils import IN_CI, rerun, skip_test_if_not_installed

if TYPE_CHECKING:
    import datasets  # type: ignore[import-untyped]


@pytest.mark.skipif(
    sysconfig.get_platform() == 'linux-aarch64', reason='libsndfile.so is missing on Linux ARM instances in CI'
)
@rerun(reruns=3, reruns_delay=15)  # Guard against connection errors downloading datasets
class TestHfDatasets:
    NUM_SAMPLES = 100

    def test_import_hf_dataset(self, uses_db: None, tmp_path: pathlib.Path) -> None:
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
            {
                'dataset_name': 'rotten_tomatoes',
                'dataset': datasets.load_dataset('cornell-movie-review-data/rotten_tomatoes'),
            },
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

    def test_insert_hf_dataset(self, uses_db: None, tmp_path: pathlib.Path) -> None:
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
            {
                'dataset_name': 'rotten_tomatoes',
                'dataset': datasets.load_dataset('cornell-movie-review-data/rotten_tomatoes'),
            },
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

    def test_import_hf_dataset_invalid(self, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        with pytest.raises(pxt.Error) as exc_info:
            pxt.io.import_huggingface_dataset('test', {})
        assert 'Unsupported data source type' in str(exc_info.value)

    @pytest.mark.parametrize('streaming', [False, True])
    def test_import_images(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        # Test that datasets with images load properly
        split = f'test[:{self.NUM_SAMPLES}]' if not streaming else 'test'
        hf_dataset = datasets.load_dataset('ylecun/mnist', split=split, streaming=streaming)
        if streaming:
            hf_dataset = hf_dataset.take(self.NUM_SAMPLES)
        t = pxt.create_table('mnist', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'image', 'label'}
        assert md['columns']['image']['type_'] == 'Image'
        assert md['columns']['label']['type_'] == 'String'

        res = t.select(t.image.localpath).collect()
        assert all(pathlib.Path(row['image_localpath']).exists() for row in res)

    @pytest.mark.parametrize('streaming', [False, True])
    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_arrays(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        split = f'train[:{self.NUM_SAMPLES}]' if not streaming else 'train'
        hf_dataset = datasets.load_dataset('Hani89/medical_asr_recording_dataset', split=split, streaming=streaming)
        if streaming:
            hf_dataset = hf_dataset.take(self.NUM_SAMPLES)
        t = pxt.create_table('hfds', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'audio', 'sentence'}
        assert md['columns']['audio']['type_'] == 'Json'

        res = t.collect()
        assert all(isinstance(row['audio'], dict) for row in res)
        assert all(isinstance(row['audio']['array'], np.ndarray) for row in res)

    @pytest.mark.parametrize('streaming', [False, True])
    def test_import_audio_small(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        hf_dataset = datasets.load_dataset(
            'hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation', streaming=streaming
        )
        t = pxt.create_table('audio_test', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'file', 'audio', 'text', 'speaker_id', 'chapter_id', 'id'}
        assert md['columns']['audio']['type_'] == 'Audio'

        res = t.collect()
        assert all(pathlib.Path(row['audio']).exists() for row in res)

    # This dataset is too large not to use in streaming mode (124GB)
    # TODO: find dataset containing Audio that is not gigantic
    @pytest.mark.parametrize('streaming', [True])
    def test_import_audio(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        hf_dataset = datasets.load_dataset(
            'openslr/librispeech_asr', split='train.clean.100', streaming=streaming
        ).take(100)
        t = pxt.create_table('audio_test', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'file', 'audio', 'text', 'speaker_id', 'chapter_id', 'id'}
        assert md['columns']['audio']['type_'] == 'Audio'

        res = t.collect()
        assert all(pathlib.Path(row['audio']).exists() for row in res)

    # This test fails with streaming=True due to a known bug in datasets:
    # https://github.com/huggingface/datasets/issues/3738
    # TODO: find out whether we need a workaround
    # @pytest.mark.parametrize('streaming', [False, True])
    @pytest.mark.parametrize('streaming', [False])
    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_list_of_dict(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        num_samples = 1000  # we need more samples to get non-Null prev_messages
        split = f'train[:{num_samples}]' if not streaming else 'train'
        dataset = datasets.load_dataset('natolambert/GeneralThought-430K-filtered', split=split, streaming=streaming)
        if streaming:
            dataset = dataset.take(num_samples)
        t = pxt.create_table('natolambert', source=dataset, primary_key='question_id', if_exists='replace')
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {
            'question_id',
            'question_url',
            'question',
            'reference_answer',
            'prev_messages',
            'model_name',
            'model_answer',
            'model_reasoning',
            'task',
            'question_license',
            'question_source',
            'community_answer_score',
            'community_question_score',
            'verifier_score',
        }
        assert md['columns']['prev_messages']['type_'] == 'Json'

        res = t.where(t.prev_messages != None).collect()
        assert all(isinstance(row['prev_messages'], list) for row in res)
        assert all(isinstance(x, dict) for row in res for x in row['prev_messages'])

    @pytest.mark.parametrize('streaming', [False, True])
    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_classlabel(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        split = f'train[:{self.NUM_SAMPLES}]' if not streaming else 'train'
        hf_dataset = datasets.load_dataset(
            'cornell-movie-review-data/rotten_tomatoes', split=split, streaming=streaming
        )
        if streaming:
            hf_dataset = hf_dataset.take(self.NUM_SAMPLES)
        t = pxt.create_table('test', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'label', 'text'}
        assert md['columns']['label']['type_'] == 'String'

        res = t.collect()
        assert all(row['label'] in ['neg', 'pos'] for row in res)

    @pytest.mark.parametrize('streaming', [False, True])
    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_sequence_of_float(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        # Cohere Wikipedia has embeddings as Sequence(float32); 'mi': a relatively small dataset
        split = f'train[:{self.NUM_SAMPLES}]' if not streaming else 'train'
        hf_dataset = datasets.load_dataset(
            'Cohere/wikipedia-2023-11-embed-multilingual-v3', 'mi', split=split, streaming=streaming
        )
        if streaming:
            hf_dataset = hf_dataset.take(self.NUM_SAMPLES)
        t = pxt.create_table('test', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'c_id', 'emb', 'text', 'title', 'url'}
        assert md['columns']['emb']['type_'] == 'Array[(None,), float32]'

        res = t.collect()
        assert all(isinstance(row['emb'], np.ndarray) for row in res)
        assert all(row['emb'].shape == (1024,) for row in res)
        assert all(row['emb'].dtype == np.float32 for row in res)

    @pytest.mark.parametrize('streaming', [False, True])
    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_sequence_of_dict(self, streaming: bool, uses_db: None) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        # SQuAD has answers as {'text': List(string), 'answer_start': List(int32)}
        split = f'validation[:{self.NUM_SAMPLES}]' if not streaming else 'validation'
        hf_dataset = datasets.load_dataset('squad', split=split, streaming=streaming)
        if streaming:
            hf_dataset = hf_dataset.take(self.NUM_SAMPLES)
        t = pxt.create_table('squad_test', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'answers', 'context', 'id', 'question', 'title'}
        assert md['columns']['answers']['type_'] == 'Json'

        res = t.collect()
        # answers should be a dict containing lists; however, the list of ints gets turned into an ndarray
        # TODO: what kinds of flags should we provide to control whether an inlined numerical array turns into a list or
        # an ndarray?
        assert all(isinstance(row['answers'], dict) for row in res)
        assert all(isinstance(row['answers']['text'], list) for row in res)
        assert all(isinstance(row['answers']['answer_start'], np.ndarray) for row in res)

    @pytest.mark.parametrize('streaming', [False, True])
    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_nested_struct(self, streaming: bool, uses_db: None) -> None:
        """
        Test importing dataset with nested structures:
        - supporting_facts: Sequence({'title': string, 'sent_id': int32})
        - context: Sequence({'title': string, 'sentences': Sequence(string)})
        """
        skip_test_if_not_installed('datasets')
        import datasets

        # HotpotQA has complex nested structures
        split = f'train[:{self.NUM_SAMPLES}]' if not streaming else 'train'
        hf_dataset = datasets.load_dataset('hotpotqa/hotpot_qa', 'distractor', split=split, streaming=streaming)
        if streaming:
            hf_dataset = hf_dataset.take(self.NUM_SAMPLES)
        t = pxt.create_table('hotpotqa_test', source=hf_dataset)
        md = t.get_metadata()
        assert set(md['columns'].keys()) == {'id', 'question', 'answer', 'supporting_facts', 'context', 'level', 'type'}
        assert md['columns']['supporting_facts']['type_'] == 'Json'
        assert md['columns']['context']['type_'] == 'Json'

        res = t.collect()
        assert all(isinstance(row['supporting_facts'], dict) for row in res)
        assert all(isinstance(row['supporting_facts']['title'], list) for row in res)
        assert all(isinstance(row['supporting_facts']['sent_id'], np.ndarray) for row in res)
        assert all(isinstance(row['context'], dict) for row in res)
        assert all(isinstance(row['context']['title'], list) for row in res)
        assert all(isinstance(row['context']['sentences'], list) for row in res)

    @pytest.mark.parametrize('streaming', [False, True])
    @pytest.mark.skipif(IN_CI, reason='Too much IO for CI')
    def test_import_arraynd(self, streaming: bool, uses_db: None) -> None:
        """Test dataset with Array2D and Array3D features."""
        skip_test_if_not_installed('datasets')
        import datasets

        split = f'train[:{self.NUM_SAMPLES}]' if not streaming else 'train'
        hf_dataset = datasets.load_dataset('tanganke/nyuv2', split=split, streaming=streaming)
        if streaming:
            hf_dataset = hf_dataset.take(self.NUM_SAMPLES)
        t = pxt.create_table('nyuv2_test', source=hf_dataset)
        md = t.get_metadata()
        assert md['columns']['image']['type_'] == 'Array[(3, 288, 384), float32]'
        assert md['columns']['segmentation']['type_'] == 'Array[(288, 384), int64]'
        assert md['columns']['depth']['type_'] == 'Array[(1, 288, 384), float32]'
        assert md['columns']['normal']['type_'] == 'Array[(3, 288, 384), float32]'
        assert md['columns']['noise']['type_'] == 'Array[(1, 288, 384), float32]'

        res = t.collect()
        assert set(res.schema.keys()) == {'image', 'segmentation', 'depth', 'normal', 'noise'}
        assert all(isinstance(row['image'], np.ndarray) for row in res)
        assert all(row['image'].shape == (3, 288, 384) for row in res)
        assert all(row['image'].dtype == np.float32 for row in res)
        assert all(isinstance(row['segmentation'], np.ndarray) for row in res)
        assert all(row['segmentation'].shape == (288, 384) for row in res)
        assert all(row['segmentation'].dtype == np.int64 for row in res)
        assert all(isinstance(row['depth'], np.ndarray) for row in res)
        assert all(row['depth'].shape == (1, 288, 384) for row in res)
        assert all(row['depth'].dtype == np.float32 for row in res)
        assert all(isinstance(row['normal'], np.ndarray) for row in res)
        assert all(row['normal'].shape == (3, 288, 384) for row in res)
        assert all(row['normal'].dtype == np.float32 for row in res)
        assert all(isinstance(row['noise'], np.ndarray) for row in res)
        assert all(row['noise'].shape == (1, 288, 384) for row in res)
        assert all(row['noise'].dtype == np.float32 for row in res)
