import pathlib
import sysconfig
from collections import namedtuple
from typing import TYPE_CHECKING

import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs

from ..utils import skip_test_if_not_installed

if TYPE_CHECKING:
    import datasets


@pytest.mark.skipif(
    sysconfig.get_platform() == 'linux-aarch64',
    reason='libsndfile.so is missing on Linux ARM instances in CI'
)
@pytest.mark.flaky(reruns=3, only_rerun='HfHubHTTPError')
class TestHfDatasets:
    def test_import_hf_dataset(self, reset_db, tmp_path: pathlib.Path) -> None:
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
                'schema_override': {'emb': pxt.Array[(1024,), pxt.Float]},
            },
            # example of dataset dictionary with multiple splits
            {
                'dataset_name': 'rotten_tomatoes',
                'dataset': datasets.load_dataset('rotten_tomatoes'),
            }
        ]

        # test a column name for splits other than the default of 'split'
        split_column_name = 'my_split_col'
        for rec in test_cases:
            dataset_name = rec['dataset_name']
            hf_dataset = rec['dataset']

            tab = pxt.io.import_huggingface_dataset(
                dataset_name,
                hf_dataset,
                column_name_for_split=split_column_name,
                schema_overrides=rec.get('schema_override', None),
            )
            if isinstance(hf_dataset, datasets.Dataset):
                self._assert_hf_dataset_equal(hf_dataset, tab.select(), split_column_name)
            elif isinstance(hf_dataset, datasets.DatasetDict):
                assert tab.count() == sum(hf_dataset.num_rows.values())
                assert split_column_name in tab._schema.keys()

                for dataset_name in hf_dataset:
                    df = tab.where(tab.my_split_col == dataset_name)
                    self._assert_hf_dataset_equal(hf_dataset[dataset_name], df, split_column_name)
            else:
                assert False

    @classmethod
    def _assert_hf_dataset_equal(
        cls, hf_dataset: 'datasets.Dataset', df: pxt.DataFrame, split_column_name: str
    ) -> None:
        import datasets

        assert df.count() == hf_dataset.num_rows
        assert set(df.schema.keys()) == (set(hf_dataset.features.keys()) | {split_column_name})

        # immutable so we can use it as in a set
        DatasetTuple = namedtuple('DatasetTuple', ' '.join(hf_dataset.features.keys()))
        acc_dataset: set[DatasetTuple] = set()
        for tup in hf_dataset:
            immutable_tup = {}
            for k in tup:
                if isinstance(tup[k], list):
                    immutable_tup[k] = tuple(tup[k])
                else:
                    immutable_tup[k] = tup[k]

            acc_dataset.add(DatasetTuple(**immutable_tup))

        for tup in df.collect():
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
                    value = feature_type.encode_example(value)
                elif isinstance(feature_type, datasets.Sequence):
                    assert feature_type.feature.dtype == 'float32', 'may need to add more types'
                    value = tuple([float(x) for x in value])

                encoded_tup[column_name] = value

            check_tup = DatasetTuple(**encoded_tup)
            assert check_tup in acc_dataset

    def test_import_hf_dataset_with_images(self, reset_db) -> None:
        skip_test_if_not_installed('datasets')
        import datasets

        # Test that datasets with images load properly
        t = pxt.io.import_huggingface_dataset('mnist', datasets.load_dataset('ylecun/mnist', split='test'))
        assert t.count() == 10000
        img = t.head(1)['image'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size == (28, 28)

    def test_import_hf_dataset_invalid(self, reset_db) -> None:
        skip_test_if_not_installed('datasets')
        with pytest.raises(excs.Error) as exc_info:
            pxt.io.import_huggingface_dataset('test', {})
        assert 'type(dataset)' in str(exc_info.value)
