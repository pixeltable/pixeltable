import logging

import pytest

import pixeltable as pxt
from pixeltable.functions import reve

from ..utils import IN_CI, rerun_on_network_error, skip_test_if_no_client, validate_update_status

pytestmark = pytest.mark.local('UDF/integration test')

_logger = logging.getLogger('pixeltable_test')


@pytest.mark.remote_api
@pytest.mark.very_expensive
@rerun_on_network_error()
class TestReve:
    @pytest.mark.parametrize('default_params', [True, False], ids=['default_params', 'nondefault_params'])
    def test_create(self, default_params: bool, uses_db: None) -> None:
        skip_test_if_no_client('reve')

        t = pxt.create_table('test_tbl', {'logo': pxt.Image})

        # Create from prompt
        kwargs = (
            {}
            if default_params
            else {
                'aspect_ratio': '1:1',
                'version': 'latest',
                'postprocessing': [{'process': 'effect', 'effect_name': 'low_light'}],
            }
        )
        t.add_computed_column(
            city_skyline=(
                reve.create('A futuristic city skyline with at night with a skyscraper featured prominently', **kwargs)
            )
        )

        # Create from prompt with reference images
        kwargs = (
            {}
            if default_params
            else {
                'aspect_ratio': '1:1',
                'version': 'latest',
                'postprocessing': [{'process': 'fit_image', 'max_dim': 1024}],
            }
        )
        t.add_computed_column(
            city_skyline_with_logo=(
                reve.create(
                    'Put the company logo from <frame>1</frame> (without the company name) on the skyscraper from '
                    '<frame>0</frame>',
                    references=[t.city_skyline['image'], t.logo],
                    **kwargs,
                )
            )
        )
        # Extract json layout of the generated image to a separate column
        t.add_computed_column(city_skyline_with_logo_layout=t.city_skyline_with_logo['layout'])

        validate_update_status(t.insert(logo='./docs/resources/pixeltable-logo-large.png'), expected_rows=1)
        row = t.select().collect()[0]
        layout = row['city_skyline_with_logo_layout']
        assert 'prompt' in layout
        assert 'regions' in layout
        assert len(layout['regions']) > 0

        if not IN_CI:
            # When running on a dev machine, save generated images to a temporary location, and print filepaths so that
            # the images can be manually examined.
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                skyline_path = f.name
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                skyline_with_logo_path = f.name
            row['city_skyline']['image'].save(skyline_path)
            row['city_skyline_with_logo']['image'].save(skyline_with_logo_path)

            _logger.info('original logo: %s', row['logo'].filename)
            _logger.info('reve: city skyline: %s', skyline_path)
            _logger.info(
                'reve: city skyline with pxt logo: %s, layout: %s',
                skyline_with_logo_path,
                row['city_skyline_with_logo_layout'],
            )
