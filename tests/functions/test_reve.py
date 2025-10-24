import logging

import pytest

import pixeltable as pxt
from pixeltable.functions import reve

from ..utils import rerun, validate_update_status

_logger = logging.getLogger('pixeltable')


@pytest.mark.remote_api
@pytest.mark.expensive
@rerun(reruns=3, reruns_delay=8)
class TestReve:
    def test_create_edit_remix(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'pixeltable_logo': pxt.Image})
        t.add_computed_column(just_logo=(reve.edit(t.pixeltable_logo, 'extract the company logo and drop the name')))
        t.add_computed_column(
            city_skyline=(
                reve.create(
                    'A futuristic city skyline with at night with a skyscraper featured prominently',
                    format='jpeg',
                    aspect_ratio='1:1',
                )
            )
        )
        t.add_computed_column(
            city_skyline_with_pxt=(
                reve.remix(
                    'Put a company logo from $0 on the skyscraper from $1',
                    images=[t.just_logo, t.city_skyline],
                    output_format='jpeg',
                    aspect_ratio='16:9',
                )
            )
        )
        validate_update_status(t.insert(pixeltable_logo='./docs/resources/pixeltable-logo-large.png'), expected_rows=1)
        for row in t.select().collect():
            _logger.info('original logo: %s', row['pixeltable_logo'].filename)
            _logger.info('reve: just logo with text removed: %s', row['just_logo'].filename)
            _logger.info('reve: city skyline: %s', row['city_skyline'].filename)
            _logger.info('reve: city skyline with pxt logo: %s', row['city_skyline_with_pxt'].filename)
