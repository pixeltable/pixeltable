import json
import pathlib
import re

import pytest

import pixeltable as pxt
from pixeltable.config import Config

from .utils import pxt_raises


@pxt.udf
def double(x: int) -> int:
    return 2 * x


class TestModal:
    @pytest.mark.db_roots('local', reason="a proxy catalog's daemon evaluates its queries with its own config")
    def test_udf_resources_config(
        self, uses_db: None, project_env: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """udf_resources comes from the project config, the environment or pxt.init(), and a query that calls a UDF
        reports an invalid entry together with its source."""
        t = pxt.create_table('tbl', {'x': pxt.Int})
        t.insert([{'x': 1}])
        config_file = project_env / 'pixeltable.toml'
        in_file = rf'pixeltable\.udf_resources in {re.escape(str(config_file))}'

        def load(text: str) -> None:
            config_file.write_text(text, encoding='utf-8')
            Config.init(reinit=True, project_root=project_env)

        def entry(path: str, settings: str = 'cpu = 4') -> str:
            return f"[pixeltable.udf_resources.'{path}']\n{settings}\n"

        # built-in UDFs, one of them under a re-exported path
        valid = (
            entry('pixeltable.functions.video.clip', 'cpu = 4\nmemory_mb = 8192\ntimeout_seconds = 600')
            + entry('pixeltable.functions.huggingface.image_captioning', "gpu = 'T4'")
            + entry('pixeltable.functions.image.rotate', 'cpu = 0.5')
        )
        load(valid)
        assert t.select(y=double(t.x)).collect()['y'] == [2]

        for text, message in (
            (entry('pixeltable.functions.video.clip', 'cpus = 4'), r'Extra inputs are not permitted'),
            (entry('pixeltable.functions.video.clip', 'cpu = 0'), r'greater than 0'),
            (entry('pixeltable.functions.video.clip', "memory_mb = 'lots'"), r'valid integer'),
            (entry('pixeltable.functions.video.clip', "gpu = ''"), r'at least 1 character'),
            ("[pixeltable.udf_resources]\n'pixeltable.functions.video.clip' = 4\n", r'valid dictionary'),
            (entry('pixeltable.functions.video.no_such_udf'), r'is not a UDF declared with @pxt\.udf'),
            (entry('no_such_module.no_such_udf'), r'is not a UDF declared with @pxt\.udf'),
            (entry('pixeltable.functions.video'), r'is not a UDF declared with @pxt\.udf'),
            (entry('pixeltable.functions.count'), r'is not a UDF declared with @pxt\.udf'),
            (entry('tests.test_modal.double'), r'is not a built-in UDF'),
            (entry('pixeltable.functions.openai.chat_completions'), r'is async'),
            (
                entry('pixeltable.functions.video.clip') + entry('pixeltable.functions.video.editing.clip'),
                r"'pixeltable\.functions\.video\.clip' and 'pixeltable\.functions\.video\.editing\.clip' are the "
                r'same UDF',
            ),
        ):
            load(text)
            with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION, match=rf'(?s){in_file}.*{message}'):
                t.select(y=double(t.x)).collect()

        # a value that is not a table is refused when the file is read
        with pxt_raises(
            pxt.ErrorCode.INVALID_CONFIGURATION, match=r"Invalid type for option 'pixeltable\.udf_resources'"
        ):
            load('[pixeltable]\nudf_resources = 4\n')

        # the environment wins over the file, and pxt.init() wins over both
        load(entry('pixeltable.functions.video.clip', 'cpu = 0'))
        monkeypatch.setenv('PIXELTABLE_UDF_RESOURCES', json.dumps({'pixeltable.functions.video.clip': {'cpu': 4}}))
        assert t.select(y=double(t.x)).collect()['y'] == [2]
        monkeypatch.setenv('PIXELTABLE_UDF_RESOURCES', json.dumps({'pixeltable.functions.video.clip': {'cpus': 4}}))
        with pxt_raises(
            pxt.ErrorCode.INVALID_CONFIGURATION,
            match=r'(?s)PIXELTABLE_UDF_RESOURCES in the environment.*Extra inputs are not permitted',
        ):
            t.select(y=double(t.x)).collect()
        monkeypatch.setenv('PIXELTABLE_UDF_RESOURCES', 'not json')
        with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION, match=r"'pixeltable\.udf_resources': not json"):
            t.select(y=double(t.x)).collect()

        Config.init(
            config_overrides={'pixeltable.udf_resources': {'pixeltable.functions.video.clip': {'cpu': 4}}},
            reinit=True,
            project_root=project_env,
        )
        assert t.select(y=double(t.x)).collect()['y'] == [2]
        Config.init(
            config_overrides={'pixeltable.udf_resources': {'pixeltable.functions.video.clip': {'cpu': -1}}},
            reinit=True,
            project_root=project_env,
        )
        with pxt_raises(
            pxt.ErrorCode.INVALID_CONFIGURATION,
            match=r'(?s)pixeltable\.udf_resources in the config_overrides of pxt\.init\(\).*greater than 0',
        ):
            t.select(y=double(t.x)).collect()
