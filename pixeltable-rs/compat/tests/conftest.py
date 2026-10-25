"""pytest configuration for running upstream Pixeltable tests against pixeltable-rs.

A replacement for upstream tests/conftest.py: same fixture names and semantics for everything the selected test files
use, implemented with the public API plus the two test hooks in `pixeltable._testing`:
- `reset_db()`: drop everything; the next operation starts with an empty catalog (upstream: clean_db() + reload);
- `reload_catalog()`: drop all in-memory state; the next call reloads it from disk.

Only the 'local' database root exists. Before pixeltable is imported, this module points PIXELTABLE_HOME at a
per-process temporary directory and installs the stubs for upstream-internal modules that pixeltable-rs does not
provide (see pxt_compat/stubs.py and compat/README.md).
"""

import functools
import http.server
import json
import logging
import os
import pathlib
import shutil
import site
import sys
import sysconfig
import tempfile
import threading
import urllib.parse
import warnings
from typing import Any, Callable, Iterator, get_args

import pytest
import requests
import tenacity
from _pytest.config import Config as PytestConfig, argparsing

_TESTS_DIR = pathlib.Path(__file__).parent
_COMPAT_DIR = _TESTS_DIR.parent
_RS_DIR = _COMPAT_DIR.parent
_UPSTREAM_ROOT = _RS_DIR.parent

if str(_COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPAT_DIR))

from pxt_compat import stubs  # noqa: E402
from pxt_compat.selection import manifest_files  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Process environment, set before pixeltable is imported (pixeltable-rs may read it at import time)
# ---------------------------------------------------------------------------------------------------------------------

_WORKER = os.environ.get('PYTEST_XDIST_WORKER', 'main')
_SESSION_TMP = pathlib.Path(
    tempfile.mkdtemp(prefix=f'pxt-compat-{_WORKER}-', dir=os.environ.get('PXT_COMPAT_TMPDIR') or None)
)
_HOME_DIR = _SESSION_TMP / '.pixeltable'
os.environ['PIXELTABLE_HOME'] = str(_HOME_DIR)
os.environ['PIXELTABLE_CONFIG'] = str(_HOME_DIR / 'config.toml')
os.environ['PIXELTABLE_LOG_LEVEL'] = 'DEBUG'
# hermetic: no hosted catalogs, no external database, no user media destinations
for _var in (
    'PIXELTABLE_API_KEY',
    'PIXELTABLE_API_URL',
    'PIXELTABLE_CLOUD_HOST',
    'PIXELTABLE_DB_CONNECT_STR',
    'PIXELTABLE_DB',
    'PIXELTABLE_PGDATA',
    'PIXELTABLE_INPUT_MEDIA_DEST',
    'PIXELTABLE_OUTPUT_MEDIA_DEST',
):
    os.environ.pop(_var, None)
# subprocesses started by tests (`python tests/script_with_udf.py`) must run this interpreter, with this package
os.environ['PATH'] = f'{pathlib.Path(sys.executable).parent}{os.pathsep}{os.environ.get("PATH", "")}'

# ---------------------------------------------------------------------------------------------------------------------
# Stubs for upstream modules/names that pixeltable-rs lacks, then import hygiene
# ---------------------------------------------------------------------------------------------------------------------

_COPIED_FILES = [_TESTS_DIR / f for f in manifest_files() if f.endswith('.py')] or sorted(_TESTS_DIR.glob('test_*.py'))
_IMPORT_SCAN = stubs.scan_imports(_COPIED_FILES, upstream_root=_UPSTREAM_ROOT)
stubs.install(_IMPORT_SCAN)

import pixeltable as pxt  # noqa: E402


def _check_import_hygiene() -> None:
    """`import pixeltable` must resolve to the package installed in this interpreter's environment (or to
    pixeltable-rs/python), never to the upstream checkout; `tests` must be compat/tests."""
    pxt_file = pathlib.Path(pxt.__file__).resolve()
    paths = sysconfig.get_paths()
    allowed_roots = {pathlib.Path(p).resolve() for p in (paths['purelib'], paths['platlib'])}
    allowed_roots.update(pathlib.Path(p).resolve() for p in site.getsitepackages())
    allowed_roots.add((_RS_DIR / 'python').resolve())
    upstream_pkg = (_UPSTREAM_ROOT / 'pixeltable').resolve()
    assert not pxt_file.is_relative_to(upstream_pkg), (
        f'`import pixeltable` resolved to the upstream checkout ({pxt_file}); run the harness through compat/run.py '
        f'or from compat/, not from {_UPSTREAM_ROOT}'
    )
    assert any(pxt_file.is_relative_to(root) for root in allowed_roots), (
        f'`import pixeltable` resolved to {pxt_file}, which is neither in site-packages of {sys.prefix} '
        f'nor in {_RS_DIR / "python"}'
    )
    tests_pkg = sys.modules.get('tests')
    assert tests_pkg is not None and pathlib.Path(tests_pkg.__file__).resolve().parent == _TESTS_DIR.resolve(), (
        f'package `tests` resolved to {getattr(tests_pkg, "__file__", None)}, expected {_TESTS_DIR}'
    )


_check_import_hygiene()

# Resolve every allow-listed pixeltable.* module now, so that the stubs are created deterministically in every process
# (including the xdist controller, which never imports test modules) and the header can report them.
for _name in sorted(_IMPORT_SCAN.modules, key=lambda n: (n.count('.'), n)):
    try:
        __import__(_name)
    except Exception:
        # a real module that fails to import (e.g. for a missing optional dependency) is not cached, so it fails again
        # where a test imports it
        pass
stubs.install_attribute_fallbacks(_IMPORT_SCAN)

from .utils import (  # noqa: E402
    DEFINITION_ERRORS,
    TESTS_DIR,
    DatabaseRoot,
    DbRootId,
    ReloadTester,
    cloud_env_configured,
    create_all_datatypes_tbl,
    create_img_tbl,
    create_test_tbl,
    local_embedding,
    skip_test_if_not_installed,
    validate_async_teardown,
)

_logger = logging.getLogger('pixeltable_test')
_LOG_FMT_STR = '%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s'


def _hooks() -> Any:
    """The pixeltable._testing module (reset_db(), reload_catalog())."""
    try:
        import pixeltable._testing as hooks
    except ImportError as exc:
        raise RuntimeError(
            'pixeltable._testing is not importable; the harness needs its reset_db() and reload_catalog() hooks'
        ) from exc
    return hooks


DO_RERUN: bool = True


def pytest_addoption(parser: argparsing.Parser) -> None:
    parser.addoption('--no-rerun', action='store_true', default=False, help='Do not rerun any failed tests.')


# upstream pyproject's filterwarnings entries that name Pixeltable classes; added here rather than in pytest.ini so that
# a pixeltable-rs that lacks one of the classes does not keep pytest from starting
_PXT_WARNING_FILTERS = [
    ('PixeltableWarning', 'error::pixeltable.exceptions.PixeltableWarning'),
    (
        'PixeltableDeprecationWarning',
        r'ignore:.*apply\(\) is deprecated:pixeltable.exceptions.PixeltableDeprecationWarning',
    ),
]


def pytest_configure(config: PytestConfig) -> None:
    global DO_RERUN  # noqa: PLW0603
    DO_RERUN = not config.getoption('--no-rerun')
    exceptions_module = sys.modules.get('pixeltable.exceptions')
    for cls_name, line in _PXT_WARNING_FILTERS:
        if exceptions_module is not None and isinstance(getattr(exceptions_module, cls_name, None), type):
            config.addinivalue_line('filterwarnings', line)
        else:
            warnings.warn(f'pixeltable.exceptions.{cls_name} is missing; filter {line!r} not applied', stacklevel=1)


def pytest_unconfigure(config: PytestConfig) -> None:
    if not os.environ.get('PXT_COMPAT_KEEP_HOME'):
        shutil.rmtree(_SESSION_TMP, ignore_errors=True)


def pytest_report_header(config: PytestConfig) -> list[str]:
    lines = [
        f'pixeltable: {pxt.__file__} (version {getattr(pxt, "__version__", "?")})',
        f'PIXELTABLE_HOME: {_HOME_DIR}',
    ]
    report = stubs.report()
    internal = [m for m, kind in report['modules'].items() if kind == 'internal']
    public = [m for m, kind in report['modules'].items() if kind == 'public']
    lines.append(f'stubbed internal modules ({len(internal)}): {", ".join(internal) or "-"}')
    lines.append(f'stubbed public modules ({len(public)}): {", ".join(public) or "-"}')
    for module, names in report['attributes'].items():
        lines.append(f'stubbed names in {module}: {", ".join(names)}')
    for name, err in DEFINITION_ERRORS.items():
        lines.append(f'tests.utils.{name} could not be defined: {err[:200]}')
    return lines


# Fixtures that download a Hugging Face model. Any test requesting one of these is implicitly a
# Hugging Face test and is marked 'very_expensive'.
_HF_FIXTURES = frozenset({'clip_embed', 'e5_embed', 'all_mpnet_embed'})


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _HF_FIXTURES.intersection(getattr(item, 'fixturenames', ())):
            item.add_marker(pytest.mark.very_expensive)

    offenders: list[str] = []
    for item in items:
        if 'uses_db' in getattr(item, 'fixturenames', ()):
            marker = item.get_closest_marker('db_roots')
            if marker is None or marker.args != ('local',):
                offenders.append(item.nodeid)
    if offenders:
        raise pytest.UsageError(
            "tests using the 'uses_db' fixture must be marked @pytest.mark.db_roots('local', reason='...'):\n  "
            + '\n  '.join(offenders)
        )


def pytest_runtest_setup(item: pytest.Item) -> None:
    _logger.info(f'Running Pixeltable test: {os.environ.get("PYTEST_CURRENT_TEST")}')


def pytest_runtest_teardown(item: pytest.Item) -> None:
    _logger.info(f'Finished Pixeltable test: {os.environ.get("PYTEST_CURRENT_TEST")}')


@pytest.fixture
def project_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A directory that is a project root, for a test that writes an application or schema file into it."""
    (tmp_path / 'pixeltable.toml').write_text('', encoding='utf-8')
    return tmp_path


@pytest.fixture(scope='session')
def init_env() -> Iterator[None]:
    # Upstream: chdir to the project root (the parent of tests/), point PIXELTABLE_HOME at a per-worker directory,
    # reinitialize the database, pxt.init(). PIXELTABLE_HOME is set when this module is imported, since pixeltable-rs
    # may read it at import time.
    os.chdir(_COMPAT_DIR)
    print(f'{"PIXELTABLE_HOME":25} = {os.environ.get("PIXELTABLE_HOME")}')
    _HOME_DIR.mkdir(parents=True, exist_ok=True)
    _hooks().reset_db()
    pxt.init()

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(_LOG_FMT_STR))
    logging.getLogger('pixeltable').addHandler(stdout_handler)
    test_logger = logging.getLogger('pixeltable_test')
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(stdout_handler)

    yield

    validate_async_teardown()


def _reset_catalog_state() -> None:
    """Start the test with an empty catalog. Upstream: clean_db(), Config/Env resets, reload_catalog(), file-cache
    resets; the reset_db() hook is responsible for the equivalent state in pixeltable-rs."""
    _hooks().reset_db()


def _validate_catalog_state() -> None:
    """Upstream validates the store after each test (catalog.validate_store()); pixeltable-rs may offer the same as
    the optional hook pixeltable._testing.validate_store()."""
    validate = getattr(_hooks(), 'validate_store', None)
    if validate is not None:
        validate()


@pytest.fixture(scope='function')
def uses_db(init_env: None, request: pytest.FixtureRequest) -> Iterator[None]:
    """Fixture for tests that interact with the underlying store. Cleans up the database before the test, and
    validates it after the test."""
    _reset_catalog_state()
    yield
    _validate_catalog_state()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Drive the catalog-backend and data-versioning axes, as upstream does.

    is_data_versioned: any test that (transitively) reaches is_data_versioned runs against both a data-versioned and
    an operational table.

    db_root: upstream runs a test against 'local', 'proxy' and a hosted database, or the roots its db_roots marker
    names. Only 'local' exists here, so every test runs once, unparametrized (as upstream does when 'local' is the only
    target), and a test whose marker excludes 'local' gets an empty parameter set (skipped).
    """
    if 'is_data_versioned' in metafunc.fixturenames:
        metafunc.parametrize('is_data_versioned', [True, False], ids=['data_versioned', 'operational'])

    if 'db_root' in metafunc.fixturenames:
        db_roots_marker = metafunc.definition.get_closest_marker('db_roots')
        if db_roots_marker is not None:
            params = db_roots_marker.args
            if not set(params) <= set(get_args(DbRootId)):
                raise pytest.UsageError(
                    f'Invalid db_roots marker args. Must be a nonempty subset of {get_args(DbRootId)}; got: {params!r}'
                )
            if (
                not isinstance(db_roots_marker.kwargs.get('reason'), str)
                or not db_roots_marker.kwargs['reason'].strip()
            ):
                raise pytest.UsageError("db_roots marker must include a nonempty 'reason' kwarg")
        else:
            params = ('local', 'proxy', 'cloud')

        if not cloud_env_configured():
            params = tuple(p for p in params if not p.startswith('cloud'))
        params = tuple(p for p in params if p == 'local')  # the harness has no proxy daemon

        if params != ('local',):
            metafunc.parametrize('db_root', params, indirect=True)


@pytest.fixture(scope='function')
def db_root(init_env: None, request: pytest.FixtureRequest) -> Iterator[DatabaseRoot]:
    """Parameterized variant of uses_db; here always the in-process catalog: DatabaseRoot('local', '', '')."""
    _reset_catalog_state()
    db_root_id = getattr(request, 'param', 'local')
    assert db_root_id == 'local', f'unsupported db_root {db_root_id!r}'
    yield DatabaseRoot('local', '', '')
    _validate_catalog_state()


def _requested_is_data_versioned(request: pytest.FixtureRequest) -> bool:
    """The `is_data_versioned` value requested by a dependent test, or True if it doesn't declare the axis."""
    return request.getfixturevalue('is_data_versioned') if 'is_data_versioned' in request.fixturenames else True


@pytest.fixture(scope='function')
def test_tbl(db_root: DatabaseRoot, request: pytest.FixtureRequest) -> pxt.Table:
    """The standard test table."""
    return create_test_tbl(
        db_root.make_catalog_path('test_tbl'), is_data_versioned=_requested_is_data_versioned(request)
    )


@pytest.fixture(scope='function')
def reload_tester(init_env: None) -> ReloadTester:
    return ReloadTester()


_SERVED_DIR = TESTS_DIR.parent  # compat/, which holds tests/ (with tests/data) and docs/resources


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


class SampleFileServer:
    """Handle on the HTTP server started by the `sample_file_server` fixture, which serves the compat tree."""

    base_url: str

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def url(self, path: str | pathlib.Path, db_root: DatabaseRoot | None = None) -> str:
        """Return the http:// URL of `path`, given either as an absolute path in the tree or relative to its root."""
        rel_path = pathlib.Path(path)
        if rel_path.is_absolute():
            assert rel_path.is_relative_to(_SERVED_DIR)
            rel_path = rel_path.relative_to(_SERVED_DIR)

        if db_root is not None and db_root.is_cloud:
            base_url = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/'
        else:
            base_url = self.base_url

        return base_url + urllib.parse.quote(rel_path.as_posix())


@pytest.fixture
def sample_file_server() -> Iterator[SampleFileServer]:
    """Serve the local sample files (`tests/data`, `docs/resources`, ...) over a localhost HTTP server."""
    handler = functools.partial(_QuietHandler, directory=str(_SERVED_DIR))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SampleFileServer(f'http://127.0.0.1:{server.server_address[1]}/')
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope='function')
def test_tbl_exprs(test_tbl: pxt.Table) -> list[Any]:
    """Same expressions as upstream. Upstream builds some with the internal `pixeltable.exprs` classes (Literal,
    InlineDict, InlineList); without them the fixture fails with the stub's 'not available in pixeltable-rs' error."""
    from pixeltable import exprs, functions as pxtf

    t = test_tbl
    return [
        t.c1,
        t.c7['*'].f1,
        exprs.Literal('test'),
        exprs.InlineDict(
            {'a': t.c1, 'b': t.c6.f1, 'c': 17, 'd': exprs.InlineDict({'e': t.c2}), 'f': exprs.InlineList([t.c3, t.c3])}
        ),
        exprs.InlineList([[t.c2, t.c2], [t.c2, t.c2]]),
        t.c2 > 5,
        t.c2 == None,
        ~(t.c2 > 5),
        (t.c2 > 5) & (t.c1 == 'test'),
        (t.c2 > 5) | (t.c1 == 'test'),
        pxtf.map(t.c7['*'].f5, lambda x: [x[3], x[2], x[1], x[0]]),
        t.c8[0, 1:],
        t.c2.isin([1, 2, 3]),
        t.c2.isin(t.c6.f5),
        t.c2.astype(pxt.Float | None),
        (t.c2 + 1).astype(pxt.Float | None),
        t.c2.apply(str),
        (t.c2 + 1).apply(str),
        t.c3.apply(str),
        t.c4.apply(str),
        t.c5.apply(str),
        t.c6.apply(str),
        t.c1.apply(json.loads),
        t.c8.errortype,
        t.c8.errormsg,
        pxtf.sum(t.c2, group_by=t.c4, order_by=t.c3),
    ]


@pytest.fixture(scope='function')
def all_datatypes_tbl(db_root: DatabaseRoot, request: pytest.FixtureRequest) -> pxt.Table:
    return create_all_datatypes_tbl(
        name=db_root.make_catalog_path('all_datatype_tbl'), is_data_versioned=_requested_is_data_versioned(request)
    )


@pytest.fixture(scope='function')
def img_tbl(db_root: DatabaseRoot, request: pytest.FixtureRequest) -> pxt.Table:
    # local: the full image set (upstream limits cloud/proxy roots to 20 rows)
    num_rows = 0 if db_root.id == 'local' else 20
    return create_img_tbl(db_root.make_catalog_path('test_img_tbl'), num_rows, _requested_is_data_versioned(request))


@pytest.fixture(scope='function')
def img_tbl_exprs(indexed_img_tbl: pxt.Table) -> list[Any]:
    t = indexed_img_tbl
    return [
        t.img.width,
        t.img.rotate(90),
        # we're using a list here, not a tuple; the latter turns into a list during the back/forth conversion
        t.img.rotate(90).resize([224, 224]),
        t.img.fileurl,
        t.img.localpath,
        t.img.similarity(string='red truck', idx='img_idx0'),
    ]


@pytest.fixture(scope='function')
def multi_img_tbl_exprs(multi_idx_img_tbl: pxt.Table) -> list[Any]:
    t = multi_idx_img_tbl
    return [t.img.similarity(string='red truck', idx='img_idx1'), t.img.similarity(string='red truck', idx='img_idx2')]


@pytest.fixture(scope='function')
def small_img_tbl(db_root: DatabaseRoot, request: pytest.FixtureRequest) -> pxt.Table:
    return create_img_tbl(
        db_root.make_catalog_path('small_img_tbl'), num_rows=40, is_data_versioned=_requested_is_data_versioned(request)
    )


@pytest.fixture(scope='function')
def indexed_img_tbl(db_root: DatabaseRoot, local_embed: pxt.Function, request: pytest.FixtureRequest) -> pxt.Table:
    t = create_img_tbl(
        db_root.make_catalog_path('indexed_img_tbl'),
        num_rows=40,
        is_data_versioned=_requested_is_data_versioned(request),
    )
    t.add_embedding_index(
        'img', idx_name='img_idx0', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    return t


@pytest.fixture(scope='function')
def multi_idx_img_tbl(db_root: DatabaseRoot, local_embed: pxt.Function, request: pytest.FixtureRequest) -> pxt.Table:
    t = create_img_tbl(
        db_root.make_catalog_path('multi_idx_img_tbl'),
        num_rows=4,
        is_data_versioned=_requested_is_data_versioned(request),
    )
    t.add_embedding_index(
        'img', idx_name='img_idx1', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    t.add_embedding_index(
        'img', idx_name='img_idx2', metric='cosine', image_embed=local_embed, string_embed=local_embed
    )
    return t


# Deterministic, download-free embedding (see tests.utils.local_embedding). Same shape as clip.
@pytest.fixture
def local_embed() -> pxt.Function:
    return local_embedding.using(dim=512)


@pytest.fixture(
    params=[
        pytest.param(False, id='local_embed'),
        pytest.param(True, id='clip_embed', marks=pytest.mark.very_expensive),
    ]
)
def clip_or_local(request: pytest.FixtureRequest) -> tuple[pxt.Function, bool]:
    if request.param:
        return request.getfixturevalue('clip_embed'), False
    return local_embedding.using(dim=512), True


@pytest.fixture(
    params=[
        pytest.param(False, id='local_embed'),
        pytest.param(True, id='mpnet_embed', marks=pytest.mark.very_expensive),
    ]
)
def mpnet_or_local(request: pytest.FixtureRequest) -> tuple[pxt.Function, bool]:
    if request.param:
        skip_test_if_not_installed('sentence_transformers')
        return request.getfixturevalue('all_mpnet_embed'), False
    return local_embedding.using(dim=512), True


def _retry_hf(fn: Callable) -> Callable:
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
        retry=tenacity.retry_if_exception_type(requests.ReadTimeout),
        reraise=True,
    )(fn)


@pytest.fixture(scope='session')
@_retry_hf
def clip_embed() -> pxt.Function:
    try:
        from pixeltable.functions.huggingface import clip

        return clip.using(model_id='openai/clip-vit-base-patch32')
    except ImportError:
        return None


@pytest.fixture(scope='session')
@_retry_hf
def e5_embed() -> pxt.Function:
    try:
        from pixeltable.functions.huggingface import sentence_transformer

        return sentence_transformer.using(model_id='intfloat/e5-large-v2')
    except ImportError:
        return None


@pytest.fixture(scope='session')
@_retry_hf
def all_mpnet_embed() -> pxt.Function:
    try:
        from pixeltable.functions.huggingface import sentence_transformer

        return sentence_transformer.using(model_id='all-mpnet-base-v2')
    except ImportError:
        return None
