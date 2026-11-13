import pytest

from tool.ci_tool import (
    ALTERNATIVE_PLATFORMS,
    BASIC_PLATFORMS,
    DEFAULT_MARKERS,
    EXPENSIVE_MARKERS,
    EXPENSIVE_PLATFORMS,
    MAIN_PLATFORM,
    RESOURCE_INTENSIVE_MARKERS,
    TRIGGERS,
    VERY_EXPENSIVE_MARKERS,
    build_configs,
)

# The triggers under which a fully populated matrix is generated; 'pull_request' deliberately runs a reduced one.
FULL_TRIGGERS = tuple(t for t in TRIGGERS if t != 'pull_request')

TIERED_MARKERS = (EXPENSIVE_MARKERS, VERY_EXPENSIVE_MARKERS, RESOURCE_INTENSIVE_MARKERS)


class TestBuildConfigs:
    """Invariants on the generated CI matrix.

    A mistake in CI matrix can silently reduce test coverage.
    """

    @pytest.mark.parametrize('trigger', TRIGGERS)
    def test_matrix_is_populated(self, trigger: str) -> None:
        configs = build_configs(trigger, force_all=False, has_aws_credentials=False)
        assert sum(cfg.test_category == 'lint' for cfg in configs) == 1
        assert any(cfg.test_category == 'py' and cfg.os == MAIN_PLATFORM for cfg in configs)

    @pytest.mark.parametrize('trigger', FULL_TRIGGERS)
    def test_basic_platforms_get_a_full_install(self, trigger: str) -> None:
        configs = build_configs(trigger, force_all=False, has_aws_credentials=False)
        full_install = {cfg.os for cfg in configs if cfg.test_category == 'py' and cfg.uv_options == ''}
        assert {MAIN_PLATFORM, *BASIC_PLATFORMS} <= full_install

    def test_every_declared_platform_is_used(self) -> None:
        configs = build_configs('workflow_dispatch', force_all=True, has_aws_credentials=True)
        assert {MAIN_PLATFORM, *BASIC_PLATFORMS, *EXPENSIVE_PLATFORMS, *ALTERNATIVE_PLATFORMS} <= {
            cfg.os for cfg in configs
        }

    @pytest.mark.parametrize('trigger', TRIGGERS)
    def test_display_names_are_unique(self, trigger: str) -> None:
        names = [cfg.display_name for cfg in build_configs(trigger, force_all=True, has_aws_credentials=True)]
        assert len(names) == len(set(names)), sorted(names)

    @pytest.mark.parametrize('force_all', [False, True])
    def test_dispatch_matches_schedule(self, force_all: bool) -> None:
        assert build_configs('schedule', force_all=force_all, has_aws_credentials=False) == build_configs(
            'workflow_dispatch', force_all=force_all, has_aws_credentials=False
        )

    @pytest.mark.parametrize(
        ('trigger', 'force_all', 'tiered'),
        [
            ('pull_request', False, ()),
            ('merge_group', False, ()),
            ('schedule', False, (EXPENSIVE_MARKERS, RESOURCE_INTENSIVE_MARKERS)),
            ('schedule', True, (VERY_EXPENSIVE_MARKERS, RESOURCE_INTENSIVE_MARKERS)),
        ],
    )
    def test_markers_by_trigger(self, trigger: str, force_all: bool, tiered: tuple[str, ...]) -> None:
        configs = build_configs(trigger, force_all=force_all, has_aws_credentials=True)
        markers = [cfg.pytest_markers for cfg in configs if cfg.test_category == 'py']
        assert sorted(m for m in markers if m in TIERED_MARKERS) == sorted(tiered)
        untiered = (
            f'{DEFAULT_MARKERS} and not skip_merge_queue'
            if trigger in ('pull_request', 'merge_group')
            else DEFAULT_MARKERS
        )
        assert {m for m in markers if m not in TIERED_MARKERS} == {untiered}

    def test_unknown_trigger_is_rejected(self) -> None:
        with pytest.raises(AssertionError):
            build_configs('push', force_all=False, has_aws_credentials=False)
