import pytest

from tool.ci_tool import (
    ALTERNATIVE_PLATFORMS,
    BASIC_PLATFORMS,
    DEFAULT_PYTEST,
    EXPENSIVE_PLATFORMS,
    EXPENSIVE_PYTEST,
    MAIN_PLATFORM,
    RESOURCE_INTENSIVE_PYTEST,
    SCHEDULED_PYTEST,
    TRIGGERS,
    VERY_EXPENSIVE_PYTEST,
    build_configs,
)

TIERED_PYTEST = (EXPENSIVE_PYTEST, VERY_EXPENSIVE_PYTEST, RESOURCE_INTENSIVE_PYTEST)


class TestBuildConfigs:
    """Invariants on the generated CI matrix.

    A mistake in CI matrix can silently reduce test coverage.
    """

    @pytest.mark.parametrize('trigger', TRIGGERS)
    def test_matrix_is_populated(self, trigger: str) -> None:
        configs = build_configs(trigger, force_all=False, has_aws_credentials=False)
        assert sum(cfg.test_category == 'lint' for cfg in configs) == 1
        assert any(cfg.test_category == 'py' and cfg.os == MAIN_PLATFORM for cfg in configs)

    @pytest.mark.parametrize('trigger', TRIGGERS)
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

    def test_weekly_schedule_matches_forced_dispatch(self) -> None:
        assert build_configs('schedule', force_all=True, has_aws_credentials=False) == build_configs(
            'workflow_dispatch', force_all=True, has_aws_credentials=False
        )

    @pytest.mark.parametrize(
        ('trigger', 'force_all', 'tiered'),
        [
            ('merge_group', False, ()),
            ('workflow_dispatch', False, ()),
            ('schedule', False, (EXPENSIVE_PYTEST, RESOURCE_INTENSIVE_PYTEST)),
            ('schedule', True, (VERY_EXPENSIVE_PYTEST, RESOURCE_INTENSIVE_PYTEST)),
            ('workflow_dispatch', True, (VERY_EXPENSIVE_PYTEST, RESOURCE_INTENSIVE_PYTEST)),
        ],
    )
    def test_tiered_tests_run_only_when_scheduled(self, trigger: str, force_all: bool, tiered: tuple[str, ...]) -> None:
        configs = build_configs(trigger, force_all=force_all, has_aws_credentials=True)
        py_options = [cfg.pytest_options for cfg in configs if cfg.test_category == 'py']
        assert sorted(opts for opts in py_options if opts in TIERED_PYTEST) == sorted(tiered)
        untiered = SCHEDULED_PYTEST if trigger == 'schedule' or force_all else DEFAULT_PYTEST
        assert {opts for opts in py_options if opts not in TIERED_PYTEST} == {untiered}

    def test_unknown_trigger_is_rejected(self) -> None:
        with pytest.raises(AssertionError):
            build_configs('push', force_all=False, has_aws_credentials=False)
