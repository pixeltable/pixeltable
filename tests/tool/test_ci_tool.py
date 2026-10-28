import pytest

from tool.ci_tool import (
    ALTERNATIVE_PLATFORMS,
    BASIC_PLATFORMS,
    DEFAULT_XDIST_N,
    EXPENSIVE_PLATFORMS,
    EXPENSIVE_PYTEST,
    MAIN_PLATFORM,
    STANDARD_PP_XDIST_WORKERS,
    TRIGGERS,
    UBUNTU_LARGE_MEM_GB,
    UBUNTU_LARGE_VCPUS,
    VERY_EXPENSIVE_PYTEST,
    XDIST_WORKER_BUDGET_GB,
    build_configs,
    xdist_workers_for_host,
)

# The triggers under which a fully populated matrix is generated; 'pull_request' deliberately runs a reduced one.
FULL_TRIGGERS = tuple(t for t in TRIGGERS if t != 'pull_request')


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

    def test_schedule_matches_forced_dispatch(self) -> None:
        assert build_configs('schedule', force_all=False, has_aws_credentials=False) == build_configs(
            'workflow_dispatch', force_all=True, has_aws_credentials=False
        )

    @pytest.mark.parametrize('trigger', FULL_TRIGGERS)
    def test_expensive_tests_run_exactly_once(self, trigger: str) -> None:
        for force_all in (False, True):
            configs = build_configs(trigger, force_all=force_all, has_aws_credentials=False)
            expensive = [cfg for cfg in configs if cfg.pytest_options in (EXPENSIVE_PYTEST, VERY_EXPENSIVE_PYTEST)]
            assert len(expensive) == 1, [cfg.display_name for cfg in expensive]
            if force_all or trigger == 'schedule':
                assert expensive[0].pytest_options == VERY_EXPENSIVE_PYTEST

    def test_unknown_trigger_is_rejected(self) -> None:
        with pytest.raises(AssertionError):
            build_configs('push', force_all=False, has_aws_credentials=False)

    def test_xdist_worker_cap_formula(self) -> None:
        # 16 vCPU / 64 GB / 6 GB per worker -> min(16, 48 // 6) = 8
        assert xdist_workers_for_host(vcpus=16, mem_gb=64, worker_budget_gb=6) == 8
        # fewer vCPUs than the memory budget allows
        assert xdist_workers_for_host(vcpus=4, mem_gb=64, worker_budget_gb=6) == 4
        assert (
            xdist_workers_for_host(
                vcpus=UBUNTU_LARGE_VCPUS, mem_gb=UBUNTU_LARGE_MEM_GB, worker_budget_gb=XDIST_WORKER_BUDGET_GB
            )
            == STANDARD_PP_XDIST_WORKERS
        )
        assert STANDARD_PP_XDIST_WORKERS == 8

    @pytest.mark.parametrize('trigger', FULL_TRIGGERS)
    def test_standard_pp_caps_xdist_workers(self, trigger: str) -> None:
        configs = build_configs(trigger, force_all=True, has_aws_credentials=False)
        pp = [cfg for cfg in configs if cfg.display_name_prefix == 'standard++']
        assert len(pp) == 1
        assert pp[0].pytest_n == str(STANDARD_PP_XDIST_WORKERS)
        assert pp[0].matrix_entry['pytest-n'] == str(STANDARD_PP_XDIST_WORKERS)
        others = [cfg for cfg in configs if cfg.display_name_prefix != 'standard++']
        assert others
        assert all(cfg.pytest_n == DEFAULT_XDIST_N for cfg in others)

    def test_standard_plus_keeps_logical_workers(self) -> None:
        configs = build_configs('merge_group', force_all=False, has_aws_credentials=False)
        plus = next(cfg for cfg in configs if cfg.display_name_prefix == 'standard+')
        assert plus.pytest_n == DEFAULT_XDIST_N
        assert plus.matrix_entry['pytest-n'] == DEFAULT_XDIST_N
