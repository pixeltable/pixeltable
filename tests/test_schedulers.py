# type: ignore

import asyncio
import os
import sys
import tempfile
import threading
import types
from unittest.mock import AsyncMock, MagicMock, Mock

import cloudpickle
import pytest

import pixeltable as pxt
from pixeltable import func
from pixeltable.exec.expr_eval.globals import Dispatcher, ExprEvalCtx, FnCallArgs
from pixeltable.exec.expr_eval.schedulers import ModalScheduler, RateLimitsScheduler
from pixeltable.func import runtime_adapter as ra
from pixeltable.func.modal_adapter import ModalAdapter, _FilePayload, _invoke_pickled
from pixeltable.func.runtime_adapter import get_runtime_adapter
from pixeltable.func.signature import Batch


class DummyError(Exception):
    pass


@pxt.udf(batch_size=4)
def add_constant_batched(vals: Batch[int], *, offset: int = 0) -> Batch[int]:
    return [v + offset for v in vals]


@pxt.udf(gpu='A100')
def gpu_noop(x: int) -> int:
    return x


@pxt.udf(gpu='A100', pip=['torch'])
def gpu_noop_torch(x: int) -> int:
    return x


@pxt.udf(gpu='A100', batch_size=4)
def gpu_add_batched(vals: Batch[int], *, offset: int = 0) -> Batch[int]:
    return [v + offset for v in vals]


@pxt.udf(gpu='A100')
def media_probe(v: pxt.Video) -> int:
    import os as _os

    return _os.path.getsize(v)


def _fake_modal() -> Mock:
    fake_modal = Mock()
    fake_app = Mock()
    fake_app.run = Mock(return_value=MagicMock())  # MagicMock supports __enter__/__exit__
    fake_modal.App = Mock(return_value=fake_app)
    return fake_modal


class TestSchedulers:
    def test_modal_adapter_invoke_batch(self) -> None:
        """invoke_batch must reproduce CallableFunction.exec_batch semantics: constant kwargs collapsed to scalars,
        batched kwargs and positional args passed as lists. We simulate the remote by running the real entrypoint."""
        fn = add_constant_batched
        batch_args = [[1, 2, 3]]
        batch_kwargs = {'offset': [10, 10, 10]}

        adapter = ModalAdapter('modal:A100')
        captured: dict = {}

        def fake_remote(remote_fn, args: list, kwargs: dict):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return _invoke_pickled(cloudpickle.dumps(remote_fn.py_fn), args, kwargs)

        adapter._invoke_remote = fake_remote  # type: ignore[method-assign]

        result = adapter.invoke_batch(fn, batch_args, batch_kwargs)

        # Result matches the local batched execution path exactly.
        assert result == fn.exec_batch(batch_args, batch_kwargs)
        assert result == [11, 12, 13]
        # The constant kwarg was collapsed to a scalar before being sent to the remote.
        assert captured['kwargs'] == {'offset': 10}
        assert captured['args'] == [[1, 2, 3]]

    def test_modal_adapter_builds_app_once_per_spec(self) -> None:
        """The Modal app/function for a given image spec must be built and its context entered exactly once, even
        under concurrent (multi-threaded) first invocations; distinct specs get distinct apps."""
        adapter = ModalAdapter('modal:A100')
        fake_modal = _fake_modal()
        adapter._modal = lambda: fake_modal  # type: ignore[method-assign]

        threads = [threading.Thread(target=adapter._ensure_remote, args=(gpu_noop,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # One app + one entered context for the single image spec, despite 8 concurrent first invocations.
        assert fake_modal.App.call_count == 1
        ctx = fake_modal.App.return_value.run.return_value
        assert ctx.__enter__.call_count == 1

        # A UDF with a different image spec (pip=['torch']) builds a second app.
        adapter._ensure_remote(gpu_noop_torch)
        assert fake_modal.App.call_count == 2

        adapter.close()
        assert ctx.__exit__.called

    def test_modal_adapter_media_serialization(self) -> None:
        """File-backed media arguments are sent as inline bytes and materialized to a real local path on the
        remote, so the UDF sees a valid file. Verified end-to-end through the real remote entrypoint."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            payload = b'fake video bytes'
            f.write(payload)
            local_path = f.name
        try:
            enc_args, enc_kwargs = ModalAdapter._encode_media(media_probe, [local_path], {}, batched=False)
            # The local path was replaced by an inline payload carrying the file's bytes and suffix.
            assert isinstance(enc_args[0], _FilePayload)
            assert enc_args[0].data == payload
            assert enc_args[0].suffix == '.mp4'

            # Simulate the remote: the entrypoint must materialize the payload to a path the UDF can stat.
            pickled = cloudpickle.dumps(media_probe.py_fn)
            result = _invoke_pickled(pickled, enc_args, enc_kwargs)
            assert result == len(payload)
        finally:
            os.unlink(local_path)

    def test_runtime_adapter_registry(self) -> None:
        """get_runtime_adapter returns a cached singleton per pool; only modal: pools get an adapter."""
        a1 = get_runtime_adapter('modal:T4')
        a2 = get_runtime_adapter('modal:T4')
        try:
            assert a1 is a2
            assert isinstance(a1, ModalAdapter)
            # Distinct pools get distinct adapters.
            assert get_runtime_adapter('modal:A100') is not a1
            # No external adapter for rate-limited / pool-less calls.
            assert get_runtime_adapter('rate-limits:foo') is None
            assert get_runtime_adapter('request-rate:bar') is None
            assert get_runtime_adapter(None) is None
            # The CallableFunction resolves its adapter through the same registry.
            assert gpu_noop._runtime_adapter is get_runtime_adapter('modal:A100')
            assert add_constant_batched._runtime_adapter is None
        finally:
            ra._adapter_cache.pop('modal:T4', None)
            ra._adapter_cache.pop('modal:A100', None)

    def test_exec_routes_gpu_udf_to_adapter(self) -> None:
        """Synchronous CallableFunction.exec() must route a GPU UDF to the external runtime adapter (the path used
        by embedding-index query-time embedding), not run it in-process."""
        adapter = get_runtime_adapter('modal:A100')
        captured: dict = {}

        def fake_remote(remote_fn, args: list, kwargs: dict):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return 999

        adapter._invoke_remote = fake_remote  # type: ignore[method-assign]
        try:
            result = gpu_noop.exec([5], {})
            assert result == 999
            assert captured['args'] == [5]
            assert captured['kwargs'] == {}
        finally:
            ra._adapter_cache.pop('modal:A100', None)

    def test_exec_batch_routes_gpu_udf_to_adapter(self) -> None:
        """Both exec_batch() (many rows) and exec() (single row of a batched UDF) route to the adapter and reproduce
        the local batched semantics."""
        adapter = get_runtime_adapter('modal:A100')

        def fake_remote(remote_fn, args: list, kwargs: dict):
            return _invoke_pickled(cloudpickle.dumps(remote_fn.py_fn), args, kwargs)

        adapter._invoke_remote = fake_remote  # type: ignore[method-assign]
        try:
            # exec_batch: constant kwarg collapsed, batched positional passed as a list.
            batch_result = gpu_add_batched.exec_batch([[1, 2, 3]], {'offset': [10, 10, 10]})
            assert batch_result == [11, 12, 13]

            # exec (single row of a batched UDF): packs into a singleton batch and unwraps the result.
            scalar_result = gpu_add_batched.exec([5], {'offset': 10})
            assert scalar_result == 15
        finally:
            ra._adapter_cache.pop('modal:A100', None)

    def test_exec_routes_on_running_loop(self) -> None:
        """When exec() is called from inside a running event loop (e.g. query-time embedding on the executor loop),
        the adapter call is hopped to a worker thread and still returns correctly."""
        adapter = get_runtime_adapter('modal:A100')

        def fake_remote(remote_fn, args: list, kwargs: dict):
            return 7

        adapter._invoke_remote = fake_remote  # type: ignore[method-assign]

        async def run() -> int:
            # Calling the synchronous exec() from within a coroutine -> there is a running loop.
            await asyncio.sleep(0)
            return gpu_noop.exec([1], {})

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(run())
            assert result == 7
        finally:
            loop.close()
            ra._adapter_cache.pop('modal:A100', None)

    def test_poolless_exec_unchanged(self) -> None:
        """Pool-less UDFs must continue to execute in-process (no adapter involvement)."""
        assert add_constant_batched._runtime_adapter is None
        assert add_constant_batched.exec_batch([[1, 2, 3]], {'offset': [5, 5, 5]}) == [6, 7, 8]
        assert add_constant_batched.exec([2], {'offset': 5}) == 7

    def test_dumps_by_value_inlines_sibling(self) -> None:
        """_dumps_by_value must serialize a UDF's sibling module-level helpers by value, so the pickle is
        self-contained on a remote that does not have the user's module installed."""
        # Build a user-style module (importable name, non-site-packages file) whose `caller` references `helper`.
        mod = types.ModuleType('pxt_fake_udf_mod')
        # A user-code file path (not under site-packages/stdlib); it need not exist on disk.
        mod.__file__ = __file__ + '.fake_udf_mod.py'
        exec('def helper(x):\n    return x + 100\n\ndef caller(x):\n    return helper(x)\n', mod.__dict__)
        sys.modules['pxt_fake_udf_mod'] = mod
        adapter = ModalAdapter('modal:A100')
        try:
            pickled_by_value = adapter._dumps_by_value(mod.caller)
            pickled_by_ref = cloudpickle.dumps(mod.caller)
        finally:
            del sys.modules['pxt_fake_udf_mod']

        # The module is now gone and not importable from disk, simulating the remote container.
        # By-value pickle is self-contained and runs; by-reference pickle cannot resolve the module.
        assert cloudpickle.loads(pickled_by_value)(5) == 105
        with pytest.raises(ModuleNotFoundError):
            cloudpickle.loads(pickled_by_ref)

    def test_modal_scheduler_in_flight_cap(self) -> None:
        """The scheduler must not run more than MAX_IN_FLIGHT remote invocations concurrently."""

        async def run_test():
            mock_dispatcher = Mock(spec=Dispatcher)
            mock_dispatcher.register_task = Mock()
            mock_dispatcher.dispatch = Mock()
            mock_dispatcher.dispatch_exc = Mock()
            mock_dispatcher.exc_event = asyncio.Event()

            scheduler = ModalScheduler('modal:A100', mock_dispatcher)
            scheduler.MAX_IN_FLIGHT = 2

            lock = threading.Lock()
            state = {'current': 0, 'max': 0}
            release = threading.Event()

            def blocking_invoke(fn, args, kwargs):
                with lock:
                    state['current'] += 1
                    state['max'] = max(state['max'], state['current'])
                release.wait(timeout=5)
                with lock:
                    state['current'] -= 1
                return 0

            scheduler.adapter = Mock()
            scheduler.adapter.invoke = blocking_invoke
            scheduler.adapter.close = Mock()

            for _ in range(6):
                mock_row = MagicMock()
                mock_row.has_val = [False]
                mock_row.has_exc = Mock(return_value=False)
                mock_fn = Mock(spec=func.CallableFunction)
                mock_fn_call = Mock()
                mock_fn_call.fn = mock_fn
                mock_fn_call.slot_idx = 0
                request = Mock(spec=FnCallArgs)
                request.fn_call = mock_fn_call
                request.rows = [mock_row]
                request.row = mock_row
                request.is_batched = False
                request.args = [1]
                request.kwargs = {}
                scheduler.submit(request, Mock(spec=ExprEvalCtx))

            # Give the loop time to dispatch as many tasks as the cap allows while invocations are blocked.
            await asyncio.sleep(0.3)
            assert scheduler.num_in_flight == 2
            assert state['max'] == 2

            # Release the blocked invocations and let everything drain.
            release.set()
            for _ in range(50):
                await asyncio.sleep(0.05)
                if scheduler.num_in_flight == 0 and scheduler.total_requests == 6:
                    break
            assert scheduler.total_requests == 6
            assert state['max'] == 2

            for call in mock_dispatcher.register_task.call_args_list:
                call.args[0].cancel()

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_test())
        finally:
            loop.close()

    def test_modal_scheduler_exec(self) -> None:
        async def run_test():
            mock_dispatcher = Mock(spec=Dispatcher)
            mock_dispatcher.register_task = Mock()
            mock_dispatcher.dispatch = Mock()
            mock_dispatcher.dispatch_exc = Mock()
            mock_dispatcher.exc_event = asyncio.Event()

            scheduler = ModalScheduler('modal:A100', mock_dispatcher)
            scheduler.adapter = Mock()
            scheduler.adapter.invoke = Mock(return_value=2)

            mock_fn = Mock(spec=func.CallableFunction)
            mock_fn_call = Mock()
            mock_fn_call.fn = mock_fn
            mock_fn_call.slot_idx = 0

            mock_row = MagicMock()
            mock_row.has_val = [False]
            mock_row.has_exc = Mock(return_value=False)

            mock_request = Mock(spec=FnCallArgs)
            mock_request.fn_call = mock_fn_call
            mock_request.rows = [mock_row]
            mock_request.row = mock_row
            mock_request.is_batched = False
            mock_request.args = [1]
            mock_request.kwargs = {}

            mock_ctx = Mock(spec=ExprEvalCtx)

            await scheduler._exec(mock_request, mock_ctx, is_task=False)

            scheduler.adapter.invoke.assert_called_once_with(mock_fn, [1], {})
            mock_row.__setitem__.assert_called_once_with(0, 2)
            mock_dispatcher.dispatch.assert_called_once_with([mock_row], mock_ctx)

            for call in mock_dispatcher.register_task.call_args_list:
                call.args[0].cancel()

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_test())
        finally:
            loop.close()

    def test_rate_limits_scheduler_exception_before_pool(self) -> None:
        """Test that RateLimitsScheduler properly handles exceptions if pool_info is not yet set."""

        async def run_test():
            mock_dispatcher = Mock(spec=Dispatcher)
            mock_dispatcher.register_task = Mock()
            mock_dispatcher.dispatch_exc = Mock()

            scheduler = RateLimitsScheduler('rate-limits:test', mock_dispatcher)

            # Create a mock function that will throw an exception
            mock_fn = Mock(spec=func.CallableFunction)
            mock_fn.aexec = AsyncMock(side_effect=DummyError('Test exception'))
            mock_fn.signature = Mock()
            mock_fn.signature.system_parameters = {'_runtime_ctx'}

            mock_fn_call = Mock()
            mock_fn_call.fn = mock_fn
            mock_fn_call.slot_idx = 0
            mock_fn_call.get_param_values = Mock(return_value=[{}])

            mock_row = Mock()
            mock_row.has_val = [False]
            mock_row.has_exc = Mock(return_value=False)
            mock_row.set_exc = Mock()
            mock_row.__setitem__ = Mock()

            mock_request = Mock(spec=FnCallArgs)
            mock_request.fn_call = mock_fn_call
            mock_request.rows = [mock_row]
            mock_request.row = mock_row
            mock_request.is_batched = False
            mock_request.args = []
            mock_request.kwargs = {}

            mock_ctx = Mock(spec=ExprEvalCtx)

            await scheduler._exec(mock_request, mock_ctx, 0, is_task=False)

            # Verify that the exception was recorded
            assert mock_row.set_exc.called
            assert mock_dispatcher.dispatch_exc.called

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_test())
        finally:
            loop.close()
