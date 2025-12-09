# type: ignore

import asyncio
from unittest.mock import AsyncMock, Mock

from pixeltable import func
from pixeltable.exec.expr_eval.globals import Dispatcher, ExprEvalCtx, FnCallArgs
from pixeltable.exec.expr_eval.schedulers import RateLimitsScheduler


class DummyError(Exception):
    pass


class TestSchedulers:
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
