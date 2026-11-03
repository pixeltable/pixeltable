"""The router stub that stands in for FastAPIRouter when fastapi is not installed."""

from pixeltable.serving import ROUTE_DEFINITION_METHODS

from ..utils import skip_test_if_not_installed


class TestStubRouter:
    def test_declaration_surface(self) -> None:
        """The stub answers every route declaration the real router offers, so an app file imports either way."""
        skip_test_if_not_installed('fastapi')
        from pixeltable.serving import FastAPIRouter

        # vars(), not dir(): fastapi.APIRouter contributes route methods of its own that an application
        # file does not call, and add_api_route is what fastapi itself calls on the way to serving
        declared = {
            name
            for name in vars(FastAPIRouter)
            if name.endswith('_route') and not name.startswith('_') and name != 'add_api_route'
        }
        assert declared == set(ROUTE_DEFINITION_METHODS), declared.symmetric_difference(ROUTE_DEFINITION_METHODS)
