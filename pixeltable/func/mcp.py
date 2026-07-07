import inspect
from typing import TYPE_CHECKING, Any, Sequence

import pixeltable as pxt
from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env
from pixeltable.func.function import Function
from pixeltable.func.signature import Parameter, Signature

if TYPE_CHECKING:
    import mcp


def mcp_udfs(url: str) -> list['pxt.func.Function']:
    from pixeltable.runtime import get_runtime

    return get_runtime().run_coro(mcp_udfs_async(url))


async def mcp_udfs_async(url: str) -> list['pxt.func.Function']:
    Env.get().require_package('mcp')
    import mcp
    from mcp.client.streamable_http import streamablehttp_client

    list_tools_result: mcp.types.ListToolsResult | None = None
    async with (
        streamablehttp_client(url) as (read_stream, write_stream, _),
        mcp.ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        list_tools_result = await session.list_tools()
    assert list_tools_result is not None

    return [mcp_tool_to_udf(url, tool) for tool in list_tools_result.tools]


def mcp_tool_to_udf(url: str, mcp_tool: 'mcp.types.Tool') -> 'pxt.func.Function':
    return McpFunction(url, mcp_tool.name, [_signature_from_tool(mcp_tool)], mcp_tool.name, mcp_tool.description)


class McpFunction(Function):
    """A Pixeltable function backed by a tool on a remote MCP server.

    Serialized by reference (server url, tool name, and signature) so that it can be persisted in a computed column and
    reconstructed later without contacting the server.
    """

    url: str
    tool_name: str
    self_name: str
    _comment: str | None

    # whether the server's current tool has been checked against this signature
    _is_verified: bool

    def __init__(self, url: str, tool_name: str, signatures: list[Signature], self_name: str, comment: str | None):
        self.url = url
        self.tool_name = tool_name
        self.self_name = self_name
        self._comment = comment

        # set on first aexec() so that reconstructing the function (eg, when loading a computed column) does not require
        # contacting the server
        self._is_verified = False

        super().__init__(signatures, self_path=None)

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        pass  # an MCP tool has a single signature

    @property
    def is_storable(self) -> bool:
        return True

    @property
    def is_async(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self.self_name

    @property
    def display_name(self) -> str:
        return self.self_name

    def comment(self) -> str | None:
        return self._comment

    async def aexec(self, *args: Any, **kwargs: Any) -> str:
        # TODO: open one session per McpFunction rather than one per call
        Env.get().require_package('mcp')
        import mcp
        from mcp.client.streamable_http import streamablehttp_client

        error_msg: str | None = None
        result: str | None = None
        async with (
            streamablehttp_client(self.url) as (read_stream, write_stream, _),
            mcp.ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            if not self._is_verified:
                current_tool = next(
                    (tool for tool in (await session.list_tools()).tools if tool.name == self.tool_name), None
                )
                if current_tool is None:
                    error_msg = (
                        f'MCP tool {self.tool_name!r} is no longer available at {self.url}.\n'
                        f'Recreate the affected column(s) from the current pxt.mcp_udfs({self.url!r}).'
                    )
                elif _tool_interface(_signature_from_tool(current_tool)) != _tool_interface(self.signature):
                    error_msg = (
                        f'The interface of MCP tool {self.tool_name!r} at {self.url} has changed since the column\n'
                        f'using it was created. Recreate the affected column(s) from the current '
                        f'pxt.mcp_udfs({self.url!r}).'
                    )
                else:
                    self._is_verified = True

            if self._is_verified:
                res = await session.call_tool(name=self.tool_name, arguments=kwargs)
                text_blocks = [item for item in res.content if isinstance(item, mcp.types.TextContent)]
                joined_text = '\n'.join(block.text for block in text_blocks)
                if res.isError:
                    detail = joined_text if len(text_blocks) > 0 else str(res.content)
                    error_msg = f'MCP tool {self.tool_name!r} at {self.url} reported an error:\n{detail}'
                elif len(res.content) > 0 and len(text_blocks) == len(res.content):
                    result = joined_text
                else:
                    # TODO: support image/audio and other non-text tool responses
                    kinds = [type(item).__name__ for item in res.content]
                    error_msg = (
                        f'MCP tool {self.tool_name!r} at {self.url} returned an unsupported response; only text '
                        f'results are supported, got: {kinds}.'
                    )

        # raise outside the client's async context: anyio wraps an exception thrown inside it in a TaskGroup group
        assert result is not None or error_msg is not None
        if error_msg is not None:
            raise excs.Error(excs.ErrorCode.GENERIC_USER_ERROR, error_msg)
        return result

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        from pixeltable.runtime import get_runtime

        return get_runtime().run_coro(self.aexec(*args, **kwargs))

    def _as_dict(self) -> dict:
        return {
            'url': self.url,
            'tool_name': self.tool_name,
            'signature': self.signature.as_dict(),
            'name': self.self_name,
            'comment': self._comment,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        return cls(d['url'], d['tool_name'], [Signature.from_dict(d['signature'])], d['name'], d.get('comment'))


def _tool_interface(sig: Signature) -> dict:
    # An order-independent view of a tool's parameters and return type. MCP passes arguments by name, so parameter
    # order is not part of the interface and a reordering of the server's schema must not read as a change.
    return {
        'return_type': sig.get_return_type().as_dict(),
        'parameters': {name: (p.col_type.as_dict(), p.kind.name) for name, p in sig.parameters.items()},
    }


def _signature_from_tool(mcp_tool: 'mcp.types.Tool') -> Signature:
    input_schema = mcp_tool.inputSchema
    params = {
        name: __mcp_param_to_pxt_type(mcp_tool.name, name, param) for name, param in input_schema['properties'].items()
    }
    required = input_schema.get('required', [])

    # Ensure that any params not appearing in `required` are nullable.
    # (A required param might or might not be nullable, since its type might be an 'anyOf' containing a null.)
    for name in params.keys() - required:
        params[name] = params[name].copy(nullable=True)

    return Signature(
        return_type=ts.StringType(),  # Return type is always string
        parameters=[Parameter(name, col_type, inspect.Parameter.KEYWORD_ONLY) for name, col_type in params.items()],
    )


def __mcp_param_to_pxt_type(tool_name: str, name: str, param: dict[str, Any]) -> ts.ColumnType:
    pxt_type = ts.ColumnType.from_json_schema(param)
    if pxt_type is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_TYPE,
            f'Unknown type schema for MCP parameter {name!r} of tool {tool_name!r}: {param}',
        )
    return pxt_type
