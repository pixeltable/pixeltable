import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Optional

import pixeltable as pxt
from pixeltable import exceptions as excs, type_system as ts
from pixeltable.func.signature import Parameter

if TYPE_CHECKING:
    import mcp


def mcp_udfs(url: str) -> list['pxt.func.Function']:
    return asyncio.run(mcp_udfs_async(url))


async def mcp_udfs_async(url: str) -> list['pxt.func.Function']:
    import mcp
    from mcp.client.streamable_http import streamablehttp_client

    list_tools_result: Optional[mcp.types.ListToolsResult] = None
    async with (
        streamablehttp_client(url) as (read_stream, write_stream, _),
        mcp.ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        list_tools_result = await session.list_tools()
    assert list_tools_result is not None

    return [mcp_tool_to_udf(url, tool) for tool in list_tools_result.tools]


def mcp_tool_to_udf(url: str, mcp_tool: 'mcp.types.Tool') -> 'pxt.func.Function':
    import mcp
    from mcp.client.streamable_http import streamablehttp_client

    async def invoke(**kwargs: Any) -> str:
        # TODO: Cache session objects rather than creating a new one each time?
        async with (
            streamablehttp_client(url) as (read_stream, write_stream, _),
            mcp.ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            res = await session.call_tool(name=mcp_tool.name, arguments=kwargs)
            # TODO Handle image/audio responses?
            return res.content[0].text  # type: ignore[union-attr]

    if mcp_tool.description is not None:
        invoke.__doc__ = mcp_tool.description

    input_schema = mcp_tool.inputSchema
    params = {
        name: __mcp_param_to_pxt_type(mcp_tool.name, name, param) for name, param in input_schema['properties'].items()
    }
    required = input_schema.get('required', [])

    # Ensure that any params not appearing in `required` are nullable.
    # (A required param might or might not be nullable, since its type might be an 'anyOf' containing a null.)
    for name in params.keys() - required:
        params[name] = params[name].copy(nullable=True)

    signature = pxt.func.Signature(
        return_type=ts.StringType(),  # Return type is always string
        parameters=[Parameter(name, col_type, inspect.Parameter.KEYWORD_ONLY) for name, col_type in params.items()],
    )

    return pxt.func.CallableFunction(signatures=[signature], py_fns=[invoke], self_name=mcp_tool.name)


def __mcp_param_to_pxt_type(tool_name: str, name: str, param: dict[str, Any]) -> ts.ColumnType:
    pxt_type = ts.ColumnType.from_json_schema(param)
    if pxt_type is None:
        raise excs.Error(f'Unknown type schema for MCP parameter {name!r} of tool {tool_name!r}: {param}')
    return pxt_type
