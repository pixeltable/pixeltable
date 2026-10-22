import type { App } from '@modelcontextprotocol/ext-apps';
import type { DashboardTransport } from '@/api/client';

const routes = {
  '/api/dirs': 'directories',
  '/api/dashboard/tables/meta': 'table_meta',
  '/api/dashboard/tables/data': 'table_data',
  '/api/dashboard/tables/pipeline': 'table_pipeline',
  '/api/dashboard/pipeline': 'catalog_pipeline',
  '/api/dashboard/search': 'catalog_search',
} as const;

type DashboardEndpoint = typeof routes[keyof typeof routes];
function endpointFor(path: string): DashboardEndpoint {
  for (const [route, endpoint] of Object.entries(routes)) {
    if (route === path) return endpoint;
  }
  throw new Error('This dashboard operation is unavailable in chat.');
}

export function createMcpDashboardTransport(app: Pick<App, 'callServerTool'>, catalog: string): DashboardTransport {
  const connection = /^pxt:\/\/[^:/]+:([^/]+)$/.exec(catalog);
  if (!connection) throw new Error('The connected catalog is invalid.');
  const db = connection[1];
  return async (path) => {
    if (!path.startsWith('/api/')) throw new Error('Dashboard requests must use the declared API.');
    const url = new URL(path, 'https://dashboard.invalid');
    const endpoint = endpointFor(url.pathname);
    const parameters: Record<string, string> = {};
    for (const [key, value] of url.searchParams) {
      if (key === 'catalogs' || Object.hasOwn(parameters, key)) throw new Error('Cross-catalog requests are unavailable in chat.');
      parameters[key] = value;
    }
    const target = parameters.path;
    if (target === catalog || target === 'local') parameters.path = '';
    else if (target?.startsWith(catalog + '/')) parameters.path = target.slice(catalog.length + 1);
    else if (target?.includes('://')) throw new Error('Choose an item in the connected catalog.');
    const result = await app.callServerTool({ name: 'read_dashboard', arguments: { db, endpoint, parameters } });
    if (result.isError) {
      const message = result.content?.find((item) => item.type === 'text');
      throw new Error(message?.type === 'text' ? message.text : 'The catalog could not be read. Retry.');
    }
    if (!result.structuredContent || typeof result.structuredContent !== 'object' || !('data' in result.structuredContent)) {
      throw new Error('The server does not provide dashboard data. Update the connection.');
    }
    return Response.json(result.structuredContent.data);
  };
}
