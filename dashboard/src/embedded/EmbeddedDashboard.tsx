import { useState, useEffect } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { App as McpApp } from '@modelcontextprotocol/ext-apps';
import { connectionResultSchema } from './connection';
import App from '@/App';
import { createDashboardClient } from '@/api/client';
import { DashboardRuntimeContext, type DashboardRuntime } from '@/lib/dashboard-runtime';
import { createMemoryPreferences } from '@/lib/preferences';
import { createMcpDashboardTransport } from './transport';
import { createMcpMediaResolver } from './media';

export function EmbeddedDashboard() {
  const [runtime, setRuntime] = useState<DashboardRuntime | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [narrow, setNarrow] = useState(() => window.innerWidth < 640);
  const [expandView, setExpandView] = useState<(() => Promise<unknown>) | null>(null);
  const [expandError, setExpandError] = useState<string | null>(null);
  useEffect(() => {
    const query = window.matchMedia('(max-width: 639px)');
    const update = () => setNarrow(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  const [databaseChoice, setDatabaseChoice] = useState<{ databases: string[]; choose: (db: string) => void } | null>(null);
  useEffect(() => {
    const bridge = new McpApp({ name: 'Pixeltable Dashboard', version: '1.0.0' }, {}, { strict: true });
    let mounted = true;
    setExpandView(() => () => bridge.requestDisplayMode({ mode: 'fullscreen' }));
    const timeout = window.setTimeout(() => {
      if (mounted) setError('The catalog did not open. Retry or reopen Pixeltable from the conversation.');
    }, 20000);
    const receive = (content: unknown) => {
      if (!mounted) return;
      // Browse/search results carry the same connection inside their page.
      const parsed = connectionResultSchema.safeParse(content);
      if (!parsed.success) return;
      window.clearTimeout(timeout);
      setError(null);
      const { connection, databases } = parsed.data;
      if (!connection.db) {
        setDatabaseChoice({ databases: (databases ?? []).map((item) => item.db), choose: (db) => {
          setDatabaseChoice(null);
          bridge.callServerTool({ name: 'open_collection_explorer', arguments: { db } }).then(
            (result) => { if (mounted) { if (result.isError) setError('The database could not be opened. Retry the connection.'); else receive(result.structuredContent); } },
            () => { if (mounted) setError('The database could not be opened. Retry the connection.'); },
          );
        } });
        return;
      }
      const catalog = `pxt://${connection.organization}:${connection.db}`;
      const client = createDashboardClient(createMcpDashboardTransport(bridge, catalog));
      setRuntime((current) => current ?? {
        kind: 'embedded', catalog, client, preferences: createMemoryPreferences(),
        resolveMedia: createMcpMediaResolver(bridge, catalog),
        theme: bridge.getHostContext()?.theme === 'dark' ? 'dark' : 'light',
      });
    };
    bridge.ontoolresult = (result) => receive(result.structuredContent);
    bridge.onhostcontextchanged = (context) => {
      if (mounted && context.theme) setRuntime((current) => current?.kind === 'embedded' ? { ...current, theme: context.theme === 'dark' ? 'dark' : 'light' } : current);
    };
    bridge.connect().catch(() => {
      if (mounted) setError('The chat connection could not be opened. Retry or reopen Pixeltable.');
    });
    return () => { mounted = false; window.clearTimeout(timeout); void bridge.close(); };
  }, [attempt]);
  if (error) return <main className="p-6 space-y-4" role="alert"><p>{error}</p><button className="rounded-md border px-4 py-2" onClick={() => {
    setError(null); setRuntime(null); setDatabaseChoice(null); setAttempt((value) => value + 1);
  }}>Retry connection</button></main>;
  if (databaseChoice) return <main className="p-6 space-y-4"><h1 className="text-lg font-semibold">Choose a database</h1>{databaseChoice.databases.length
    ? databaseChoice.databases.map((db) => <button key={db} className="block rounded-md border px-4 py-2 hover:bg-accent" onClick={() => databaseChoice.choose(db)}>{db}</button>)
    : <p>No databases are available. Create one in Pixeltable Cloud, then reopen this app.</p>}</main>;
  if (!runtime) return <main className="p-6 text-sm text-muted-foreground" role="status">Connecting to your Pixeltable catalog…</main>;
  if (narrow) return <main className="p-6 space-y-4"><h1 className="text-lg font-semibold">Open your catalog in a larger view</h1><p className="text-sm text-muted-foreground">The table and lineage views need more space. Expand the conversation panel or open full screen.</p><button className="rounded-md border px-4 py-2 hover:bg-accent" onClick={() => { void expandView?.().catch(() => setExpandError('Expand the panel to at least 640 pixels, or open Pixeltable from the sidebar.')); }}>Open full screen</button>{expandError && <p className="text-sm" role="status">{expandError}</p>}</main>;
  return <DashboardRuntimeContext.Provider value={runtime}><MemoryRouter><App /></MemoryRouter></DashboardRuntimeContext.Provider>;
}
