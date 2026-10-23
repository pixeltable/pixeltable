import { useState, useEffect, useRef } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { App as McpApp } from '@modelcontextprotocol/ext-apps';
import { connectionResultSchema } from './connection';
import App from '@/App';
import { createDashboardClient } from '@/api/client';
import { DashboardRuntimeContext, type DashboardRuntime } from '@/lib/dashboard-runtime';
import { createMemoryPreferences } from '@/lib/preferences';
import { createMcpDashboardTransport } from './transport';
import { createMcpMediaResolver } from './media';

const expandFailure = 'Expand the panel to at least 640 pixels, or open Pixeltable from the sidebar.';
const openFailure = 'The database could not be opened. Choose it again.';
const openTimeoutFailure = 'The database did not open. Choose it again.';

type DatabaseChoice = {
  databases: string[];
  error: string | null;
  choose: (db: string) => void;
};

export function EmbeddedDashboard() {
  const [runtime, setRuntime] = useState<DashboardRuntime | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [narrow, setNarrow] = useState(() => window.innerWidth < 640);
  const [expandView, setExpandView] = useState<(() => Promise<void>) | null>(null);
  const [expandError, setExpandError] = useState<string | null>(null);
  const [databaseChoice, setDatabaseChoice] = useState<DatabaseChoice | null>(null);
  const opened = useRef(false);
  useEffect(() => {
    const query = window.matchMedia('(max-width: 639px)');
    const update = () => setNarrow(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  useEffect(() => {
    const bridge = new McpApp({ name: 'Pixeltable Dashboard', version: '1.0.0' }, {}, { strict: true });
    let mounted = true;
    let choosing = false;
    let requestId = 0;
    let openTimeout = 0;
    opened.current = false;
    setExpandView(() => async () => {
      const result = await bridge.requestDisplayMode({ mode: 'fullscreen' });
      if (!mounted) return;
      setExpandError(result.mode === 'fullscreen' ? null : expandFailure);
    });
    const initialTimeout = window.setTimeout(() => {
      if (mounted && !opened.current && !choosing) {
        setError('The catalog did not open. Retry or reopen Pixeltable from the conversation.');
      }
    }, 20000);
    const openCatalog = (organization: string, db: string) => {
      if (opened.current) return;
      opened.current = true;
      window.clearTimeout(initialTimeout);
      window.clearTimeout(openTimeout);
      setDatabaseChoice(null);
      setError(null);
      const catalog = `pxt://${organization}:${db}`;
      setRuntime({
        kind: 'embedded', catalog, client: createDashboardClient(createMcpDashboardTransport(bridge, catalog)),
        preferences: createMemoryPreferences(), resolveMedia: createMcpMediaResolver(bridge, catalog),
        theme: bridge.getHostContext()?.theme === 'dark' ? 'dark' : 'light',
      });
    };
    const showChooser = (databases: string[]) => {
      // A later chooser result must not cover a catalog that is already open, or replace a choice in progress.
      if (opened.current || choosing) return;
      choosing = true;
      window.clearTimeout(initialTimeout);
      setError(null);
      const choose = (db: string) => {
        const id = ++requestId;
        window.clearTimeout(openTimeout);
        setDatabaseChoice((current) => current ? { ...current, error: null } : current);
        openTimeout = window.setTimeout(() => {
          if (mounted && !opened.current && id === requestId) {
            setDatabaseChoice((current) => current ? { ...current, error: openTimeoutFailure } : current);
          }
        }, 20000);
        bridge.callServerTool({ name: 'open_collection_explorer', arguments: { db } }).then(
          (result) => {
            window.clearTimeout(openTimeout);
            if (!mounted || opened.current || id !== requestId) return;
            const parsed = connectionResultSchema.safeParse(result.structuredContent);
            if (result.isError || !parsed.success || !parsed.data.connection.db) {
              setDatabaseChoice((current) => current ? { ...current, error: openFailure } : current);
              return;
            }
            openCatalog(parsed.data.connection.organization, parsed.data.connection.db);
          },
          () => {
            window.clearTimeout(openTimeout);
            if (!mounted || opened.current || id !== requestId) return;
            setDatabaseChoice((current) => current ? { ...current, error: openFailure } : current);
          },
        );
      };
      setDatabaseChoice({ databases, error: null, choose });
    };
    const receive = (content: unknown) => {
      if (!mounted || opened.current) return;
      const parsed = connectionResultSchema.safeParse(content);
      if (!parsed.success) return;
      const { connection, databases } = parsed.data;
      if (!connection.db) {
        showChooser((databases ?? []).map((item) => item.db));
        return;
      }
      openCatalog(connection.organization, connection.db);
    };
    bridge.ontoolresult = (result) => receive(result.structuredContent);
    bridge.onhostcontextchanged = (context) => {
      if (mounted && context.theme) setRuntime((current) => current?.kind === 'embedded' ? { ...current, theme: context.theme === 'dark' ? 'dark' : 'light' } : current);
    };
    bridge.connect().catch(() => {
      if (mounted) setError('The chat connection could not be opened. Retry or reopen Pixeltable.');
    });
    return () => {
      mounted = false;
      window.clearTimeout(initialTimeout);
      window.clearTimeout(openTimeout);
      void bridge.close();
    };
  }, [attempt]);
  if (error) return <main className="p-6 space-y-4" role="alert"><p>{error}</p><button className="rounded-md border px-4 py-2" onClick={() => {
    setError(null); setRuntime(null); setDatabaseChoice(null); setAttempt((value) => value + 1);
  }}>Retry connection</button></main>;
  if (databaseChoice) return <main className="p-6 space-y-4"><h1 className="text-lg font-semibold">Choose a database</h1>{databaseChoice.error && <p className="text-sm" role="alert">{databaseChoice.error}</p>}{databaseChoice.databases.length
    ? databaseChoice.databases.map((db) => <button key={db} className="block rounded-md border px-4 py-2 hover:bg-accent" onClick={() => databaseChoice.choose(db)}>{db}</button>)
    : <p>No databases are available. Create one in Pixeltable Cloud, then reopen this app.</p>}</main>;
  if (!runtime) return <main className="p-6 text-sm text-muted-foreground" role="status">Connecting to your Pixeltable catalog…</main>;
  if (narrow) return <main className="p-6 space-y-4"><h1 className="text-lg font-semibold">Open your catalog in a larger view</h1><p className="text-sm text-muted-foreground">The table and lineage views need more space. Expand the conversation panel or open full screen.</p><button className="rounded-md border px-4 py-2 hover:bg-accent" onClick={() => { void expandView?.().catch(() => setExpandError(expandFailure)); }}>Open full screen</button>{expandError && <p className="text-sm" role="status">{expandError}</p>}</main>;
  return <DashboardRuntimeContext.Provider value={runtime}><MemoryRouter><App /></MemoryRouter></DashboardRuntimeContext.Provider>;
}
