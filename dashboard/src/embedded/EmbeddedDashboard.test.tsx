import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { EmbeddedDashboard } from './EmbeddedDashboard';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

type ToolResult = { isError?: boolean; structuredContent?: unknown; content?: Array<{ type: string; text: string }> };
type Bridge = {
  ontoolresult?: (result: ToolResult) => void;
  onhostcontextchanged?: (context: { theme?: string }) => void;
  requestDisplayMode: ReturnType<typeof vi.fn>;
  callServerTool: ReturnType<typeof vi.fn>;
  connect: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
};

const { bridges } = vi.hoisted(() => ({ bridges: [] as Bridge[] }));

vi.mock('@modelcontextprotocol/ext-apps', () => ({
  App: class {
    ontoolresult?: Bridge['ontoolresult'];
    onhostcontextchanged?: Bridge['onhostcontextchanged'];
    requestDisplayMode = vi.fn(async () => ({ mode: 'fullscreen' as const }));
    callServerTool = vi.fn(async (request: { name: string }) => request.name === 'read_dashboard'
      ? { structuredContent: { data: { tree: { entries: [] } } } }
      : { isError: true, content: [{ type: 'text', text: 'The database could not be opened.' }] });
    getHostContext() { return { theme: 'light' as const }; }
    connect = vi.fn(async () => undefined);
    close = vi.fn(async () => undefined);
    constructor() { bridges.push(this); }
  },
}));

const mounted: Array<ReturnType<typeof createRoot>> = [];
afterEach(async () => {
  await act(async () => { mounted.splice(0).forEach((root) => root.unmount()); });
  bridges.splice(0);
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

function installViewport(width: number) {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: width });
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: width < 640 && query === '(max-width: 639px)',
    media: query,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
}

async function renderDashboard() {
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  mounted.push(root);
  await act(async () => { root.render(<EmbeddedDashboard />); });
  return bridges[0];
}

const catalog = { connection: { organization: 'demo', db: 'main' } };
const chooser = { connection: { organization: 'demo', db: null }, databases: [{ db: 'main' }, { db: 'other' }] };

it('keeps an opened catalog when a later result offers another database', async () => {
  installViewport(1024);
  const bridge = await renderDashboard();
  await act(async () => { bridge.ontoolresult?.({ structuredContent: catalog }); });
  expect(document.body.textContent).toContain('pxt://demo:main');
  await act(async () => { bridge.ontoolresult?.({ structuredContent: chooser }); });
  expect(document.body.textContent).toContain('pxt://demo:main');
  expect(document.body.textContent).not.toContain('Choose a database');
});

it('restores the database list when opening one fails or returns no connection', async () => {
  installViewport(1024);
  const bridge = await renderDashboard();
  await act(async () => { bridge.ontoolresult?.({ structuredContent: chooser }); });
  const button = [...document.querySelectorAll('button')].find((item) => item.textContent === 'main');
  await act(async () => { button?.click(); });
  expect(document.body.textContent).toContain('Choose a database');
  expect(document.body.textContent).toContain('The database could not be opened. Choose it again.');
  bridge.callServerTool.mockResolvedValueOnce({ structuredContent: { connection: { organization: 'demo' } } });
  await act(async () => { button?.click(); });
  expect(document.body.textContent).toContain('Choose a database');
  expect(document.body.textContent).not.toContain('Connecting to your Pixeltable catalog');
});

it('says when the host keeps the panel inline', async () => {
  installViewport(390);
  const bridge = await renderDashboard();
  bridge.requestDisplayMode.mockResolvedValueOnce({ mode: 'inline' });
  await act(async () => { bridge.ontoolresult?.({ structuredContent: catalog }); });
  const button = [...document.querySelectorAll('button')].find((item) => item.textContent === 'Open full screen');
  await act(async () => { button?.click(); });
  expect(document.body.textContent).toContain('Expand the panel to at least 640 pixels');
});
