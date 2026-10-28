// Renders the pixeltable.com hero's table build as two animated SVGs for the GitHub README
// (light and dark, chosen with <picture>). GitHub shows README SVGs as images: CSS animations
// play, scripts and web fonts do not. Every element's base style is the finished frame, so
// reduced motion (animations off) shows it whole. Timeline and data mirror HeroTableBuild.tsx; the code lines are the README's app.py.
//
//   node build-table-svg.mjs        writes table-build-light.svg and table-build-dark.svg here

import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

const STEPS = [
  { at: 0, job: 'database', line: "class Videos(TableModel, name='videos'):" },
  { at: 2600, job: 'orchestration', line: "transcript = openai.transcriptions(audio, model='whisper-1')" },
  { at: 5400, job: 'orchestration', line: 'summary = openai.chat_completions(' },
  { at: 8200, job: 'serving', line: "api.add_query_route(path='/search', query=search)" },
];
const ROWS = [
  { video: 'keynote.mp4', transcript: "Today we're launching search over every…", summary: 'Product launch demo and pricing' },
  { video: 'standup.mp4', transcript: 'Yesterday I shipped the upload route…', summary: 'Team standup: upload route shipped' },
  { video: 'welcome.mp4', transcript: 'Welcome to the team. First, clone…', summary: 'New hire setup walkthrough' },
];

const ROW_GAP = 450;
const COMPUTE_LAG = 500;
const COMPUTE_TIME = 700;
const REQUEST_AT = STEPS[3].at + 700;
const ANSWER_AT = REQUEST_AT + 900;
const FADE_IN = 500;
const FADE_OUT = 700;
const LOOP = ANSWER_AT + 4200;
const END = LOOP - FADE_OUT;
const rowAt = (row) => 600 + row * ROW_GAP;
const fillAt = (step, row) => STEPS[step].at + COMPUTE_LAG + row * ROW_GAP;
const filledAt = (step) => fillAt(step, ROWS.length - 1) + COMPUTE_TIME;

const THEMES = {
  light: { text: '#1f2328', muted: '#59636e', faint: '#8c959f', border: 'rgba(31,35,40,0.15)', rule: 'rgba(31,35,40,0.08)', bar: '#f6f8fa', accent: '#d97706', route: '#1d4ed8', hit: 'rgba(31,35,40,0.05)' },
  dark: { text: '#e6edf3', muted: '#9198a1', faint: '#656c76', border: 'rgba(240,246,252,0.14)', rule: 'rgba(240,246,252,0.07)', bar: 'rgba(240,246,252,0.04)', accent: '#f1ae03', route: '#93c5fd', hit: 'rgba(240,246,252,0.05)' },
};

const W = 640;
const MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace";
const SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif";

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const clip = (s, n) => (s.length <= n ? s : `${s.slice(0, n - 1).trimEnd()}…`);

// One CSS animation per visibility window: opacity 0 → 1 at `from`, 1 → 0 at `to`, over `fade` ms.
const animations = [];
function shown(from, to, { fade = 400, finished = true } = {}) {
  const pct = (ms) => `${Math.min(100, Math.max(0, (ms / LOOP) * 100)).toFixed(3)}%`;
  const name = `a${animations.length}`;
  const stops = [
    ['0%', 0],
    [pct(from), 0],
    [pct(from + fade), 1],
    [pct(to), 1],
    [pct(to + fade), 0],
    ['100%', 0],
  ];
  animations.push(`@keyframes ${name}{${stops.map(([at, o]) => `${at}{opacity:${o}}`).join('')}}`);
  // The base opacity is the finished frame, which reduced motion keeps.
  return `class="${name}" style="opacity:${finished ? 1 : 0}"`;
}

function render(themeName) {
  animations.length = 0;
  const c = THEMES[themeName];
  const out = [];
  const text = (x, y, body, { size = 14, font = SANS, fill = c.text, anim = '', anchor = 'start', weight = 400 } = {}) =>
    `<text x="${x}" y="${y}" font-family="${font}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" ${anim}>${body}</text>`;

  // Line bar: the app.py line being declared.
  out.push(`<rect x="0.5" y="0.5" width="${W - 1}" height="43" rx="8" fill="${c.bar}" stroke="${c.border}"/>`);
  out.push(`<path d="M16 14h7l4 4v12h-11z M23 14v4h4" fill="none" stroke="${c.muted}" stroke-width="1.2" stroke-linejoin="round"/>`);
  out.push(text(36, 27, 'app.py', { font: MONO, fill: c.muted }));
  STEPS.forEach((step, i) => {
    const from = Math.max(step.at, FADE_IN);
    const to = i < STEPS.length - 1 ? STEPS[i + 1].at : END;
    const tail = step.line.endsWith('(') ? `<tspan fill="${c.muted}">…)</tspan>` : '';
    out.push(text(100, 27, esc(step.line) + tail, { font: MONO, anim: shown(from, to, { finished: i === STEPS.length - 1 }) }));
  });

  // The three jobs, each lit once its line lands. One text run lays them out, so the gaps hold in
  // any system font; each lit copy repeats the run with only its own job visible.
  const JOBS = ['database', 'orchestration', 'serving'];
  const jobsRun = (lit) =>
    JOBS.map((job, i) => `<tspan${i ? ' dx="18"' : ''} fill-opacity="${lit === null || lit === job ? 1 : 0}">● ${job}</tspan>`).join('');
  out.push(text(0, 66, jobsRun(null), { size: 12, fill: c.faint }));
  for (const job of JOBS) {
    const at = Math.max(STEPS.find((s) => s.job === job).at, FADE_IN);
    out.push(text(0, 66, jobsRun(job), { size: 12, anim: shown(at, END) }));
  }

  // The videos table.
  const top = 84;
  const COLS = [{ x: 0, w: 176 }, { x: 176, w: 232 }, { x: 408, w: 232 }];
  const HEAD = 36;
  const NAMES = 34;
  const ROW = 38;
  const height = HEAD + NAMES + ROWS.length * ROW;
  out.push(`<rect x="0.5" y="${top + 0.5}" width="${W - 1}" height="${height}" rx="12" fill="none" stroke="${c.border}"/>`);
  out.push(`<line x1="0.5" y1="${top + HEAD}" x2="${W - 0.5}" y2="${top + HEAD}" stroke="${c.border}"/>`);
  out.push(`<g ${shown(FADE_IN, END)}>${text(14, top + 23, 'videos', { font: MONO, fill: c.muted })}</g>`);
  ROWS.forEach((_, i) => {
    const n = i + 1;
    const from = i === 0 ? FADE_IN : rowAt(i);
    const to = i < ROWS.length - 1 ? rowAt(i + 1) : END;
    out.push(text(76, top + 23, `· ${n} ${n === 1 ? 'row' : 'rows'}`, { font: MONO, size: 12, fill: c.muted, anim: shown(from, to, { fade: 150, finished: n === ROWS.length }) }));
  });

  const namesY = top + HEAD + 22;
  out.push(text(14, namesY, 'video', { font: MONO, size: 12, fill: c.muted, anim: shown(FADE_IN, END) }));
  [1, 2].forEach((step) => {
    const name = step === 1 ? 'transcript' : 'summary';
    const x = COLS[step].x + 12;
    out.push(text(x, namesY, name, { font: MONO, size: 12, fill: c.muted, anim: shown(STEPS[step].at, filledAt(step), { finished: false }) }));
    out.push(text(x, namesY, name, { font: MONO, size: 12, fill: c.accent, anim: shown(filledAt(step), END) }));
  });

  ROWS.forEach((row, i) => {
    const y = top + HEAD + NAMES + i * ROW;
    out.push(`<line x1="0.5" y1="${y}" x2="${W - 0.5}" y2="${y}" stroke="${c.rule}"/>`);
    if (i === 0) out.push(`<rect x="1" y="${y + 0.5}" width="${W - 2}" height="${ROW - 1}" fill="${c.hit}" ${shown(ANSWER_AT, END)}/>`);
    const base = y + 24;
    out.push(`<g ${shown(rowAt(i), END)}><path d="M14 ${base - 11}h7l3 3v9h-10z" fill="none" stroke="${c.muted}" stroke-width="1.1" stroke-linejoin="round"/>${text(32, base, row.video)}</g>`);
    [[1, row.transcript, c.muted], [2, row.summary, c.text]].forEach(([step, value, fill]) => {
      const x = COLS[step].x + 12;
      const at = fillAt(step, i);
      out.push(`<rect x="${x}" y="${base - 7}" width="140" height="6" rx="3" fill="${c.accent}" ${shown(at, at + COMPUTE_TIME, { fade: 200, finished: false })}/>`);
      out.push(text(x, base, esc(clip(value, 28)), { fill, anim: shown(at + COMPUTE_TIME, END) }));
    });
  });

  // Serving: the route answers from the same table.
  const reqY = top + height + 30;
  out.push(`<g ${shown(REQUEST_AT, END)}>${text(0, reqY, 'POST /search', { font: MONO, fill: c.route })}${text(112, reqY, esc('{"query": "launch demo"}'), { font: MONO, fill: c.muted })}</g>`);
  out.push(text(W, reqY, '200 · keynote.mp4', { font: MONO, anchor: 'end', anim: shown(ANSWER_AT, END) }));

  const H = reqY + 8;
  const style = [
    ...animations,
    ...animations.map((_, i) => `.a${i}{animation:a${i} ${LOOP}ms linear infinite}`),
    '@media (prefers-reduced-motion:reduce){[class^="a"]{animation:none}}',
  ].join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1 -1 ${W + 2} ${H + 2}" width="${W + 2}" height="${H + 2}" role="img" aria-labelledby="t">
<title id="t">app.py declares a videos table line by line: the table is the database, its computed transcript and summary columns fill for every row, and a /search route serves the same table.</title>
<style>${style}</style>
${out.join('\n')}
</svg>
`;
}

for (const theme of ['light', 'dark']) {
  const file = join(HERE, `table-build-${theme}.svg`);
  writeFileSync(file, render(theme));
  console.log('wrote', file);
}
