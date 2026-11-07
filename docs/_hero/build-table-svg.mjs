// Renders the README's images: the hero, app.py's videos table filling in, and the logo, each in a light and a
// dark file chosen with <picture>. <picture> follows the browser's color scheme, not the GitHub theme, and PyPI
// drops <source>, so either file can land on either page color; each one carries its own background.
// GitHub shows README SVGs as images: CSS animations play, scripts and web fonts do not.
// Every element's base style is the finished frame, so reduced motion (animations off) shows it whole.
// The code lines are the README's app.py, shortened to fit. The rows are illustrative, named after parts of
// docs/resources/10-minute tour of Pixeltable.mp3.
//
//   node build-table-svg.mjs        writes table-build-{light,dark}.svg and logo-{light,dark}.svg here

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

const STEPS = [
  { at: 0, job: 'database', line: "class Videos(TableModel, name='videos'):" },
  { at: 2000, job: 'orchestration', line: 'transcript = openai.transcriptions(', tail: '…)' },
  { at: 4000, job: 'orchestration', line: 'summary = openai.chat_completions(', tail: '…)' },
  { at: 6000, job: 'serving', line: "api.add_query_route(path='/search',", tail: ' …)' },
];
const ROWS = [
  { video: 'tour-1.mp4', transcript: 'First, we create a table with an image column', summary: 'A table of images' },
  { video: 'tour-2.mp4', transcript: 'Next, a computed column runs object detection', summary: 'Detection as a column' },
  { video: 'tour-3.mp4', transcript: 'Then we ask GPT-4o about each image', summary: 'GPT-4o on each image' },
];
const QUERY = 'object detection';
const ANSWER_ROW = 1;

const ROW_GAP = 350;
const COMPUTE_LAG = 500;
const COMPUTE_TIME = 600;
const REQUEST_AT = STEPS[3].at + 600;
const ANSWER_AT = REQUEST_AT + 800;
const FADE_IN = 500;
const FADE_OUT = 700;
// the finished frame holds for about half the loop, so a glance usually finds the table complete
const LOOP = ANSWER_AT + 8000;
const END = LOOP - FADE_OUT;
const rowAt = (row) => 600 + row * ROW_GAP;
const fillAt = (step, row) => STEPS[step].at + COMPUTE_LAG + row * ROW_GAP;
const filledAt = (step) => fillAt(step, ROWS.length - 1) + COMPUTE_TIME;

// surface matches GitHub's code blocks; every text color clears 4.5:1 on it
const THEMES = {
  light: { surface: '#f6f8fa', text: '#1f2328', muted: '#59636e', faint: '#656d76', border: 'rgba(31,35,40,0.15)', rule: 'rgba(31,35,40,0.08)', accent: '#b45309', route: '#1d4ed8', hit: 'rgba(31,35,40,0.05)' },
  dark: { surface: '#151b23', text: '#e6edf3', muted: '#9198a1', faint: '#7d8590', border: 'rgba(240,246,252,0.14)', rule: 'rgba(240,246,252,0.07)', accent: '#f1ae03', route: '#93c5fd', hit: 'rgba(240,246,252,0.05)' },
};
// GitHub's page background per theme, so a logo plate disappears when the file matches the page
const PAGE = { light: '#ffffff', dark: '#0d1117' };

// drawn at the 420px the README shows it at, so sizes are CSS pixels
const W = 420;
const MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace";
const SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif";

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
// cuts at a word boundary, since a cell holds about 20 characters of 13px text
function clip(s, n) {
  if (s.length <= n) return s;
  const head = s.slice(0, n - 1);
  const cut = s[n - 1] === ' ' ? head : head.slice(0, head.lastIndexOf(' '));
  return `${cut.trimEnd()}…`;
}

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
  const text = (x, y, body, { size = 13, font = SANS, fill = c.text, anim = '', anchor = 'start', weight = 400 } = {}) =>
    `<text x="${x}" y="${y}" font-family="${font}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" ${anim}>${body}</text>`;
  const rule = (y, stroke) => `<line x1="1" y1="${y}" x2="${W - 1}" y2="${y}" stroke="${stroke}"/>`;

  // Header: the app.py line being declared.
  out.push(`<path d="M13 13h6l4 4v10h-10z M19 13v4h4" fill="none" stroke="${c.muted}" stroke-width="1.2" stroke-linejoin="round"/>`);
  out.push(text(30, 25, 'app.py', { font: MONO, size: 12, fill: c.muted }));
  STEPS.forEach((step, i) => {
    const from = Math.max(step.at, FADE_IN);
    const to = i < STEPS.length - 1 ? STEPS[i + 1].at : END;
    const tail = step.tail ? `<tspan fill="${c.muted}">${step.tail}</tspan>` : '';
    out.push(text(88, 25, esc(step.line) + tail, { font: MONO, size: 12, anim: shown(from, to, { finished: i === STEPS.length - 1 }) }));
  });
  out.push(rule(40, c.border));

  // The three jobs, each lit once its line lands. One text run lays them out, so the gaps hold in
  // any system font; each lit copy repeats the run with only its own job visible.
  const JOBS = ['database', 'orchestration', 'serving'];
  const jobsRun = (lit) =>
    JOBS.map((job, i) => `<tspan${i ? ' dx="16"' : ''} fill-opacity="${lit === null || lit === job ? 1 : 0}">● ${job}</tspan>`).join('');
  out.push(text(12, 62, jobsRun(null), { size: 12, fill: c.faint }));
  for (const job of JOBS) {
    const at = Math.max(STEPS.find((s) => s.job === job).at, FADE_IN);
    out.push(text(12, 62, jobsRun(job), { size: 12, anim: shown(at, END) }));
  }

  // The videos table.
  const top = 76;
  const COLS = [{ x: 0, w: 104 }, { x: 104, w: 150 }, { x: 254, w: 166 }];
  const HEAD = 32;
  const NAMES = 28;
  const ROW = 36;
  const height = HEAD + NAMES + ROWS.length * ROW;
  out.push(rule(top, c.border));
  out.push(rule(top + HEAD, c.border));
  out.push(rule(top + height, c.border));
  out.push(`<g ${shown(FADE_IN, END)}>${text(12, top + 21, 'videos', { font: MONO, size: 12, fill: c.muted })}</g>`);
  ROWS.forEach((_, i) => {
    const n = i + 1;
    const from = i === 0 ? FADE_IN : rowAt(i);
    const to = i < ROWS.length - 1 ? rowAt(i + 1) : END;
    out.push(text(64, top + 21, `· ${n} ${n === 1 ? 'row' : 'rows'}`, { font: MONO, size: 12, fill: c.muted, anim: shown(from, to, { fade: 150, finished: n === ROWS.length }) }));
  });

  const namesY = top + HEAD + 19;
  out.push(text(12, namesY, 'video', { font: MONO, size: 12, fill: c.muted, anim: shown(FADE_IN, END) }));
  [1, 2].forEach((step) => {
    const name = step === 1 ? 'transcript' : 'summary';
    const x = COLS[step].x + 12;
    out.push(text(x, namesY, name, { font: MONO, size: 12, fill: c.muted, anim: shown(STEPS[step].at, filledAt(step), { finished: false }) }));
    out.push(text(x, namesY, name, { font: MONO, size: 12, fill: c.accent, anim: shown(filledAt(step), END) }));
  });

  ROWS.forEach((row, i) => {
    const y = top + HEAD + NAMES + i * ROW;
    out.push(rule(y, c.rule));
    if (i === ANSWER_ROW) out.push(`<rect x="1" y="${y + 0.5}" width="${W - 2}" height="${ROW - 1}" fill="${c.hit}" ${shown(ANSWER_AT, END)}/>`);
    const base = y + 23;
    out.push(`<g ${shown(rowAt(i), END)}><path d="M12 ${base - 10}h6l3 3v8h-9z" fill="none" stroke="${c.muted}" stroke-width="1.1" stroke-linejoin="round"/>${text(28, base, row.video)}</g>`);
    [[1, clip(row.transcript, 20), c.muted], [2, row.summary, c.text]].forEach(([step, value, fill]) => {
      const x = COLS[step].x + 12;
      const at = fillAt(step, i);
      out.push(`<rect x="${x}" y="${base - 7}" width="${COLS[step].w - 36}" height="6" rx="3" fill="${c.accent}" ${shown(at, at + COMPUTE_TIME, { fade: 200, finished: false })}/>`);
      out.push(text(x, base, esc(value), { fill, anim: shown(at + COMPUTE_TIME, END) }));
    });
  });

  // Serving: the route answers from the same table.
  const reqY = top + height + 26;
  const resY = reqY + 20;
  out.push(`<g ${shown(REQUEST_AT, END)}>${text(12, reqY, 'POST /search', { font: MONO, size: 12, fill: c.route })}${text(108, reqY, esc(`{"query": "${QUERY}"}`), { font: MONO, size: 12, fill: c.muted })}</g>`);
  out.push(text(12, resY, `200 · ${ROWS[ANSWER_ROW].video}`, { font: MONO, size: 12, anim: shown(ANSWER_AT, END) }));

  const H = resY + 14;
  const style = [
    ...animations,
    ...animations.map((_, i) => `.a${i}{animation:a${i} ${LOOP}ms linear infinite}`),
    '@media (prefers-reduced-motion:reduce){[class^="a"]{animation:none}}',
  ].join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="t">
<title id="t">app.py declares a videos table line by line. Three videos are inserted, the transcript and summary columns fill in for each row, and a POST to /search returns the matching video from the same table.</title>
<style>${style}</style>
<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="10" fill="${c.surface}" stroke="${c.border}"/>
${out.join('\n')}
</svg>
`;
}

// The docs site's logo, on a plate of its theme's page color; the docs site keeps using the plain files.
function logo(themeName) {
  const src = readFileSync(join(HERE, '..', 'release', '_logo', `pxt-${themeName}.svg`), 'utf8');
  const open = src.match(/<svg[^>]*>/)[0];
  const [w, h] = [/width="(\d+)"/, /height="(\d+)"/].map((re) => Number(open.match(re)[1]));
  const pad = Math.round(h * 0.15);
  const body = src.slice(src.indexOf(open) + open.length, src.lastIndexOf('</svg>'));
  return `<svg width="${w + 2 * pad}" height="${h + 2 * pad}" viewBox="0 0 ${w + 2 * pad} ${h + 2 * pad}" fill="none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" role="img" aria-label="Pixeltable">
<rect width="${w + 2 * pad}" height="${h + 2 * pad}" rx="${pad}" fill="${PAGE[themeName]}"/>
<g transform="translate(${pad} ${pad})">${body}</g>
</svg>
`;
}

for (const theme of ['light', 'dark']) {
  for (const [file, svg] of [[`table-build-${theme}.svg`, render(theme)], [`logo-${theme}.svg`, logo(theme)]]) {
    writeFileSync(join(HERE, file), svg);
    console.log('wrote', join(HERE, file));
  }
}
