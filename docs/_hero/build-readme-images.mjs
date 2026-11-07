// Renders the README's images, each in a light and a dark file chosen with <picture>:
// - the hero, pixeltable.com's before/after slider (StackSlider.tsx in pixeltable-cloud) as an animated SVG, with
//   the README's app.py on the Pixeltable side
// - the logo, on a plate of the page color
// <picture> follows the browser's color scheme, not the GitHub theme, and PyPI drops <source>, so either file can
// land on either page color; each one carries its own background. GitHub shows README SVGs as images: CSS
// animations play, scripts and web fonts do not. Every element's base style is the resting frame, so reduced
// motion (animations off) and PyPI show both sides.
//
//   node build-readme-images.mjs        writes stack-slider-{light,dark}.svg and logo-{light,dark}.svg here

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

// the README's "Why we built it" table and the systems each step usually takes; keep in step with
// pixeltable-cloud's frontend/src/data/request-lifecycle.ts
const STEPS = [
  ['Store files and their results', ['bucket', 'sync script', 'metadata DB']],
  ['Split documents or sample frames', ['queue', 'preprocessing job']],
  ['Call models', ['queue', 'workers', 'status table']],
  ['Explore and query the results', ['download script', 'notebook', 'local copies']],
  ['Search the results', ['embed job', 'vector DB']],
  ['Serve the results', ['API service', 'metadata DB', 'vector DB']],
  ['Let an agent call tools', ['agent framework', 'tool logs']],
  ['Remember across sessions', ['chat cache', 'vector store']],
  ['Trace a running app', ['tracing SDK', 'log store']],
  ['Evaluate model output', ['log export', 'eval notebook']],
  ['Rerun what failed', ['dead-letter queue', 'retry script']],
  ['Send results elsewhere', ['ETL job', 'second database', 'bucket sync']],
  ['Analyze and audit', ['ETL job', 'warehouse']],
];
// each box on the usual side and the systems it stands for (StackSlider.tsx's SYSTEMS)
const SYSTEMS = [
  ['bucket', ['bucket']],
  ['metadata DB', ['metadata DB', 'status table']],
  ['queue + workers', ['queue', 'workers', 'dead-letter queue']],
  ['preprocess + embed jobs', ['preprocessing job', 'embed job']],
  ['vector DB', ['vector DB', 'vector store']],
  ['API service', ['API service']],
  ['agent framework', ['agent framework']],
  ['chat cache', ['chat cache']],
  ['tracing + log store', ['tracing SDK', 'log store', 'tool logs', 'log export']],
  ['notebooks + local copies', ['notebook', 'eval notebook', 'local copies']],
  ['ETL + warehouse', ['ETL job', 'second database', 'warehouse']],
  ['sync + retry scripts', ['sync script', 'download script', 'retry script', 'bucket sync']],
];
const jobSystems = (i) => [...new Set(STEPS[i][1].map((chip) => SYSTEMS.findIndex(([, chips]) => chips.includes(chip))))];

// the README's app.py from the class down, rewrapped to 54 columns
const CODE_LINES = `class Videos(TableModel, name='videos'):
    video: pxt.Video
    audio = extract_audio(video, format='mp3')
    transcript = openai.transcriptions(
        audio, model='whisper-1')
    summary = openai.chat_completions(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content':
            'Summarize this video in two sentences.'},
            {'role': 'user',
             'content': transcript.text},
        ],
    ).choices[0].message.content.astype(pxt.String)

    __indexes__ = [pxt.EmbeddingIndex(
        summary, embedding=embed)]

@pxt.query
def search(query: str):
    sim = Videos.summary.similarity(string=query)
    return (Videos.order_by(sim, asc=False).limit(3)
            .select(Videos.video, Videos.summary))

api = FastAPIRouter(name='video-api')
api.add_insert_route(
    Videos, path='/videos',
    uploadfile_inputs=[Videos.video],
    outputs=[Videos.summary])
api.add_query_route(path='/search', query=search)`.split('\n');
const ROWS = [
  ['tour-1.mp4', 'A table of images'],
  ['tour-2.mp4', 'Detection as a column'],
  ['tour-3.mp4', 'GPT-4o on each image'],
];
const CHIPS = ['EmbeddingIndex', 'add_query_route', '--otel', 'invoke_tools', 'response.usage', 'pxt errors', 'export_sql', 'destination=', 'history()'];

// pixeltable.com's tokens (globals.css) and the hero window's code colors
const THEMES = {
  light: {
    bg: '#fafaf9', card: '#ffffff', muted: '#f4f4f3', fg: '#070000', dim: '#595959', border: '#e0e0e0',
    accent: '#d97706', accentText: '#b45309', knob: '#ffffff',
    code: { plain: '#070000', comment: '#008000', str: '#a31515', kw: '#af00db', fn: '#795e26', type: '#267f99', num: '#098658' },
  },
  dark: {
    bg: '#070807', card: '#0e110e', muted: '#171b17', fg: '#f2f2f2', dim: '#a6a6a6', border: '#383838',
    accent: '#f1ae03', accentText: '#f1ae03', knob: '#020704',
    code: { plain: '#f2f2f2', comment: '#6a9955', str: '#ce9178', kw: '#c586c0', fn: '#dcdcaa', type: '#4ec9b0', num: '#b5cea8' },
  },
};

const MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace";
const SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif";
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Drawn at 920px, about the README's width on github.com, so sizes are close to CSS pixels. Stage coordinates
// start at PAD; the slider positions below are stage x.
const W = 920;
const PAD = 16;
const STAGE = { x: PAD, y: 44, w: W - 2 * PAD, h: 500 };
const CODE = { x: 0, w: 384, head: 32, line: 15, blank: 8 };
const JOBS = { x: 396, w: 236, head: 30, row: 27 };
JOBS.h = JOBS.head + JOBS.row * STEPS.length;
JOBS.y = Math.round((STAGE.h - JOBS.h) / 2);
const BOX = { x: 656, y: JOBS.y, w: STAGE.w - 656, h: JOBS.h };
const SYS = { x: 0, w: 172, h: 27, gap: 5 };
SYS.y = JOBS.y + Math.round((JOBS.h - (SYS.h * SYSTEMS.length + SYS.gap * (SYSTEMS.length - 1))) / 2);
const MID_Y = JOBS.y + JOBS.h / 2;
const APP = { w: 150, h: 60 };
APP.x = Math.round((JOBS.x + JOBS.w + STAGE.w) / 2 - APP.w / 2);
const REST = Math.round((JOBS.x + JOBS.w + BOX.x) / 2); // between the job list and the table box
const jobY = (i) => JOBS.y + JOBS.head + JOBS.row * i + JOBS.row / 2;
const sysY = (j) => SYS.y + (SYS.h + SYS.gap) * j + SYS.h / 2;
const curve = (x0, y0, x1, y1) => {
  const dx = (x1 - x0) * 0.5;
  return `M${x0} ${y0} C${x0 + dx} ${y0} ${x1 - dx} ${y1} ${x1} ${y1}`;
};

// StackSlider's sweep: rest, the Pixeltable side whole, the usual side whole, rest; here it loops
const PERIOD = 14;
const SWEEP = [[0, REST], [3, REST], [4, 0], [8, 0], [9, STAGE.w], [11, STAGE.w], [12, REST], [14, REST]];

// the hero window's tokenizer (hero-use-cases/stories.ts), plus def, class and return
function tokenize(line) {
  const out = [];
  const re = /(#.*$)|('[^']*'|"[^"]*")|\b(import|from|as|in|for|def|class|return)\b|([A-Za-z_]\w*)(?=\()|\b([A-Z]\w*)\b|(\b\d+\b)/g;
  let last = 0;
  let m;
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) out.push([line.slice(last, m.index), 'plain']);
    out.push([m[0], m[1] ? 'comment' : m[2] ? 'str' : m[3] ? 'kw' : m[4] ? 'fn' : m[5] ? 'type' : 'num']);
    last = m.index + m[0].length;
  }
  if (last < line.length) out.push([line.slice(last), 'plain']);
  return out;
}

function render(themeName) {
  const c = THEMES[themeName];
  const text = (x, y, body, { size = 12, font = SANS, fill = c.fg, weight = 400, anchor = 'start', extra = '' } = {}) =>
    `<text x="${x}" y="${y}" font-family="${font}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" ${extra}>${body}</text>`;
  const card = (x, y, w, h, rx = 10) => `<rect x="${x + 0.5}" y="${y + 0.5}" width="${w - 1}" height="${h - 1}" rx="${rx}" fill="${c.card}" stroke="${c.border}"/>`;
  const header = (x, y, w, h) =>
    `<path d="M${x + 0.5} ${y + h} V${y + 10.5} a10 10 0 0 1 10 -10 H${x + w - 10.5} a10 10 0 0 1 10 10 V${y + h} Z" fill="${c.muted}"/><line x1="${x + 1}" y1="${y + h}" x2="${x + w - 1}" y2="${y + h}" stroke="${c.border}"/>`;
  const wire = (d) => `<path d="${d}" fill="none" stroke="${c.dim}" stroke-opacity="0.45"/>`;

  // The usual side: the systems, wired to the jobs that use them, and the app that talks to each.
  const usual = [];
  STEPS.forEach((_, i) => jobSystems(i).forEach((j) => usual.push(wire(curve(JOBS.x, jobY(i), SYS.x + SYS.w, sysY(j))))));
  usual.push(wire(`M${JOBS.x + JOBS.w} ${MID_Y} H${APP.x}`));
  SYSTEMS.forEach(([label], j) => {
    const y = sysY(j) - SYS.h / 2;
    usual.push(`<rect x="${SYS.x + 0.5}" y="${y + 0.5}" width="${SYS.w - 1}" height="${SYS.h - 1}" rx="6" fill="${c.card}" stroke="${c.border}"/>`);
    usual.push(text(SYS.x + 12, y + 18, esc(label), { fill: c.dim }));
  });
  usual.push(card(APP.x, MID_Y - APP.h / 2, APP.w, APP.h));
  usual.push(text(APP.x + APP.w / 2, MID_Y - 3, 'Your app', { size: 13, weight: 600, anchor: 'middle' }));
  usual.push(text(APP.x + APP.w / 2, MID_Y + 15, 'talks to each system', { size: 11.5, fill: c.dim, anchor: 'middle' }));

  // The Pixeltable side: app.py, wired to the jobs, and the table box the jobs land in.
  const px = [`<rect x="0" y="0" width="${STAGE.w}" height="${STAGE.h}" fill="${c.bg}"/>`];
  px.push(wire(`M${CODE.x + CODE.w} ${MID_Y} H${JOBS.x}`));
  STEPS.forEach((_, i) => px.push(wire(`M${JOBS.x + JOBS.w} ${jobY(i)} H${BOX.x}`)));
  px.push(card(CODE.x, 0, CODE.w, STAGE.h));
  px.push(header(CODE.x, 0, CODE.w, CODE.head));
  px.push(`<path d="M14 9.5h6.5l4 4V23h-10.5z M20.5 9.5v4h4 M17.5 16.5l-1.5 1.5 1.5 1.5 M22 16.5l1.5 1.5-1.5 1.5" fill="none" stroke="${c.accent}" stroke-width="1.3" stroke-linejoin="round" stroke-linecap="round"/>`);
  px.push(text(32, 21, 'app.py', { size: 13, font: MONO, weight: 600 }));
  let y = CODE.head + 8;
  for (const line of CODE_LINES) {
    if (line.trim() === '') {
      y += CODE.blank;
      continue;
    }
    y += CODE.line;
    const spans = tokenize(line).map(([s, k]) => `<tspan fill="${c.code[k]}">${esc(s)}</tspan>`).join('');
    px.push(text(12, y - 3, spans, { size: 11, font: MONO, extra: 'xml:space="preserve"' }));
  }
  px.push(card(BOX.x, BOX.y, BOX.w, BOX.h));
  px.push(header(BOX.x, BOX.y, BOX.w, 32));
  const bx = BOX.x + 12;
  px.push(`<g fill="none" stroke="${c.accent}" stroke-width="1.3"><ellipse cx="${bx + 6}" cy="${BOX.y + 11.5}" rx="5.5" ry="2.2"/><path d="M${bx + 0.5} ${BOX.y + 11.5}v9c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2v-9 M${bx + 0.5} ${BOX.y + 16}c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2"/></g>`);
  px.push(text(bx + 20, BOX.y + 21, 'videos', { size: 13, font: MONO, weight: 600 }));
  px.push(text(BOX.x + BOX.w - 12, BOX.y + 21, '3 rows', { size: 11, font: MONO, fill: c.dim, anchor: 'end' }));
  const ty = BOX.y + 32;
  const fmark = `<tspan fill="${c.accentText}" font-family="Georgia,serif" font-style="italic"> ƒ</tspan>`;
  px.push(text(bx, ty + 22, 'video', { size: 11, font: MONO, fill: c.dim }));
  px.push(text(bx + 80, ty + 22, `summary${fmark}`, { size: 11, font: MONO, fill: c.dim }));
  px.push(`<rect x="${bx - 4.5}" y="${ty + 30.5}" width="${BOX.w - 15}" height="${ROWS.length * 30 - 1}" rx="7" fill="none" stroke="${c.border}"/>`);
  ROWS.forEach(([video, summary], i) => {
    const ry = ty + 31 + i * 30;
    if (i) px.push(`<line x1="${bx - 4}" y1="${ry}" x2="${BOX.x + BOX.w - 8}" y2="${ry}" stroke="${c.border}" stroke-opacity="0.6"/>`);
    px.push(text(bx + 4, ry + 19, video, { fill: c.fg }));
    px.push(text(bx + 80, ry + 19, esc(summary), { fill: c.fg }));
  });
  // the chips, wrapped into rows from the bottom of the box up
  const chipW = (s) => Math.ceil(s.length * 6.6) + 14;
  const lines = [[]];
  let used = 0;
  for (const chip of CHIPS) {
    const w = chipW(chip);
    if (used && used + 6 + w > BOX.w - 24) {
      lines.push([]);
      used = 0;
    }
    lines.at(-1).push(chip);
    used += (used ? 6 : 0) + w;
  }
  lines.forEach((row, r) => {
    let x = bx;
    const cy = BOX.y + BOX.h - 12 - (lines.length - r) * 30;
    for (const chip of row) {
      const w = chipW(chip);
      px.push(`<rect x="${x + 0.5}" y="${cy + 0.5}" width="${w - 1}" height="23" rx="6" fill="${c.muted}" stroke="${c.border}"/>`);
      px.push(text(x + 7, cy + 16, esc(chip), { size: 11, font: MONO, fill: c.dim }));
      x += w + 6;
    }
  });

  // The jobs, above both sides.
  const jobs = [card(JOBS.x, JOBS.y, JOBS.w, JOBS.h), header(JOBS.x, JOBS.y, JOBS.w, JOBS.head)];
  jobs.push(text(JOBS.x + 12, JOBS.y + 20, 'What a request needs', { size: 13, weight: 600 }));
  STEPS.forEach(([need], i) => {
    const ry = JOBS.y + JOBS.head + JOBS.row * i;
    if (i) jobs.push(`<line x1="${JOBS.x + 1}" y1="${ry}" x2="${JOBS.x + JOBS.w - 1}" y2="${ry}" stroke="${c.border}" stroke-opacity="0.5"/>`);
    jobs.push(text(JOBS.x + 12, ry + 18, String(i + 1).padStart(2, '0'), { size: 11, font: MONO, fill: c.dim }));
    jobs.push(text(JOBS.x + 34, ry + 18, esc(need), { fill: c.dim }));
  });

  // The handle, drawn at stage x = 0 and moved by the sweep.
  const handle = `<g class="handle"><line x1="0" y1="0" x2="0" y2="${STAGE.h}" stroke="${c.accent}" stroke-width="2"/><rect x="-14" y="${MID_Y - 18}" width="28" height="36" rx="6" fill="${c.accent}"/><path d="M-3 ${MID_Y - 4}l-4 4 4 4 M3 ${MID_Y - 4}l4 4-4 4" fill="none" stroke="${c.knob}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></g>`;

  const pct = (t) => `${((t / PERIOD) * 100).toFixed(2)}%`;
  const ease = 'animation-timing-function:cubic-bezier(.65,0,.35,1)';
  const frames = (prop) => SWEEP.map(([t, x]) => `${pct(t)}{${prop(x)};${ease}}`).join('');
  const clip = (x) => `clip-path:inset(0 0 0 ${x}px)`;
  const move = (x) => `transform:translateX(${x}px)`;
  const style = [
    `@keyframes clip{${frames(clip)}}`,
    `@keyframes move{${frames(move)}}`,
    `.pixeltable{${clip(REST)};animation:clip ${PERIOD}s infinite}`,
    `.handle{${move(REST)};animation:move ${PERIOD}s infinite}`,
    '@media (prefers-reduced-motion:reduce){.pixeltable,.handle{animation:none}}',
  ].join('');

  // Labels above, captions below.
  const capY = STAGE.y + STAGE.h + 16;
  const colX = PAD + STAGE.w / 2;
  const H = capY + 64;
  const chevron = (x, dir) => `<path d="M${x} 21l${dir * -4} 4 ${dir * 4} 4" fill="none" stroke="${c.dim}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" transform="translate(0 -4)"/>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="t">
<title id="t">Usually, the 13 jobs of a request run across 12 systems that your app talks to one by one. With Pixeltable, the same 13 jobs live in one app.py, with its videos table and the APIs the other jobs use. A handle sweeps between the two sides.</title>
<style>${style}</style>
<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="12" fill="${c.bg}" stroke="${c.border}"/>
${chevron(PAD + 8, 1)}${text(PAD + 16, 26, 'Usually', { size: 13, weight: 600, fill: c.dim })}
${text(W - PAD - 16, 26, 'With Pixeltable', { size: 13, weight: 600, fill: c.dim, anchor: 'end' })}${chevron(W - PAD - 8, -1)}
<g transform="translate(${STAGE.x} ${STAGE.y})">
<g>${usual.join('')}</g>
<g class="pixeltable">${px.join('')}</g>
<g>${jobs.join('')}</g>
${handle}
</g>
<line x1="${PAD}" y1="${capY}" x2="${W - PAD}" y2="${capY}" stroke="${c.border}"/>
${text(PAD + 4, capY + 22, 'Usually', { size: 13, weight: 600 })}
${text(PAD + 4, capY + 42, `13 jobs across 12 systems,<tspan fill="${c.dim}"> and the scripts that keep them in sync</tspan>`, { size: 13 })}
${text(colX, capY + 22, 'With Pixeltable', { size: 13, weight: 600 })}
${text(colX, capY + 42, `13 jobs in one app.py,<tspan fill="${c.dim}"> as tables, computed columns, indexes and routes</tspan>`, { size: 13 })}
</svg>
`;
}

// The docs site's logo, on a plate of its theme's page color; the docs site keeps using the plain files.
const PAGE = { light: '#ffffff', dark: '#0d1117' };
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
  for (const [file, svg] of [[`stack-slider-${theme}.svg`, render(theme)], [`logo-${theme}.svg`, logo(theme)]]) {
    writeFileSync(join(HERE, file), svg);
    console.log('wrote', join(HERE, file));
  }
}
