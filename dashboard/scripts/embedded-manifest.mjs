import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

const html = readFileSync(new URL('../../pixeltable_cli/server/embedded/embedded.html', import.meta.url));
if (html.byteLength > 1024 * 1024) throw new Error('Embedded dashboard exceeds the 1 MiB build budget');
const manifest = {
  contractVersion: 1,
  sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  sourceDirty: execFileSync('git', ['status', '--porcelain', '--', 'dashboard'], { encoding: 'utf8' }).trim().length > 0,
  sha256: createHash('sha256').update(html).digest('hex'),
  bytes: html.byteLength,
};
writeFileSync(new URL('../../pixeltable_cli/server/embedded/manifest.json', import.meta.url), JSON.stringify(manifest, null, 2) + '\n');
console.log(`Embedded dashboard: ${manifest.bytes} bytes; SHA-256 ${manifest.sha256}`);
