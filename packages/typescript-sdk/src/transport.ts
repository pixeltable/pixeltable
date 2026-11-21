/**
 * Reaches a Pixeltable proxy daemon: a local one over plain HTTP, or a hosted one through the authenticated TLS
 * tunnel that the Python client uses (`PXT/1.0 CONNECT org/db`, then HTTP/1.1 on the same connection).
 */

import { readFileSync } from 'node:fs';
import { connect as netConnect, type Socket } from 'node:net';
import { homedir } from 'node:os';
import { connect as tlsConnect } from 'node:tls';

import { PixeltableError } from './errors.ts';

const DEFAULT_CLOUD_HOST = 'pxt.run';
const DEFAULT_CLOUD_PORT = 443;
const TIMEOUT_MS = 300_000;

/** A parsed `pxt://org:db/dir/table` URI; org `local` names a local proxy. */
export interface PxtUri {
  readonly org: string;
  readonly db: string;
  readonly path: readonly string[];
}

export function parseUri(uri: string): PxtUri {
  const m = /^pxt:\/\/([A-Za-z0-9_-]+):([A-Za-z0-9_-]+)(?:\/(.*))?$/.exec(uri);
  if (m === null) {
    throw new PixeltableError('INVALID_ARGUMENT', `Not a pxt://org:db/path URI: '${uri}'`);
  }
  const path = (m[3] ?? '').split('/').filter((p) => p !== '');
  return { org: m[1]!, db: m[2]!, path };
}

export interface Transport {
  post(path: string, body: Uint8Array): Promise<Uint8Array>;
}

/** The transport for the database a URI names. */
export function transportFor(uri: PxtUri): Transport {
  if (uri.org === 'local') {
    return new LocalTransport(uri.db);
  }
  return new TunnelTransport(uri.org, uri.db);
}

class LocalTransport implements Transport {
  readonly #db: string;

  constructor(db: string) {
    this.#db = db;
  }

  async post(path: string, body: Uint8Array): Promise<Uint8Array> {
    const home = process.env['PIXELTABLE_HOME'] ?? `${homedir()}/.pixeltable`;
    const lockFile = `${home}/proxy_${this.#db.toLowerCase()}/port.lock`;
    let port: number;
    try {
      port = (JSON.parse(readFileSync(lockFile, 'utf8')) as { port: number }).port;
    } catch {
      throw new PixeltableError(
        'SERVICE_NOT_FOUND',
        `No local proxy is running for '${this.#db}'. Start it with: pxt localproxy start ${this.#db}`,
      );
    }
    const socket = netConnect({ host: '127.0.0.1', port });
    await connected(socket, 'connect');
    return httpPost(socket, `127.0.0.1:${port}`, path, body);
  }
}

class TunnelTransport implements Transport {
  readonly #org: string;
  readonly #db: string;

  constructor(org: string, db: string) {
    this.#org = org;
    this.#db = db;
  }

  async post(path: string, body: Uint8Array): Promise<Uint8Array> {
    const apiKey = process.env['PIXELTABLE_API_KEY'];
    if (apiKey === undefined || apiKey === '') {
      throw new PixeltableError(
        'MISSING_CREDENTIALS',
        `Set PIXELTABLE_API_KEY to reach pxt://${this.#org}:${this.#db}.`,
      );
    }
    const setting = process.env['PIXELTABLE_CLOUD_HOST'] ?? DEFAULT_CLOUD_HOST;
    const sep = setting.lastIndexOf(':');
    const domain = sep < 0 ? setting : setting.slice(0, sep);
    const port = sep < 0 ? DEFAULT_CLOUD_PORT : Number(setting.slice(sep + 1));
    const host = `${this.#org}-${this.#db}.${domain}`;

    const socket = tlsConnect({ host, port, servername: host });
    await connected(socket, 'secureConnect');
    socket.write(`PXT/1.0 CONNECT ${this.#org}/${this.#db}\r\nAuthorization: Bearer ${apiKey}\r\n\r\n`);
    const reader = new SocketReader(socket);
    const statusLine = (await reader.readHead()).split('\r\n')[0] ?? '';
    if (!statusLine.startsWith('PXT/1.0 200')) {
      socket.destroy();
      throw new PixeltableError(
        'DATABASE_UNAVAILABLE',
        `pxt://${this.#org}:${this.#db} refused the connection: ${statusLine}`,
      );
    }
    return httpPost(socket, host, path, body, reader);
  }
}

function connected(socket: Socket, event: 'connect' | 'secureConnect'): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.once('error', reject);
    socket.once(event, () => {
      socket.removeListener('error', reject);
      resolve();
    });
  });
}

/** Reads a socket as a byte stream: up to a blank line, then a given number of bytes. */
class SocketReader {
  #buffer = new Uint8Array(0);
  #waiter: (() => void) | undefined;
  #failure: Error | undefined;
  #ended = false;

  constructor(socket: Socket) {
    socket.setTimeout(TIMEOUT_MS, () => socket.destroy(new Error('timed out waiting for the daemon')));
    socket.on('data', (chunk) => {
      const next = new Uint8Array(this.#buffer.length + chunk.length);
      next.set(this.#buffer);
      next.set(chunk, this.#buffer.length);
      this.#buffer = next;
      this.#wake();
    });
    socket.on('error', (error) => {
      this.#failure = error;
      this.#wake();
    });
    socket.on('end', () => {
      this.#ended = true;
      this.#wake();
    });
  }

  #wake(): void {
    const waiter = this.#waiter;
    this.#waiter = undefined;
    waiter?.();
  }

  async #until(ready: () => boolean): Promise<void> {
    while (!ready()) {
      if (this.#failure !== undefined) {
        throw this.#failure;
      }
      if (this.#ended) {
        throw new Error('the connection closed before the response was complete');
      }
      await new Promise<void>((resolve) => {
        this.#waiter = resolve;
      });
    }
  }

  /** The text up to the next blank line, which it consumes. */
  async readHead(): Promise<string> {
    let end = -1;
    await this.#until(() => {
      end = indexOfBlankLine(this.#buffer);
      return end >= 0;
    });
    const head = new TextDecoder().decode(this.#buffer.subarray(0, end));
    this.#buffer = this.#buffer.subarray(end + 4);
    return head;
  }

  async readBytes(n: number): Promise<Uint8Array> {
    await this.#until(() => this.#buffer.length >= n);
    const bytes = this.#buffer.subarray(0, n);
    this.#buffer = this.#buffer.subarray(n);
    return bytes;
  }
}

function indexOfBlankLine(buffer: Uint8Array): number {
  for (let i = 0; i + 3 < buffer.length; i++) {
    if (buffer[i] === 13 && buffer[i + 1] === 10 && buffer[i + 2] === 13 && buffer[i + 3] === 10) {
      return i;
    }
  }
  return -1;
}

async function httpPost(
  socket: Socket,
  host: string,
  path: string,
  body: Uint8Array,
  reader: SocketReader = new SocketReader(socket),
): Promise<Uint8Array> {
  try {
    socket.write(
      `POST ${path} HTTP/1.1\r\nHost: ${host}\r\nContent-Type: application/octet-stream\r\n` +
        `Content-Length: ${body.length}\r\nConnection: close\r\n\r\n`,
    );
    socket.write(body);
    const head = await reader.readHead();
    const [statusLine, ...headerLines] = head.split('\r\n');
    const status = Number((statusLine ?? '').split(' ')[1]);
    const header = (name: string): string | undefined =>
      headerLines.find((l) => l.toLowerCase().startsWith(`${name}:`))?.slice(name.length + 1).trim();
    const length = Number(header('content-length'));
    if (!Number.isInteger(length)) {
      throw new Error(`the daemon answered without a Content-Length (${statusLine})`);
    }
    const response = await reader.readBytes(length);
    if (status !== 200) {
      throw new PixeltableError('INTERNAL_ERROR', `${statusLine}: ${new TextDecoder().decode(response)}`);
    }
    return response;
  } finally {
    socket.destroy();
  }
}
