/**
 * The proxy daemon's /rpc protocol, as the Python client speaks it: a framed body whose JSON head is a ProxyRequest
 * naming a catalog, table or query method and its arguments.
 */

import { PixeltableError } from './errors.ts';
import type { Transport } from './transport.ts';

/** pixeltable/service/proxy_protocol.py PROTOCOL_VERSION; the daemon refuses any other. */
export const PROTOCOL_VERSION = 10;

export interface RpcRequest {
  readonly class_name: string;
  readonly method: string;
  readonly args: Record<string, unknown>;
  readonly path_key?: unknown;
  readonly snapshot_path_key?: unknown;
}

export interface RpcResponse {
  readonly result?: unknown;
  readonly error?: Record<string, unknown> | null;
  readonly current_md?: unknown[] | null;
  readonly is_stale_md?: boolean;
  readonly prohibited_write?: boolean;
}

export async function call(transport: Transport, request: RpcRequest): Promise<RpcResponse> {
  const head = new TextEncoder().encode(
    JSON.stringify({ protocol_version: PROTOCOL_VERSION, catalog_frozen: false, ...request }),
  );
  const response = decodeBody(await transport.post('/rpc', encodeBody(head)));
  const error = response.error;
  if (error !== null && error !== undefined) {
    const code = String(error['error_code'] ?? 'INTERNAL_ERROR');
    const server = error['server_protocol_version'];
    const message =
      typeof server === 'number' && server !== PROTOCOL_VERSION
        ? `The daemon speaks protocol version ${server}; this SDK speaks ${PROTOCOL_VERSION}.`
        : String(error['message'] ?? 'unknown error');
    throw new PixeltableError(code, message, error['retryable'] === true);
  }
  if (response.prohibited_write === true) {
    throw new PixeltableError('INTERNAL_ERROR', 'the daemon refused a write');
  }
  return response;
}

/** u32 head length, head, u32 part count, then each part with its u32 length; big-endian. No parts are sent. */
function encodeBody(head: Uint8Array): Uint8Array {
  const out = new Uint8Array(4 + head.length + 4);
  const view = new DataView(out.buffer);
  view.setUint32(0, head.length);
  out.set(head, 4);
  view.setUint32(4 + head.length, 0);
  return out;
}

function decodeBody(body: Uint8Array): RpcResponse {
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  const headLength = view.getUint32(0);
  const head = new TextDecoder().decode(body.subarray(4, 4 + headLength));
  const nParts = view.getUint32(4 + headLength);
  if (nParts !== 0) {
    // media and arrays come back as binary parts, which this prototype does not read
    throw new PixeltableError('UNSUPPORTED_OPERATION', 'the response carries binary values, which are not supported');
  }
  return JSON.parse(head) as RpcResponse;
}

/** A value as the protocol carries it: plain JSON, with tags for what JSON cannot represent. */
export function encodeValue(value: unknown): unknown {
  if (value instanceof Date) {
    return { $pxt: 'datetime', v: value.toISOString() };
  }
  if (typeof value === 'number' && !Number.isFinite(value)) {
    return { $pxt: 'float', v: Number.isNaN(value) ? 'nan' : value > 0 ? 'inf' : '-inf' };
  }
  if (typeof value === 'bigint') {
    return Number(value);
  }
  if (Array.isArray(value)) {
    return value.map(encodeValue);
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, encodeValue(v)]));
  }
  return value;
}

export function decodeValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(decodeValue);
  }
  if (value === null || typeof value !== 'object') {
    return value;
  }
  const tagged = value as Record<string, unknown>;
  switch (tagged['$pxt']) {
    case undefined:
      return Object.fromEntries(Object.entries(tagged).map(([k, v]) => [k, decodeValue(v)]));
    case 'datetime':
      return new Date(String(tagged['v']));
    case 'date':
    case 'UUID':
      return tagged['v'];
    case 'float': {
      const v = String(tagged['v']);
      return v === 'inf' ? Infinity : v === '-inf' ? -Infinity : Number.NaN;
    }
    case 'tuple':
      return decodeValue(tagged['v']);
    case 'rawdict':
      return Object.fromEntries((tagged['v'] as [unknown, unknown][]).map(([k, v]) => [String(k), decodeValue(v)]));
    default:
      return tagged;
  }
}
