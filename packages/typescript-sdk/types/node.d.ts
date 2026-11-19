// The few Node.js APIs the package uses, declared here so that type checking needs no @types/node.

declare const process: {
  readonly env: Readonly<Record<string, string | undefined>>;
  readonly argv: readonly string[];
};

declare const console: {
  log(...data: unknown[]): void;
  table(data: unknown): void;
};

declare class TextEncoder {
  encode(input: string): Uint8Array;
}

declare class TextDecoder {
  decode(input: Uint8Array): string;
}

declare module 'node:fs' {
  export function readFileSync(path: string, encoding: 'utf8'): string;
}

declare module 'node:os' {
  export function homedir(): string;
}

declare module 'node:url' {
  export function pathToFileURL(path: string): { readonly href: string };
  export function fileURLToPath(url: string): string;
}

declare module 'node:net' {
  export interface Socket {
    write(data: Uint8Array | string): boolean;
    end(): void;
    destroy(error?: Error): void;
    setTimeout(ms: number, callback?: () => void): void;
    on(event: 'data', listener: (chunk: Uint8Array) => void): this;
    on(event: 'error', listener: (error: Error) => void): this;
    on(event: 'end' | 'close' | 'timeout', listener: () => void): this;
    once(event: 'connect' | 'secureConnect', listener: () => void): this;
    once(event: 'error', listener: (error: Error) => void): this;
    removeListener(event: string, listener: (...args: never[]) => void): this;
  }
  export function connect(options: { host: string; port: number }): Socket;
}

declare module 'node:tls' {
  import type { Socket } from 'node:net';
  export function connect(options: { host: string; port: number; servername: string }): Socket;
}

declare module 'node:module' {
  export interface LoadResult {
    format: string;
    source?: string;
    shortCircuit?: boolean;
  }
  export function registerHooks(hooks: {
    load?: (url: string, context: unknown, nextLoad: (url: string, context: unknown) => LoadResult) => LoadResult;
  }): unknown;
  export function stripTypeScriptTypes(code: string, options?: { mode?: 'strip' | 'transform' }): string;
}

declare module 'node:test' {
  export function test(name: string, fn: () => void | Promise<void>): void;
}

declare module 'node:assert/strict' {
  const assert: {
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    equal(actual: unknown, expected: unknown, message?: string): void;
    throws(fn: () => unknown, expected?: RegExp): void;
    rejects(promise: Promise<unknown>, expected?: RegExp): Promise<void>;
  };
  export default assert;
}
