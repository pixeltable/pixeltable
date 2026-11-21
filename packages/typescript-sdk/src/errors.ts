/** An error that Pixeltable reported, with its `ErrorCode` name, such as `NOT_FOUND` or `TYPE_MISMATCH`. */
export class PixeltableError extends Error {
  readonly errorCode: string;
  readonly retryable: boolean;

  constructor(errorCode: string, message: string, retryable = false) {
    super(message);
    this.name = 'PixeltableError';
    this.errorCode = errorCode;
    this.retryable = retryable;
  }
}
