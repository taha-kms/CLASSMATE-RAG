/**
 * Server-Sent Events over fetch.
 *
 * /ask is a POST, which EventSource cannot send, so the stream is read with
 * fetch and parsed here. Frames are separated by a blank line and can be split
 * across reads at any byte, so this keeps whatever follows the last blank line
 * until more arrives.
 */

export interface SseFrame {
  event: string;
  data: string;
}

export function splitFrames(buffer: string): { frames: SseFrame[]; rest: string } {
  const normalised = buffer.replace(/\r\n/g, '\n');
  const parts = normalised.split('\n\n');
  const rest = parts.pop() ?? '';

  const frames: SseFrame[] = [];
  for (const part of parts) {
    let event = 'message';
    const data: string[] = [];
    for (const line of part.split('\n')) {
      if (line.startsWith(':')) continue; // comment / keep-alive
      const colon = line.indexOf(':');
      const field = colon === -1 ? line : line.slice(0, colon);
      const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
      if (field === 'event') event = value;
      else if (field === 'data') data.push(value);
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return { frames, rest };
}

export async function* readFrames(body: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { frames, rest } = splitFrames(buffer);
      buffer = rest;
      yield* frames;
    }
    // Flush a final frame the server did not terminate with a blank line.
    buffer += decoder.decode();
    const { frames } = splitFrames(buffer + '\n\n');
    yield* frames;
  } finally {
    reader.releaseLock();
  }
}
