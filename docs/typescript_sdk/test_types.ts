// Test file to verify TypeScript types compile correctly
import PixeltableClient from './pixeltable';
import type { PixeltableImage, PixeltableVideo, PixeltableAudio, PixeltableDocument } from './pixeltable';

// Test that we can instantiate the client
const client = new PixeltableClient('http://localhost:8000', 'api-key');

// Test some common operations with type checking
async function testPixeltableTypes() {
  // Create a table
  const table = await client.create_table();

  // Import data with all required parameters
  const importedTable = await client.import_pandas('pandas_table', null, null, null);

  // List tables with both parameters
  const tables = await client.list_tables('', true);

  // Get a table
  const retrievedTable = await client.get_table('my_table');

  // Use array
  const arr = await client.array([1, 2, 3]);

  console.log('All type checks passed!');
}

// Test media types
interface TestMediaTypes {
  img: PixeltableImage;
  vid: PixeltableVideo;
  aud: PixeltableAudio;
  doc: PixeltableDocument;
}

const testMedia: TestMediaTypes = {
  img: {
    data: 'base64...',
    width: 640,
    height: 480,
    mode: 'RGB'
  },
  vid: {
    data: 'base64...',
    duration: 10.5,
    fps: 30
  },
  aud: {
    data: 'base64...',
    duration: 60,
    sampleRate: 44100,
    channels: 2
  },
  doc: {
    data: 'base64...',
    text: 'Document content',
    mimeType: 'application/pdf'
  }
};

export { testPixeltableTypes, testMedia };
