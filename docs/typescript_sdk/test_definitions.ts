// Simple test to verify the .d.ts type definitions are valid
/// <reference path="./pixeltable.d.ts" />

// Test media types
const testImage: PixeltableImage = {
  data: 'base64...',
  width: 640,
  height: 480,
  mode: 'RGB',
  mimeType: 'image/png'
};

const testVideo: PixeltableVideo = {
  data: 'base64...',
  duration: 10.5,
  width: 1920,
  height: 1080,
  fps: 30,
  mimeType: 'video/mp4'
};

const testAudio: PixeltableAudio = {
  data: 'base64...',
  duration: 60,
  sampleRate: 44100,
  channels: 2,
  mimeType: 'audio/mp3'
};

const testDoc: PixeltableDocument = {
  data: 'base64...',
  text: 'Document content',
  mimeType: 'application/pdf',
  pages: 10
};

// Test column type
const columnType: PixeltableColumnType = {
  _classname: 'ImageType',
  nullable: false,
  width: 640,
  height: 480
};

console.log('Type definitions are valid!');
