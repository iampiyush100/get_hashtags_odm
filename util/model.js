import cocoSsd from '@tensorflow-models/coco-ssd';
import tf from '@tensorflow/tfjs-node';

async function loadDetectionModel() {
  const start = new Date();
  console.log('Loading model...');
  const model = await cocoSsd.load();
  console.log(`🤖 Model is loaded ${(new Date() - start)/1000}s`);
  return model;
}

async function detect(model, imageBuffer) {
  const arrByte = Uint8Array.from(imageBuffer);
  const decodedImage = await tf.node.decodeJpeg(arrByte);
  // make a prediction
  const prediction = await model.detect(decodedImage);
  return prediction || [];
}

function createModel() {
  let model;
  return {
    async loadModel() {
      model = await loadDetectionModel();
    },
    async detectObjects(imageBuffer) {
      if (!model) {
        throw new Error('Model not loaded');
      }
      return detect(model, imageBuffer);
    }
  };
}

module.exports = createModel;
