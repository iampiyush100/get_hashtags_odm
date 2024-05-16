import express from "express"
import cors from "cors"
import multer from "multer"
import { v4 as uuidv4 } from "uuid"
import path from "path"
import fs from "fs"
import {exec} from "child_process" // watch out
import tf from '@tensorflow/tfjs-node'
import cocoSsd from '@tensorflow-models/coco-ssd'
import fsPromise from 'fs/promises'

const app = express()

//multer middleware

const storage = multer.diskStorage({
  destination: function(req, file, cb){
    cb(null, "./uploads")
  },
  filename: function(req, file, cb){
    cb(null, file.fieldname + "-" + uuidv4() + path.extname(file.originalname))
  }
})

// multer configuration
const upload = multer({storage: storage})


app.use(
  cors({
    origin: ["http://localhost:3000", "http://localhost:5173"],
    credentials: true
  })
)

app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*") // watch it
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept"
  );
  next()
})

app.use(express.json())
app.use(express.urlencoded({extended: true}))
app.use("/uploads", express.static("uploads"))

app.get('/', function(req, res){
  res.json({message: "Hello chai aur code"})
})

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


// Function to check if a file is an image file based on its extension
function isImageFile(filename) {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']; // Add more image extensions if needed
  const ext = path.extname(filename).toLowerCase();
  return imageExtensions.includes(ext);
}


// Main function to process frames and detect objects
async function processFrames(outputDir) {
  const frameFiles = await fsPromise.readdir(outputDir);
  const hashTags = []
  const hashTagsSet = new Set()
  const model = createModel();
  await model.loadModel();
  const predictionPromises = []
  for (const frameFile of frameFiles) {
    const framePath = path.join(outputDir, frameFile);
    // Check if the file is an image
    if (isImageFile(framePath)) {
      const imageBuffer = await fsPromise.readFile(framePath);
      predictionPromises.push(model.detectObjects(imageBuffer))
    }
  }
  const predictedObjectsResult = await Promise.allSettled(predictionPromises)
  const predictedObject = predictedObjectsResult.flatMap(({ value }) => value.map(obj => `#${obj.class}`))
  return [...new Set(predictedObject)]
}



function extractFrames(videoPath, outputDir) {
  return new Promise((resolve, reject) => {
    // Check if output directory exists, if not, create it
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // FFmpeg command to extract frames
    const command = `ffmpeg -i ${videoPath} -vf fps=24 ${outputDir}/frame_%04d.jpg`;

    // Execute FFmpeg command
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error('Error extracting frames:', error);
        reject(error);
      } else {
        console.log('Frames extracted successfully');
        resolve();
      }
    });
  });
}

function isVideoFile(filename) {
  const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']; // Add more video extensions if needed
  const ext = path.extname(filename).toLowerCase();
  return videoExtensions.includes(ext);
}

app.post("/upload", upload.single('file'), async (req, res) => {
  console.log(req.file);

  const videoPath = req.file.path; // Path to the input video
  const outputDir = 'output/frames'; // Output directory to save the frames
  try {
    if (isVideoFile(videoPath)) {
      await extractFrames(videoPath, outputDir); // Extract frames from the video
    }
    const prediction = await processFrames(outputDir);
    fs.rmSync(outputDir, { recursive: true, force: true }); //remove all frames
    fs.unlinkSync(videoPath); //remove uploaded video/image file
    return res.status(200).json({
      result: prediction
    })
  } catch (error) {
    console.error('Error:', error);
  }
})

app.listen(8000, function(){
  console.log("App is listening at port 8000...")
})