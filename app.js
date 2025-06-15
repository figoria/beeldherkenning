import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const startWebcamButton = document.getElementById("startWebcam");
const logPoseButton = document.getElementById("logPose");
const savePoseButton = document.getElementById("savePose");
const trainModelButton = document.getElementById("trainModel"); // extra knop om te trainen vanaf backend
const predictPoseButton = document.getElementById("predictPose");
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const predictionDisplay = document.getElementById("predictionDisplay");

let handLandmarker = null;
let webcamStream = null;
let isWebcamRunning = false;
let nn = null; // Neural Network instantie
let poses = []; // Tijdelijke poses opgeslagen in frontend

video.style.display = "none";

// Zet ml5 backend
ml5.setBackend("cpu");

// Neural network aanmaken
function createNeuralNetwork() {
    return ml5.neuralNetwork({
        task: 'classification',
        debug: true,
    });
}

async function loadHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
    });
    console.log("Hand Landmarker model geladen.");
}

function simplifyPose(landmarks) {
    return landmarks.map(point => [point.x, point.y, point.z]).flat();
}


async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = webcamStream;

        video.addEventListener("loadeddata", () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isWebcamRunning = true;
            predictHands();
        });

        video.play();
    } catch (error) {
        console.error("Webcam kan niet worden gestart:", error);
    }
}

async function predictHands() {
    if (!isWebcamRunning || !handLandmarker) return;

    const predictions = await handLandmarker.detectForVideo(video, performance.now());

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (predictions.landmarks) {
        predictions.landmarks.forEach((landmarks) => {
            const drawingUtils = new DrawingUtils(ctx);
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
            drawingUtils.drawLandmarks(landmarks, { radius: 5, color: "#FF0000" });
        });
    }

    if (isWebcamRunning) {
        requestAnimationFrame(predictHands);
    }
}

// Log pose in frontend geheugen
logPoseButton.addEventListener("click", async () => {
    if (!handLandmarker) return;

    const predictions = await handLandmarker.detectForVideo(video, performance.now());
    if (predictions.landmarks && predictions.landmarks.length > 0) {
        const label = prompt("Welk gebaar maak je? (bijv. 'A', 'B', 'C')");
        if (label) {
            const pose = simplifyPose(predictions.landmarks[0]);
            poses.push({ label, pose });
            console.log(`Pose "${label}" opgeslagen:`, pose);
        } else {
            console.log("Geen label opgegeven. Pose niet opgeslagen.");
        }
    } else {
        console.log("Geen hand gedetecteerd.");
    }
});

// Opslaan poses naar backend
async function savePosesToBackend(posesToSave) {
    try {
        const response = await fetch("http://localhost:3000/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(posesToSave),
        });
        if (!response.ok) throw new Error("Opslaan poses mislukt");
        console.log("Poses succesvol opgeslagen op de server.");
        predictionDisplay.textContent = "Poses opgeslagen op server!";
        poses = []; // reset na opslaan
    } catch (err) {
        console.error("Fout bij opslaan poses:", err);
        predictionDisplay.textContent = "Fout bij opslaan poses.";
    }
}

// SavePose knop nu opslaan naar backend
savePoseButton.addEventListener("click", async () => {
    if (poses.length === 0) {
        console.log("Geen poses om op te slaan.");
        predictionDisplay.textContent = "Geen poses om op te slaan.";
        return;
    }
    await savePosesToBackend(poses);
});

// Train model door data op te halen van backend en daarna trainen
async function trainModelFromBackendData() {
    try {
        const response = await fetch("http://localhost:3000/load");
        if (!response.ok) throw new Error("Geen pose data gevonden op server.");

        const serverPoses = await response.json();

        if (serverPoses.length === 0) {
            console.log("Geen poses op server om te trainen.");
            predictionDisplay.textContent = "Geen poses op server om te trainen.";
            return;
        }

        nn = createNeuralNetwork();

        serverPoses.forEach(({ pose, label }) => {
            nn.addData(pose, { label });
        });

        nn.normalizeData();
        nn.train({ epochs: 10 }, () => {
            console.log("Training voltooid.");
            predictionDisplay.textContent = "Model getraind met serverdata!";
        });
    } catch (err) {
        console.error("Fout bij trainen:", err);
        predictionDisplay.textContent = "Kan niet trainen: geen data beschikbaar.";
    }
}

// Extra knop om model te trainen vanaf serverdata
trainModelButton.addEventListener("click", trainModelFromBackendData);

// Predict pose
predictPoseButton.addEventListener("click", async () => {
    if (!nn) {
        console.log("Neural Network is niet getraind.");
        predictionDisplay.textContent = "Model is niet beschikbaar. Eerst trainen.";
        return;
    }

    const predictions = await handLandmarker.detectForVideo(video, performance.now());
    if (predictions.landmarks && predictions.landmarks.length > 0) {
        const pose = simplifyPose(predictions.landmarks[0]);

        try {
            const results = await nn.classify(pose);
            const { label, confidence } = results[0];
            console.log(`Voorspelling: ${label}, Zekerheid: ${(confidence * 100).toFixed(2)}%`);
            predictionDisplay.textContent = `Prediction: ${label} (Confidence: ${(confidence * 100).toFixed(2)}%)`;
        } catch (error) {
            console.error("Fout bij voorspelling:", error);
            predictionDisplay.textContent = "Fout bij voorspelling.";
        }
    } else {
        console.log("Geen hand gedetecteerd.");
        predictionDisplay.textContent = "Geen hand gedetecteerd.";
    }
});

// Start everything
startWebcamButton.addEventListener("click", () => {
    if (!handLandmarker) {
        loadHandLandmarker().then(startWebcam);
    } else {
        startWebcam();
    }
});
