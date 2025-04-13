import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from './knear.js';

const startWebcamButton = document.getElementById("startWebcam");
const logPoseButton = document.getElementById("logPose");
const savePoseButton = document.getElementById("savePose");
const predictPoseButton = document.getElementById("predictPose");
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const predictionDisplay = document.getElementById("predictionDisplay");

let handLandmarker = null;
let webcamStream = null;
let loggedPoses = [];
let isWebcamRunning = false;
let machine = new kNear(3);

video.style.display = "none";

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

// Start the webcam
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

// Predict hand positions and draw them
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

// Log a pose
logPoseButton.addEventListener("click", async () => {
    if (!handLandmarker) return;

    try {
        const predictions = await handLandmarker.detectForVideo(video, performance.now());
        if (predictions.landmarks && predictions.landmarks.length > 0) {
            const label = prompt("Welk gebaar maak je? (bijv. 'duim omhoog', 'vuist', 'wijsvinger omhoog')");
            if (label) {
                const pose = simplifyPose(predictions.landmarks[0]);
                loggedPoses.push({ label, coordinates: pose });
                console.log(`Gebaar "${label}" opgeslagen:`, pose);
            } else {
                console.log("Geen label opgegeven. Pose niet opgeslagen.");
            }
        } else {
            console.log("Geen hand gedetecteerd.");
        }
    } catch (error) {
        console.error("Fout bij het loggen van een pose:", error);
    }
});

// save a loged pose
savePoseButton.addEventListener("click", async () => {
    if (loggedPoses.length === 0) {
        console.log("Geen poses om op te slaan.");
        return;
    }
    try {
        const response = await fetch("http://localhost:3000/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(loggedPoses),
        });
        if (response.ok) {
            console.log("Poses succesvol opgeslagen.");
            loggedPoses = [];
        } else {
            console.log("Fout bij het opslaan van poses.");
        }
    } catch (error) {
        console.error("Error bij opslaan van poses:", error);
    }
});

// Load the saved data into KNN
async function loadKNNData() {
    try {
        const response = await fetch("http://localhost:3000/load");
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        const data = await response.json();
        console.log("KNN data geladen:", data);

        if (data.length > 0) {
            data.forEach((entry) => {
                if (entry.label && entry.coordinates) {
                    machine.learn(entry.coordinates, entry.label);
                }
            });
            console.log("KNN data succesvol toegevoegd aan classifier.");
        } else {
            console.log("Geen bestaande KNN data gevonden in JSON.");
        }
    } catch (error) {
        console.error("Fout bij het laden van KNN data:", error);
    }
}

// Predict pose with KNN
predictPoseButton.addEventListener("click", async () => {
    if (!machine || machine.training.length === 0) {
        console.log("Hand Landmarker of KNN data niet beschikbaar.");
        return;
    }

    const predictions = await handLandmarker.detectForVideo(video, performance.now());
    if (predictions.landmarks && predictions.landmarks.length > 0) {
        const pose = predictions.landmarks[0].map((point) => [point.x, point.y, point.z]).flat();
        const { label, avgDistance } = machine.classify(pose);

        const certainty = Math.max(0, Math.min(100, (1 / (avgDistance + 1)) * 100));

        console.log("Voorspelde pose:", label);
        console.log("Gemiddelde afstand van de KNN-buren:", avgDistance);
        console.log("Zekerheid:", certainty.toFixed(2) + "%");

        predictionDisplay.textContent = `Prediction: ${label} (Certainty: ${certainty.toFixed(2)}%)`;
    } else {
        console.log("Geen hand gedetecteerd.");
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

// Load the KNN data when starting the page
window.addEventListener("load", () => {
    loadKNNData(); // Zorg ervoor dat de data wordt geladen zodra de pagina wordt geladen
});
