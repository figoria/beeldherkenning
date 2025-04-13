const express = require("express");
const cors = require("cors");
const app = express();
const fs = require("fs");
const PORT = 3000;

app.use(cors());
app.use(express.json());

const dataFile = "handposes.json";

app.post("/save", (req, res) => {
    const poseData = req.body;
    if (!Array.isArray(poseData)) {
        return res.status(400).send("Invalid data format. Expected an array.");
    }

    let existingData = [];
    if (fs.existsSync(dataFile)) {
        existingData = JSON.parse(fs.readFileSync(dataFile, "utf8"));
    }

    existingData.push(...poseData);

    fs.writeFileSync(dataFile, JSON.stringify(existingData, null, 2));
    res.status(200).send("Poses saved successfully.");
});

app.get("/load", (req, res) => {
    if (fs.existsSync(dataFile)) {
        const data = fs.readFileSync(dataFile, "utf8");
        res.status(200).send(JSON.parse(data));
    } else {
        res.status(404).send("No poses found.");
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
