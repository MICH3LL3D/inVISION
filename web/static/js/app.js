/**
 * app.js - Main controller that wires up the step-by-step UI.
 */
(function () {
    // --- DOM refs ---
    const panels = document.querySelectorAll(".step-panel");
    const indicators = document.querySelectorAll(".step-indicator");
    let currentStep = 1;

    // Step 1
    const cameraFeed = document.getElementById("camera-feed");
    const phoneCanvas = document.getElementById("phone-canvas");
    const phoneCtx = phoneCanvas.getContext("2d");
    const cameraOverlay = document.getElementById("camera-overlay");
    const btnStartCamera = document.getElementById("btn-start-camera");
    const btnPhoneCamera = document.getElementById("btn-phone-camera");
    const btnSwitchCamera = document.getElementById("btn-switch-camera");
    const btnToDetect = document.getElementById("btn-to-detect");
    const btnTakePhoto = document.getElementById("btn-take-photo");
    const btnUploadPhoto = document.getElementById("btn-upload-photo");
    const fileUpload = document.getElementById("file-upload");
    const phoneStatus = document.getElementById("phone-status");

    // Step 2
    const detectFeed = document.getElementById("detect-feed");
    const detectCanvas = document.getElementById("detect-canvas");
    const detectedLabel = document.getElementById("detected-label");
    const btnCapture = document.getElementById("btn-capture");
    const capturedPreview = document.getElementById("captured-preview");
    const captureCanvas = document.getElementById("capture-canvas");
    const captureLabel = document.getElementById("capture-label");
    const btnToModel = document.getElementById("btn-to-model");
    const btnBackToCamera = document.getElementById("btn-back-to-camera");

    // Step 3
    const loadingSpinner = document.getElementById("loading-spinner");
    const progressFill = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");
    const btnGenerate = document.getElementById("btn-generate");
    const btnRegenerate = document.getElementById("btn-regenerate");
    const btnToInteract = document.getElementById("btn-to-interact");
    const btnBackToDetect = document.getElementById("btn-back-to-detect");

    // Step 4
    const threeCanvas = document.getElementById("three-canvas");
    const handFeed = document.getElementById("hand-feed");
    const handOverlay = document.getElementById("hand-overlay");
    const hudHands = document.getElementById("hud-hands");
    const hudScale = document.getElementById("hud-scale");
    const hudMode = document.getElementById("hud-mode");
    const btnStartHands = document.getElementById("btn-start-hands");
    const btnRecenter = document.getElementById("btn-recenter");
    const btnToggleCam = document.getElementById("btn-toggle-cam");
    const btnBackToModel = document.getElementById("btn-back-to-model");
    const btnResetAll = document.getElementById("btn-reset-all");

    // --- State ---
    let capturedData = null;

    // --- Step navigation ---
    function goToStep(step) {
        panels.forEach((p) => p.classList.remove("active"));
        indicators.forEach((ind) => {
            const s = parseInt(ind.dataset.step);
            ind.classList.remove("active");
            if (s < step) ind.classList.add("completed");
            else ind.classList.remove("completed");
            if (s === step) ind.classList.add("active");
        });
        document.getElementById(`step-${step}`).classList.add("active");
        currentStep = step;
    }

    function activateCamera() {
        cameraOverlay.classList.add("hidden");
        btnToDetect.disabled = false;
        btnTakePhoto.disabled = false;
        btnStartCamera.disabled = true;
        btnPhoneCamera.disabled = true;
    }

    // --- Step 1: Laptop Camera ---
    btnStartCamera.addEventListener("click", async () => {
        try {
            const stream = await Camera.start();
            cameraFeed.srcObject = stream;
            cameraFeed.style.display = "";
            phoneCanvas.style.display = "none";
            activateCamera();
            btnSwitchCamera.disabled = false;
            btnStartCamera.textContent = "Laptop Active";
        } catch (err) {
            alert("Could not access camera: " + err.message);
        }
    });

    // --- Step 1: Phone Camera ---
    btnPhoneCamera.addEventListener("click", () => {
        Camera.startPhoneStream();
        cameraFeed.style.display = "none";
        phoneCanvas.style.display = "block";
        phoneStatus.classList.remove("hidden");

        const img = new Image();

        Camera.onPhoneFrame((dataUrl) => {
            img.onload = () => {
                phoneCanvas.width = img.width;
                phoneCanvas.height = img.height;
                phoneCtx.drawImage(img, 0, 0);
            };
            img.src = dataUrl;

            // First frame received = connected
            if (!phoneStatus.classList.contains("connected")) {
                phoneStatus.classList.add("connected");
                phoneStatus.textContent = "Phone connected";
                activateCamera();
                btnPhoneCamera.textContent = "Phone Active";
            }
        });
    });

    btnSwitchCamera.addEventListener("click", async () => {
        try {
            const stream = await Camera.switchCamera();
            cameraFeed.srcObject = stream;
        } catch (err) {
            alert("Could not switch camera: " + err.message);
        }
    });

    // Take photo: snapshot current feed and skip straight to step 3
    btnTakePhoto.addEventListener("click", async () => {
        const snapCanvas = document.createElement("canvas");
        const snapCtx = snapCanvas.getContext("2d");

        if (Camera.isPhoneMode()) {
            snapCanvas.width = phoneCanvas.width;
            snapCanvas.height = phoneCanvas.height;
            snapCtx.drawImage(phoneCanvas, 0, 0);
        } else {
            snapCanvas.width = cameraFeed.videoWidth;
            snapCanvas.height = cameraFeed.videoHeight;
            snapCtx.drawImage(cameraFeed, 0, 0);
        }

        const blob = await new Promise((resolve) =>
            snapCanvas.toBlob(resolve, "image/png")
        );

        capturedData = {
            canvas: snapCanvas,
            imageBlob: blob,
            label: "object",
            bbox: { x: 0, y: 0, w: snapCanvas.width, h: snapCanvas.height },
        };

        goToStep(3);
    });

    // Upload photo
    btnUploadPhoto.addEventListener("click", () => fileUpload.click());

    fileUpload.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const img = new Image();
        img.onload = async () => {
            const snapCanvas = document.createElement("canvas");
            snapCanvas.width = img.width;
            snapCanvas.height = img.height;
            snapCanvas.getContext("2d").drawImage(img, 0, 0);

            const blob = await new Promise((resolve) =>
                snapCanvas.toBlob(resolve, "image/png")
            );

            capturedData = {
                canvas: snapCanvas,
                imageBlob: blob,
                label: "object",
                bbox: { x: 0, y: 0, w: img.width, h: img.height },
            };

            goToStep(3);
        };
        img.src = URL.createObjectURL(file);
        fileUpload.value = "";
    });

    btnToDetect.addEventListener("click", () => {
        goToStep(2);

        if (Camera.isPhoneMode()) {
            // For phone mode, use the phone canvas as source for detection
            Detection.init(phoneCanvas, detectCanvas, true);
        } else {
            detectFeed.srcObject = Camera.getStream();
            Detection.init(detectFeed, detectCanvas, false);
        }

        Detection.onDetection((detections) => {
            if (detections.length > 0) {
                const d = detections[0];
                detectedLabel.textContent = `${d.label} (${(d.confidence * 100).toFixed(0)}% confidence)`;
                btnCapture.disabled = false;
            } else {
                detectedLabel.textContent = "No object detected";
                btnCapture.disabled = true;
            }
        });
        Detection.startDetecting();
    });

    // --- Step 2: Detection ---
    btnCapture.addEventListener("click", async () => {
        capturedData = await Detection.captureDetection();
        if (capturedData) {
            capturedPreview.classList.remove("hidden");
            captureCanvas.width = capturedData.canvas.width;
            captureCanvas.height = capturedData.canvas.height;
            captureCanvas
                .getContext("2d")
                .drawImage(capturedData.canvas, 0, 0);
            captureLabel.textContent = `Detected: ${capturedData.label}`;
            btnToModel.disabled = false;
        }
    });

    btnBackToCamera.addEventListener("click", () => {
        Detection.stopDetecting();
        goToStep(1);
    });

    btnToModel.addEventListener("click", () => {
        Detection.stopDetecting();
        goToStep(3);
    });

    // --- Step 3: Model Generation ---
    btnGenerate.addEventListener("click", async () => {
        if (!capturedData) return;

        loadingSpinner.classList.remove("hidden");
        btnGenerate.disabled = true;
        progressFill.style.width = "0%";
        progressText.textContent = "0%";

        ModelGen.onProgress((pct) => {
            progressFill.style.width = pct + "%";
            progressText.textContent = pct + "%";
        });

        try {
            const model = await ModelGen.generate(
                capturedData.imageBlob,
                capturedData.label
            );
            loadingSpinner.classList.add("hidden");
            btnRegenerate.classList.remove("hidden");
            btnToInteract.disabled = false;

            await HandInteract.init(threeCanvas, handFeed, handOverlay);
            HandInteract.loadOBJ(model.objUrl);
        } catch (err) {
            loadingSpinner.classList.add("hidden");
            btnGenerate.disabled = false;
            alert("Model generation failed: " + err.message);
        }
    });

    btnRegenerate.addEventListener("click", () => {
        btnRegenerate.classList.add("hidden");
        btnToInteract.disabled = true;
        btnGenerate.disabled = false;
        btnGenerate.click();
    });

    btnBackToDetect.addEventListener("click", () => goToStep(2));

    btnToInteract.addEventListener("click", () => goToStep(4));

    // --- Step 4: Hand Interaction ---
    btnStartHands.addEventListener("click", async () => {
        try {
            // Hand tracking always uses the laptop webcam
            let stream = Camera.isPhoneMode() ? null : Camera.getStream();
            if (!stream) {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false,
                });
            }
            await HandInteract.startHandTracking(stream);

            HandInteract.onHudUpdate((data) => {
                hudHands.textContent = data.hands;
                hudScale.textContent = data.scale + "x";
                hudMode.textContent = data.mode;
            });

            btnStartHands.textContent = "Tracking Active";
            btnStartHands.disabled = true;
        } catch (err) {
            alert("Hand tracking failed: " + err.message);
        }
    });

    btnRecenter.addEventListener("click", () => {
        HandInteract.recenter();
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "r" || e.key === "R") {
            if (currentStep === 4) {
                HandInteract.recenter();
            }
        }
    });

    btnToggleCam.addEventListener("click", () => {
        HandInteract.toggleCameraVisibility();
    });

    btnBackToModel.addEventListener("click", () => {
        HandInteract.stopHandTracking();
        goToStep(3);
    });

    btnResetAll.addEventListener("click", () => {
        HandInteract.stopHandTracking();
        Detection.stopDetecting();
        Camera.stop();
        capturedData = null;
        capturedPreview.classList.add("hidden");
        btnStartCamera.disabled = false;
        btnStartCamera.textContent = "Laptop Camera";
        btnPhoneCamera.disabled = false;
        btnPhoneCamera.textContent = "Phone Camera";
        btnSwitchCamera.disabled = true;
        btnTakePhoto.disabled = true;
        btnToDetect.disabled = true;
        btnCapture.disabled = true;
        btnToModel.disabled = true;
        btnGenerate.disabled = false;
        btnRegenerate.classList.add("hidden");
        btnToInteract.disabled = true;
        btnStartHands.disabled = false;
        btnStartHands.textContent = "Start Hand Tracking";
        cameraOverlay.classList.remove("hidden");
        phoneStatus.classList.add("hidden");
        phoneStatus.classList.remove("connected");
        phoneStatus.textContent = "Waiting for phone stream...";
        cameraFeed.srcObject = null;
        cameraFeed.style.display = "";
        phoneCanvas.style.display = "none";
        goToStep(1);
    });
})();
