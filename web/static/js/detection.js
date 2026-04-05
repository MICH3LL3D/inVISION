/**
 * detection.js - Object detection module.
 *
 * BACKEND STUB: Replace the mock with a real call to your YOLO / detection server.
 *
 * Exports (on window.Detection):
 *   init(videoEl, canvasEl)   - bind to DOM elements
 *   startDetecting()          - begin detection loop
 *   stopDetecting()
 *   captureDetection()        -> { imageData: ImageData, label: string, bbox: {x,y,w,h} }
 *   onDetection(callback)     - called each frame with detection results
 */
window.Detection = (() => {
    let video = null;       // video element or canvas element (phone mode)
    let canvas = null;
    let ctx = null;
    let running = false;
    let animId = null;
    let callback = null;
    let lastDetection = null;
    let isCanvasSource = false;

    function init(sourceEl, canvasEl, canvasMode) {
        video = sourceEl;
        canvas = canvasEl;
        ctx = canvas.getContext("2d");
        isCanvasSource = !!canvasMode;
    }

    function onDetection(cb) {
        callback = cb;
    }

    function startDetecting() {
        running = true;
        loop();
    }

    function stopDetecting() {
        running = false;
        if (animId) cancelAnimationFrame(animId);
    }

    function loop() {
        if (!running) return;

        const sourceW = isCanvasSource ? video.width : video.videoWidth;
        const sourceH = isCanvasSource ? video.height : video.videoHeight;

        if (sourceW > 0) {
            canvas.width = sourceW;
            canvas.height = sourceH;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // -----------------------------------------------------------
            // TODO: Replace this mock with a real detection call.
            //
            // Option A: Run YOLO in the browser via ONNX Runtime Web
            //   const results = await onnxSession.run(inputTensor);
            //
            // Option B: Send frame to your backend:
            //   const blob = await captureFrameBlob();
            //   const res = await fetch('/api/detect', {
            //       method: 'POST',
            //       body: blob,
            //   });
            //   const detections = await res.json();
            //
            // Expected format per detection:
            //   { label: "cup", confidence: 0.92, bbox: { x, y, w, h } }
            // -----------------------------------------------------------

            const mockDetections = generateMockDetection(
                canvas.width,
                canvas.height
            );

            // Draw bounding boxes
            for (const det of mockDetections) {
                drawBBox(det);
            }

            lastDetection = mockDetections.length > 0 ? mockDetections[0] : null;

            if (callback) callback(mockDetections);
        }

        animId = requestAnimationFrame(loop);
    }

    /** Mock: simulates a centered detection box */
    function generateMockDetection(w, h) {
        const bw = w * 0.4;
        const bh = h * 0.5;
        return [
            {
                label: "object",
                confidence: 0.87,
                bbox: {
                    x: (w - bw) / 2,
                    y: (h - bh) / 2,
                    w: bw,
                    h: bh,
                },
            },
        ];
    }

    function drawBBox(det) {
        const { x, y, w, h } = det.bbox;
        ctx.strokeStyle = "#00d4ff";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "rgba(0, 212, 255, 0.15)";
        ctx.fillRect(x, y, w, h);

        ctx.fillStyle = "#00d4ff";
        ctx.font = "bold 16px sans-serif";
        ctx.fillText(
            `${det.label} (${(det.confidence * 100).toFixed(0)}%)`,
            x + 6,
            y - 8
        );
    }

    /**
     * Captures the current detected object as an image + metadata.
     * Returns { canvas, imageBlob, label, bbox }
     */
    async function captureDetection() {
        if (!lastDetection || !video) return null;

        const det = lastDetection;
        const { x, y, w, h } = det.bbox;

        const captureCanvas = document.createElement("canvas");
        captureCanvas.width = w;
        captureCanvas.height = h;
        const cctx = captureCanvas.getContext("2d");

        // Draw the cropped region from the video
        cctx.drawImage(video, x, y, w, h, 0, 0, w, h);

        const blob = await new Promise((resolve) =>
            captureCanvas.toBlob(resolve, "image/png")
        );

        return {
            canvas: captureCanvas,
            imageBlob: blob,
            label: det.label,
            bbox: det.bbox,
        };
    }

    return { init, onDetection, startDetecting, stopDetecting, captureDetection };
})();
