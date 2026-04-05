/**
 * handInteract.js - Hand tracking + Three.js 3D interaction.
 *
 * Mirrors the logic from hand_cam.py:
 *   - Hand 1 controls rotation (palm orientation maps to object rotation)
 *   - Hand 2 pinch controls scale
 *   - Recenter resets neutral pose
 *
 * Uses MediaPipe Hands (browser) + Three.js for rendering.
 */

import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const HandInteract = (() => {
    // Three.js
    let scene, camera, renderer, mesh;

    // MediaPipe Hands
    let handsModule = null;
    let handLandmarker = null;

    // State (mirrors hand_cam.py)
    let neutralR = null;
    let displayR = new THREE.Matrix4().identity();
    let targetR = new THREE.Matrix4().identity();
    let currentScale = 1.0;
    let targetScale = 1.0;
    let handPresentLastFrame = false;

    // Smoothing
    const ROTATION_FOLLOW_SPEED = 2.5;
    const SCALE_FOLLOW_SPEED = 6.0;

    // DOM
    let threeCanvas, handFeed, handOverlay, handCtx;
    let running = false;
    let lastTime = performance.now();

    // Callbacks for HUD updates
    let hudCallback = null;

    function onHudUpdate(cb) {
        hudCallback = cb;
    }

    async function init(threeCanvasEl, handFeedEl, handOverlayEl) {
        threeCanvas = threeCanvasEl;
        handFeed = handFeedEl;
        handOverlay = handOverlayEl;
        handCtx = handOverlay.getContext("2d");

        initThreeScene();
    }

    function initThreeScene() {
        const rect = threeCanvas.parentElement.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0d0d1a);

        camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
        camera.position.set(0, 0, 5);

        renderer = new THREE.WebGLRenderer({
            canvas: threeCanvas,
            antialias: true,
        });
        renderer.setSize(w, h);
        renderer.setPixelRatio(window.devicePixelRatio);

        // Lights
        const ambient = new THREE.AmbientLight(0x404060, 1.5);
        scene.add(ambient);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
        dirLight.position.set(3, 5, 4);
        scene.add(dirLight);

        // Default cube
        loadPlaceholderCube();

        renderLoop();
    }

    function loadPlaceholderCube() {
        if (mesh) scene.remove(mesh);

        const geo = new THREE.BoxGeometry(1.5, 1.5, 1.5);
        const mat = new THREE.MeshStandardMaterial({
            color: 0x6c63ff,
            wireframe: false,
            metalness: 0.3,
            roughness: 0.6,
        });
        mesh = new THREE.Mesh(geo, mat);

        // Add edges for wireframe look
        const edges = new THREE.EdgesGeometry(geo);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x00d4ff });
        const wireframe = new THREE.LineSegments(edges, lineMat);
        mesh.add(wireframe);

        scene.add(mesh);
    }

    /**
     * Load a real OBJ model from a URL (from ModelGen).
     */
    function loadOBJ(objUrl) {
        const loader = new OBJLoader();
        loader.load(objUrl, (obj) => {
            if (mesh) scene.remove(mesh);

            // Center and normalize the model
            const box = new THREE.Box3().setFromObject(obj);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scaleFactor = 2.0 / maxDim;

            obj.position.sub(center);
            obj.scale.setScalar(scaleFactor);

            obj.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshStandardMaterial({
                        color: 0x6c63ff,
                        metalness: 0.3,
                        roughness: 0.6,
                    });
                    const edges = new THREE.EdgesGeometry(child.geometry);
                    const lineMat = new THREE.LineBasicMaterial({
                        color: 0x00d4ff,
                    });
                    child.add(new THREE.LineSegments(edges, lineMat));
                }
            });

            mesh = obj;
            scene.add(mesh);
        });
    }

    function renderLoop() {
        requestAnimationFrame(renderLoop);

        const now = performance.now();
        const dt = (now - lastTime) / 1000;
        lastTime = now;

        if (mesh) {
            // Smooth rotation toward target
            const rotAlpha = smoothFactor(ROTATION_FOLLOW_SPEED, dt);
            lerpMatrix4(displayR, targetR, rotAlpha);

            // Apply rotation
            mesh.rotation.setFromRotationMatrix(displayR);

            // Smooth scale
            const scaleAlpha = smoothFactor(SCALE_FOLLOW_SPEED, dt);
            currentScale = lerp(currentScale, targetScale, scaleAlpha);
            mesh.scale.setScalar(currentScale);
        }

        renderer.render(scene, camera);
    }

    /**
     * Start MediaPipe hand tracking.
     *
     * TODO: If you want server-side hand tracking instead, replace this with:
     *   - Stream frames to /api/hand-track via WebSocket
     *   - Receive landmarks back and apply the same rotation logic
     */
    async function startHandTracking(videoStream) {
        // Load MediaPipe Hands via CDN
        // @mediapipe/tasks-vision must be loaded
        const vision = await import(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs"
        );

        const { HandLandmarker, FilesetResolver } = vision;

        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );

        handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 2,
        });

        handFeed.srcObject = videoStream;
        running = true;
        detectHands();
    }

    function stopHandTracking() {
        running = false;
    }

    /**
     * Count extended fingers on a hand.
     * Mirrors count_extended_fingers() from updated_obj_render.py.
     */
    function countExtendedFingers(landmarks) {
        const fingerTips = [8, 12, 16, 20];
        const fingerPips = [6, 10, 14, 18];

        let count = 0;
        for (let i = 0; i < fingerTips.length; i++) {
            if (landmarks[fingerTips[i]].y < landmarks[fingerPips[i]].y) {
                count++;
            }
        }

        // Thumb heuristic: tip significantly left/right of IP joint
        if (Math.abs(landmarks[4].x - landmarks[3].x) > 0.04) {
            count++;
        }

        return count;
    }

    /** 1 finger = scale, 2 fingers = rotate, else = pause */
    function getModeFromFingers(fingerCount) {
        if (fingerCount === 1) return "scale";
        if (fingerCount === 2) return "rotate";
        return "pause";
    }

    function detectHands() {
        if (!running || !handLandmarker) return;

        if (handFeed.readyState >= 2) {
            // Resize overlay to match video
            handOverlay.width = handFeed.videoWidth;
            handOverlay.height = handFeed.videoHeight;
            handCtx.clearRect(0, 0, handOverlay.width, handOverlay.height);

            const results = handLandmarker.detectForVideo(
                handFeed,
                performance.now()
            );

            const hands = results.landmarks || [];
            const numHands = hands.length;

            // Draw all detected hand landmarks
            for (const hand of hands) {
                drawHandLandmarks(hand);
            }

            if (numHands >= 2) {
                const controlHand = hands[0];
                const modeHand = hands[1];

                const fingerCount = countExtendedFingers(modeHand);
                const mode = getModeFromFingers(fingerCount);

                if (mode === "rotate") {
                    const currentR = handRotationMatrix(controlHand);

                    // Recenter on first frame entering rotate mode
                    if (!handPresentLastFrame || !neutralR) {
                        neutralR = currentR.clone();
                    }

                    // Relative rotation: current * inverse(neutral)
                    const neutralInv = neutralR.clone().invert();
                    const relR = new THREE.Matrix4()
                        .copy(currentR)
                        .multiply(neutralInv);

                    // Object moves opposite (palm acts as camera)
                    targetR = relR.clone().invert();

                } else if (mode === "scale") {
                    const thumbTip = controlHand[4];
                    const indexTip = controlHand[8];
                    const wrist = controlHand[0];
                    const middleMcp = controlHand[9];

                    const pinchDist = distance2D(thumbTip, indexTip);
                    const refDist = distance2D(wrist, middleMcp);

                    if (refDist > 0) {
                        const pinchRatio = pinchDist / refDist;
                        const ratioMin = 0.35;
                        const ratioMax = 2.2;
                        const normalized = clamp(
                            (pinchRatio - ratioMin) / (ratioMax - ratioMin),
                            0,
                            1
                        );
                        const minScale = 0.4;
                        const maxScale = 3.0;
                        targetScale =
                            minScale + normalized * (maxScale - minScale);
                    }
                }
                // pause: do not update targetR or targetScale

                // handPresentLastFrame tracks whether we were in rotate mode
                handPresentLastFrame = mode === "rotate";

                if (hudCallback) {
                    hudCallback({
                        hands: numHands,
                        scale: currentScale.toFixed(1),
                        mode: `${mode} (${fingerCount} finger${fingerCount !== 1 ? "s" : ""})`,
                    });
                }
            } else {
                handPresentLastFrame = false;
                if (hudCallback) {
                    hudCallback({
                        hands: numHands,
                        scale: currentScale.toFixed(1),
                        mode: numHands === 1 ? "Waiting for mode hand" : "Waiting...",
                    });
                }
            }
        }

        requestAnimationFrame(detectHands);
    }

    /**
     * Compute a 4x4 rotation matrix from hand landmarks.
     * Mirrors compute_hand_frame / hand_rotation_matrix from hand_cam.py.
     */
    function handRotationMatrix(landmarks) {
        const wrist = lmToVec3(landmarks[0]);
        const indexMcp = lmToVec3(landmarks[5]);
        const pinkyMcp = lmToVec3(landmarks[17]);
        const middleMcp = lmToVec3(landmarks[9]);

        // x-axis: across palm (pinky -> index)
        const xAxis = new THREE.Vector3()
            .subVectors(indexMcp, pinkyMcp)
            .normalize();

        // temp: toward fingers
        const palmUp = new THREE.Vector3()
            .subVectors(middleMcp, wrist)
            .normalize();

        // z-axis: palm normal
        const zAxis = new THREE.Vector3()
            .crossVectors(xAxis, palmUp)
            .normalize();

        // y-axis: corrected orthogonal
        const yAxis = new THREE.Vector3()
            .crossVectors(zAxis, xAxis)
            .normalize();

        const m = new THREE.Matrix4();
        m.makeBasis(xAxis, yAxis, zAxis);
        return m;
    }

    function recenter() {
        neutralR = null;
        handPresentLastFrame = false;
        targetR = new THREE.Matrix4().identity();
        displayR = new THREE.Matrix4().identity();
        targetScale = 1.0;
    }

    // --- Helpers ---

    function lmToVec3(lm) {
        return new THREE.Vector3(lm.x, lm.y, lm.z);
    }

    function distance2D(a, b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(v, hi));
    }

    function lerp(a, b, t) {
        return a + (b - a) * t;
    }

    function smoothFactor(speed, dt) {
        return 1.0 - Math.exp(-speed * dt);
    }

    function lerpMatrix4(current, target, alpha) {
        const ce = current.elements;
        const te = target.elements;
        for (let i = 0; i < 16; i++) {
            ce[i] = ce[i] + (te[i] - ce[i]) * alpha;
        }
    }

    function drawHandLandmarks(landmarks) {
        const w = handOverlay.width;
        const h = handOverlay.height;

        handCtx.fillStyle = "#00ff88";
        for (const lm of landmarks) {
            const x = lm.x * w;
            const y = lm.y * h;
            handCtx.beginPath();
            handCtx.arc(x, y, 3, 0, Math.PI * 2);
            handCtx.fill();
        }

        // Draw connections
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
            [0, 5], [5, 6], [6, 7], [7, 8],       // index
            [0, 9], [9, 10], [10, 11], [11, 12],  // middle
            [0, 13], [13, 14], [14, 15], [15, 16], // ring
            [0, 17], [17, 18], [18, 19], [19, 20], // pinky
            [5, 9], [9, 13], [13, 17],              // palm
        ];

        handCtx.strokeStyle = "rgba(0, 255, 136, 0.5)";
        handCtx.lineWidth = 1;
        for (const [a, b] of connections) {
            const la = landmarks[a];
            const lb = landmarks[b];
            handCtx.beginPath();
            handCtx.moveTo(la.x * w, la.y * h);
            handCtx.lineTo(lb.x * w, lb.y * h);
            handCtx.stroke();
        }
    }

    function toggleCameraVisibility() {
        const isHidden = handFeed.style.display === "none";
        handFeed.style.display = isHidden ? "" : "none";
        handOverlay.style.display = isHidden ? "" : "none";
    }

    return {
        init,
        loadOBJ,
        startHandTracking,
        stopHandTracking,
        recenter,
        onHudUpdate,
        toggleCameraVisibility,
    };
})();

// Expose to window for non-module scripts
window.HandInteract = HandInteract;
