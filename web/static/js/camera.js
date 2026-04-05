/**
 * camera.js - Handles camera access.
 *
 * Supports two modes:
 *   1. Local webcam (getUserMedia)
 *   2. Phone stream (receives frames via SocketIO)
 *
 * Exports (on window.Camera):
 *   start()             -> Promise<MediaStream>   (local webcam)
 *   startPhoneStream()  -> sets up SocketIO receiver
 *   stop()
 *   switchCamera()      -> Promise<MediaStream>
 *   getStream()         -> MediaStream | null
 *   isPhoneMode()       -> boolean
 *   onPhoneFrame(cb)    -> register callback for phone frames
 */
window.Camera = (() => {
    let stream = null;
    let facingMode = "environment";
    let phoneMode = false;
    let phoneFrameCallback = null;
    let socket = null;

    async function start() {
        phoneMode = false;
        if (stream) stop();

        const constraints = {
            video: {
                facingMode,
                width: { ideal: 1280 },
                height: { ideal: 960 },
            },
            audio: false,
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        return stream;
    }

    /**
     * Connect to SocketIO and receive phone camera frames.
     * Frames arrive as base64 JPEG data URLs.
     * Call onPhoneFrame(cb) to handle them.
     */
    function startPhoneStream() {
        phoneMode = true;

        if (!socket) {
            socket = io({ transports: ["websocket"] });
        }

        socket.on("camera-frame", (dataUrl) => {
            console.log("Received phone frame:", dataUrl.length, "bytes");
            if (phoneFrameCallback) {
                phoneFrameCallback(dataUrl);
            }
        });

        socket.on("connect", () => {
            console.log("Laptop SocketIO connected, sid:", socket.id);
        });

        socket.on("disconnect", () => {
            console.log("Laptop SocketIO disconnected");
        });
    }

    function onPhoneFrame(cb) {
        phoneFrameCallback = cb;
    }

    function stop() {
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }
        phoneMode = false;
    }

    async function switchCamera() {
        facingMode = facingMode === "environment" ? "user" : "environment";
        return start();
    }

    function getStream() {
        return stream;
    }

    function isPhoneMode() {
        return phoneMode;
    }

    return { start, startPhoneStream, onPhoneFrame, stop, switchCamera, getStream, isPhoneMode };
})();
