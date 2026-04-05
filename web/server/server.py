"""
server.py - Flask backend for inVISION.

Serves the static frontend and relays phone camera frames to the laptop via SocketIO.

To run:
    pip install flask flask-socketio eventlet
    python server.py
"""

import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder="../static", static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 * 1024 * 1024, async_mode="eventlet")


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/phone")
def phone():
    return send_from_directory(app.static_folder, "phone.html")


# ---------------------------------------------------------------------------
# SocketIO: Phone -> Server -> Laptop camera relay
# ---------------------------------------------------------------------------
@socketio.on("phone-frame")
def handle_phone_frame(data):
    """Phone sends a JPEG frame as base64, relay to all laptop viewers."""
    print(f"Received frame from phone ({len(data)} bytes), broadcasting...")
    emit("camera-frame", data, broadcast=True, include_self=False)


@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")


# ---------------------------------------------------------------------------
# API: Object Detection
# ---------------------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def detect_object():
    """
    TODO: Implement with your detection model (e.g., YOLOv8).
    """
    return jsonify([
        {
            "label": "object",
            "confidence": 0.87,
            "bbox": {"x": 100, "y": 100, "w": 300, "h": 350},
        }
    ])


# ---------------------------------------------------------------------------
# API: 3D Model Generation
# ---------------------------------------------------------------------------
@app.route("/api/generate-model", methods=["POST"])
def generate_model():
    """
    TODO: Implement with TripoSR or your 3D generation pipeline.
    """
    mesh_path = os.path.join(os.path.dirname(__file__), "../../mesh.obj")
    if os.path.exists(mesh_path):
        return send_from_directory(
            os.path.dirname(os.path.abspath(mesh_path)),
            "mesh.obj",
            mimetype="application/octet-stream",
        )
    return jsonify({"error": "Model generation not implemented yet"}), 501


# ---------------------------------------------------------------------------
# API: Generation Progress (SSE)
# ---------------------------------------------------------------------------
@app.route("/api/generate-model/progress/<task_id>")
def generation_progress(task_id):
    """TODO: Implement progress tracking."""
    return jsonify({"error": "Not implemented"}), 501


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting inVISION server...")
    print()
    print("  Laptop:  http://localhost:8080")
    print("  Phone:   http://localhost:8080/phone")
    print("           (use ngrok URL on iPhone)")
    print()
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)
