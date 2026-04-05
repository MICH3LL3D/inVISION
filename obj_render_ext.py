# Main file with best control scheme

from pathlib import Path

import mediapipe as mp
import numpy as np
import cv2
import time
import math
import pygame

# Hand Tracker Config
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

# face_step scales ~linearly with .obj size on disk: ref_mib → ref_step (default 2.8 MiB → 5).
_FACE_STEP_REF_MIB = 2
_FACE_STEP_REF_VALUE = 5
_FACE_STEP_MIN = 1
_FACE_STEP_MAX = 50


def face_step_for_obj_size(
    size_bytes: int,
    *,
    ref_mib: float = _FACE_STEP_REF_MIB,
    ref_step: int = _FACE_STEP_REF_VALUE,
    min_step: int = _FACE_STEP_MIN,
    max_step: int = _FACE_STEP_MAX,
) -> int:
    """Pick render stride from file size: ``round(ref_step * size / (ref_mib * MiB))``, clamped."""
    ref_bytes = ref_mib * 1024 * 1024
    if size_bytes <= 0 or ref_bytes <= 0:
        return max(min_step, min(ref_step, max_step))
    step = round(ref_step * size_bytes / ref_bytes)
    return max(min_step, min(step, max_step))


def face_step_from_obj_path(
    path: str | Path,
    *,
    ref_mib: float = _FACE_STEP_REF_MIB,
    ref_step: int = _FACE_STEP_REF_VALUE,
    min_step: int = _FACE_STEP_MIN,
    max_step: int = _FACE_STEP_MAX,
) -> int:
    p = Path(path)
    if not p.is_file():
        return ref_step
    return face_step_for_obj_size(
        p.stat().st_size,
        ref_mib=ref_mib,
        ref_step=ref_step,
        min_step=min_step,
        max_step=max_step,
    )


def distance2d(p1, p2, width, height):
    x1, y1 = p1.x * width, p1.y * height
    x2, y2 = p2.x * width, p2.y * height
    return math.hypot(x1 - x2, y1 - y2)

def clamp(value, low, high):
    return max(low, min(value, high))

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_angle(current, target, t):
    diff = (target - current + 180) % 360 - 180
    return current + diff * t

def wrap_angle(a):
    return (a + 180) % 360 - 180

def smooth_factor(speed, dt):
    return 1.0 - math.exp(-speed * dt)

def angle_delta(current, target):
    return (target - current + 180) % 360 - 180

def apply_angle_deadzone(current, target, deadzone_deg=2.0):
    diff = angle_delta(current, target)
    if abs(diff) < deadzone_deg:
        return current
    return target

def move_angle_toward(current, target, max_step_deg):
    diff = angle_delta(current, target)
    diff = clamp(diff, -max_step_deg, max_step_deg)
    return wrap_angle(current + diff)

def relative_angle(current, reference):
    return wrap_angle(current - reference)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return v / norm

def landmark_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def smooth_rotation_matrix(current_R, target_R, alpha):
    blended = (1.0 - alpha) * current_R + alpha * target_R

    x = blended[:, 0]
    y = blended[:, 1]

    x = normalize(x)
    y = y - np.dot(y, x) * x
    y = normalize(y)
    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))

    return np.column_stack((x, y, z)).astype(np.float32)

def compute_hand_frame(hand_landmarks):
    wrist = landmark_to_np(hand_landmarks[0])
    index_mcp = landmark_to_np(hand_landmarks[5])
    pinky_mcp = landmark_to_np(hand_landmarks[17])
    middle_mcp = landmark_to_np(hand_landmarks[9])

    x_axis = normalize(index_mcp - pinky_mcp)
    palm_up = normalize(middle_mcp - wrist)
    z_axis = normalize(np.cross(x_axis, palm_up))
    y_axis = normalize(np.cross(z_axis, x_axis))

    return wrist, x_axis, y_axis, z_axis

def hand_rotation_matrix(hand_landmarks):
    _, x_axis, y_axis, z_axis = compute_hand_frame(hand_landmarks)
    return np.column_stack((x_axis, y_axis, z_axis))

def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

def mat_rotate_point(R, point):
    p = np.array(point, dtype=np.float32)
    rp = R @ p
    return (float(rp[0]), float(rp[1]), float(rp[2]))

def get_hand_orientation(hand_landmarks):
    wrist = landmark_to_np(hand_landmarks[0])
    index_mcp = landmark_to_np(hand_landmarks[5])
    middle_mcp = landmark_to_np(hand_landmarks[9])
    pinky_mcp = landmark_to_np(hand_landmarks[17])

    x_axis = normalize(index_mcp - pinky_mcp)
    y_temp = normalize(middle_mcp - wrist)
    z_axis = normalize(np.cross(x_axis, y_temp))
    y_axis = normalize(np.cross(z_axis, x_axis))

    R = np.column_stack((x_axis, y_axis, z_axis))
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0

    return {
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "roll": math.degrees(roll),
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    }

def count_extended_fingers(hand_landmarks):
    THUMB_TIP = 4
    THUMB_IP = 3
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    count = 0
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        if hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y:
            count += 1

    if abs(hand_landmarks[THUMB_TIP].x - hand_landmarks[THUMB_IP].x) > 0.04:
        count += 1

    return count

def get_mode_from_fingers(finger_count):
    if finger_count == 1:
        return "scale"
    elif finger_count == 2:
        return "rotate"
    else:
        return "pause"

def euler_to_matrix(yaw, pitch, roll):
    yaw   = math.radians(yaw)
    pitch = math.radians(pitch)
    roll  = math.radians(roll)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)]
    ], dtype=np.float32)

    Ry = np.array([
        [ math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ], dtype=np.float32)

    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll),  math.cos(roll), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Class: HandTracker
# ---------------------------------------------------------------------------

class HandTracker:
    """Wraps the MediaPipe HandLandmarker for video-mode hand detection.

    Usage:
        tracker = HandTracker()
        with tracker:
            result = tracker.detect(mp_img, timestamp_ms)
    """

    def __init__(self, model_path="hand_landmarker.task", num_hands=2):
        self._options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._landmarker = None
        self._t0 = time.monotonic()

    def __enter__(self):
        self._landmarker = HandLandmarker.create_from_options(self._options)
        return self

    def __exit__(self, *_):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def detect(self, bgr_frame):
        """Run detection on a raw BGR frame (numpy array from cv2).

        Returns a list of hand landmark lists (one per detected hand).
        """
        frame = cv2.flip(bgr_frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_img, ts)
        flipped_frame = frame
        return result.hand_landmarks if result.hand_landmarks else [], flipped_frame


# ---------------------------------------------------------------------------
# Class: OBJModel
# ---------------------------------------------------------------------------

class OBJModel:
    """Loads and normalizes a Wavefront .obj file.

    Attributes:
        vertices_np  – (N, 3) float32 array of normalized vertices
        faces_np     – (M, 3) int32 array of triangle indices
    """

    def __init__(self, path, target_size=2.0):
        vertices, faces = self._load(path)
        vertices = self._normalize(vertices, target_size)
        self.vertices_np = np.array(vertices, dtype=np.float32)
        self.faces_np    = np.array(faces,    dtype=np.int32)

    @staticmethod
    def _load(filename):
        vertices = []
        faces = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append(tuple(map(float, parts[1:4])))
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    face = [int(p.split("/")[0]) - 1 for p in parts]
                    if len(face) == 3:
                        faces.append(tuple(face))
                    elif len(face) > 3:
                        for i in range(1, len(face) - 1):
                            faces.append((face[0], face[i], face[i + 1]))
        return vertices, faces

    @staticmethod
    def _normalize(vertices, target_size):
        arr = np.array(vertices, dtype=np.float32)
        min_v = arr.min(axis=0)
        max_v = arr.max(axis=0)
        center = (min_v + max_v) / 2.0
        size = (max_v - min_v).max()
        if size < 1e-9:
            size = 1.0
        arr = (arr - center) * (target_size / size)
        return [tuple(v) for v in arr]


# ---------------------------------------------------------------------------
# Class: ObjectViewerApp
# ---------------------------------------------------------------------------

class ObjectViewerApp:
    """Full 3-D object viewer driven by hand gestures.

    Parameters
    ----------
    obj_path : str
        Path to the .obj file to display.
    width, height : int
        Pygame window dimensions.
    target_fps : int
        Desired frame rate.
    face_step : int or None
        Render every N-th face (higher = faster but lower quality). If None, derived from
        ``.obj`` file size (~2.8 MiB → 5).
    model_path : str
        Path to the hand_landmarker.task model file.

    Example
    -------
    >>> from updated_obj_render import ObjectViewerApp
    >>> app = ObjectViewerApp("objs/Boeing KC-46A Pegasus.obj")
    >>> app.run()
    """

    # Colours
    BG    = (20, 20, 30)
    WHITE = (240, 240, 240)
    BLUE  = (100, 200, 255)

    def __init__(
        self,
        obj_path="objs/Boeing KC-46A Pegasus.obj",
        width=1000,
        height=700,
        target_fps=30,
        face_step=None,
        model_path="hand_landmarker.task",
    ):
        self.obj_path   = obj_path
        self.width      = width
        self.height     = height
        self.target_fps = target_fps
        if face_step is None:
            self.face_step = face_step_from_obj_path(obj_path)
        else:
            self.face_step = max(1, int(face_step))
        self.model_path = model_path

        # Smoothing / sensitivity tuning
        self.rotation_follow_speed = 2.5
        self.scale_follow_speed    = 6.0
        self.fov                   = 800
        self.viewer_distance       = 5
        self.yaw_sensitivity       = 1.5
        self.pitch_sensitivity     = 2.75
        self.roll_sensitivity      = 1.2

        # Runtime state (reset each call to run())
        self._scale       = 120.0
        self._target_scale = 120.0
        self._neutral_R   = None
        self._display_R   = np.eye(3, dtype=np.float32)
        self._target_R    = np.eye(3, dtype=np.float32)
        self._recenter    = False
        self._hand_last   = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """Open the camera and run the interactive viewer until the window is closed."""
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        model = OBJModel(self.obj_path)

        pygame.init()
        pygame.font.init()
        font   = pygame.font.SysFont(None, 28)
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3D Object Viewer")
        clock  = pygame.time.Clock()

        # Reset state
        self._scale        = 120.0
        self._target_scale = 120.0
        self._neutral_R    = None
        self._display_R    = np.eye(3, dtype=np.float32)
        self._target_R     = np.eye(3, dtype=np.float32)
        self._recenter     = False
        self._hand_last    = False

        with HandTracker(model_path=self.model_path) as tracker:
            running = True
            while running:
                dt = clock.tick(self.target_fps) / 1000.0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self._recenter = True

                ok, raw_frame = cap.read()
                if not ok:
                    print("Cannot receive frame, exiting...")
                    break

                hands, frame = tracker.detect(raw_frame)
                h, w, _ = frame.shape

                self._process_hands(hands, frame, w, h)
                self._update_smoothing(dt)

                # Draw 3-D scene
                screen.fill(self.BG)
                self._draw_obj(screen, model.vertices_np, model.faces_np)

                text1 = font.render(f"Cube scale: {self._scale:.1f}", True, self.WHITE)
                text2 = font.render("Palm camera rotation: matrix mode", True, self.WHITE)
                text3 = font.render(
                    "2 hands: other hand selects mode | 1 finger scale | 2 rotate | 0 pause",
                    True, self.WHITE
                )
                screen.blit(text1, (20, 20))
                screen.blit(text2, (20, 50))
                screen.blit(text3, (20, 80))

                pygame.display.flip()

                cv2.imshow("Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False

        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_deadzone(self, val, threshold=2.0):
        return 0 if abs(val) < threshold else val

    def _process_hands(self, hands, frame, w, h):
        hand_count = len(hands)

        if hand_count >= 2:
            control_hand = hands[0]
            mode_hand    = hands[1]
            finger_count = count_extended_fingers(mode_hand)
            mode         = get_mode_from_fingers(finger_count)

            cv2.putText(frame, f"Mode hand fingers: {finger_count}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {mode}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if mode != "pause" else (0, 0, 255), 2)

            if mode == "rotate":
                current_R = hand_rotation_matrix(control_hand)

                if (not self._hand_last) or (self._neutral_R is None) or self._recenter:
                    self._neutral_R = current_R.copy()
                    self._recenter  = False
                    print("Recentered neutral pose")

                rel_R = current_R @ self._neutral_R.T
                roll, yaw, pitch = rotation_matrix_to_euler(rel_R)

                yaw   *= self.yaw_sensitivity
                pitch *= self.pitch_sensitivity
                roll  *= self.roll_sensitivity

                yaw   = self._apply_deadzone(yaw)
                pitch = self._apply_deadzone(pitch)
                roll  = self._apply_deadzone(roll)

                self._target_R = euler_to_matrix(yaw, pitch, roll).T

            elif mode == "scale":
                thumb_tip  = control_hand[4]
                index_tip  = control_hand[8]
                wrist      = control_hand[0]
                middle_mcp = control_hand[9]

                pinch_dist = distance2d(thumb_tip, index_tip, w, h)
                ref_dist   = distance2d(wrist, middle_mcp, w, h)

                if ref_dist > 0:
                    pinch_ratio = pinch_dist / ref_dist
                    normalized  = clamp((pinch_ratio - 0.35) / (2.2 - 0.35), 0.0, 1.0)
                    self._target_scale = 50 + normalized * (360 - 50)

                    cv2.putText(frame, f"pinch ratio: {pinch_ratio:.2f}",
                                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"cube target scale: {self._target_scale:.1f}",
                                (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.putText(frame, "Mode hand: 1 finger = scale, 2 fingers = rotate, 0 = pause",
                        (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, "Press R to recenter",
                        (30, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            for hand in hands:
                for lm in hand:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

            self._hand_last = (mode == "rotate")

        elif hand_count == 1:
            cv2.putText(frame, "One hand detected: waiting for mode hand",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Raise 1 finger on other hand = scale | 2 fingers = rotate | 0 = pause",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, "Press R to recenter",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            for lm in hands[0]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

            self._hand_last = False
        else:
            self._hand_last = False

    def _update_smoothing(self, dt):
        scale_alpha = smooth_factor(self.scale_follow_speed, dt)
        self._scale = lerp(self._scale, self._target_scale, scale_alpha)

        rot_alpha = smooth_factor(self.rotation_follow_speed, dt)
        self._display_R = smooth_rotation_matrix(self._display_R, self._target_R, rot_alpha)

    def _draw_obj(self, screen, vertices_np, faces_np):
        camera_distance = 600
        fov = self.fov
        scale = self._scale
        display_R = self._display_R
        face_step = self.face_step
        width, height = self.width, self.height

        transformed = (vertices_np * scale) @ display_R.T

        z = transformed[:, 2] + camera_distance
        valid_z = z > 1.0
        z_safe = np.maximum(z, 1.0)
        factor = fov / z_safe

        screen_x = transformed[:, 0] * factor + width / 2
        screen_y = -transformed[:, 1] * factor + height / 2
        projected = np.stack([screen_x, screen_y], axis=1)

        light = np.array([0.4, -0.5, -1.0], dtype=np.float32)
        light /= np.linalg.norm(light)

        draw_calls = []

        for i0, i1, i2 in faces_np[::face_step]:
            if not (valid_z[i0] and valid_z[i1] and valid_z[i2]):
                continue

            v0, v1, v2 = transformed[i0], transformed[i1], transformed[i2]
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n /= norm

            if n[2] >= 0:
                continue

            polyf = projected[[i0, i1, i2]]
            if (np.max(polyf[:, 0]) < 0 or np.min(polyf[:, 0]) > width or
                    np.max(polyf[:, 1]) < 0 or np.min(polyf[:, 1]) > height):
                continue

            avg_z = float((v0[2] + v1[2] + v2[2]) / 3.0)
            poly  = [(int(polyf[r, 0]), int(polyf[r, 1])) for r in range(3)]
            draw_calls.append((avg_z, poly))

        draw_calls.sort(key=lambda item: item[0], reverse=True)
        for _, poly in draw_calls:
            pygame.draw.polygon(screen, self.WHITE, poly, 1)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ObjectViewerApp("objs/Boeing KC-46A Pegasus.obj")
    app.run()
