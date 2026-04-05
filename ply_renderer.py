# This file already contains great controls
# Lowkey why is this one so good

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

cap = cv2.VideoCapture(0) # 1 bc mac sucks
if not cap.isOpened():
    print("cannot open camera")
    exit()
    
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

t0 = time.monotonic()

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
    # Pure time based smoothing
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

    # Re-orthonormalize so it stays a valid rotation basis
    x = blended[:, 0]
    y = blended[:, 1]

    x = normalize(x)
    y = y - np.dot(y, x) * x
    y = normalize(y)
    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))

    return np.column_stack((x, y, z)).astype(np.float32)

# Visualizer Config
WIDTH, HEIGHT = 1000, 700
BG = (20, 20, 30)
WHITE = (240, 240, 240)
BLUE = (100, 200, 255)
TARGET_FPS = 30
FACE_STEP = 10

pygame.init()
pygame.font.init()
font = pygame.font.SysFont(None, 28)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Object Viewer")
clock = pygame.time.Clock()

# 3D math is hell on earth
def rotate_x(point, angle):
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    y2 = y * cos_a - z * sin_a
    z2 = y * sin_a + z * cos_a
    return (x, y2, z2)

def rotate_y(point, angle):
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a + z * sin_a
    z2 = -x * sin_a + z * cos_a
    return (x2, y, z2)

def rotate_z(point, angle):
    x, y, z = point
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a - y * sin_a
    y2 = x * sin_a + y * cos_a
    return (x2, y2, z)
    
def project(point, fov, viewer_distance):
    x, y, z = point

    # Prevent divide-by-zero when too close
    z += viewer_distance
    if z <= 0.1:
        z = 0.1

    factor = fov / z
    screen_x = x * factor + WIDTH / 2
    screen_y = -y * factor + HEIGHT / 2
    return (int(screen_x), int(screen_y))

def compute_hand_frame(hand_landmarks):
    wrist = landmark_to_np(hand_landmarks[0])
    index_mcp = landmark_to_np(hand_landmarks[5])
    pinky_mcp = landmark_to_np(hand_landmarks[17])
    middle_mcp = landmark_to_np(hand_landmarks[9])
    
    # x-axis: across the palm from pinky side to index side
    x_axis = normalize(index_mcp - pinky_mcp)

    # temporary vector from wrist toward fingers
    palm_up = normalize(middle_mcp - wrist)

    # z-axis: palm normal
    z_axis = normalize(np.cross(x_axis, palm_up))

    # recompute y-axis so axes are perfectly orthogonal
    y_axis = normalize(np.cross(z_axis, x_axis))

    origin = wrist
    return origin, x_axis, y_axis, z_axis

def hand_rotation_matrix(hand_landmarks):
    _, x_axis, y_axis, z_axis = compute_hand_frame(hand_landmarks)

    R = np.column_stack((x_axis, y_axis, z_axis))
    return R

def rotation_matrix_to_euler(R):
    # Protect against numerical issues
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

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )
    
def mat_rotate_point(R, point):
    p = np.array(point, dtype=np.float32)
    rp = R @ p
    return (float(rp[0]), float(rp[1]), float(rp[2]))
    
def get_hand_orientation(hand_landmarks):
    wrist = landmark_to_np(hand_landmarks[0])
    index_mcp = landmark_to_np(hand_landmarks[5])
    middle_mcp = landmark_to_np(hand_landmarks[9])
    pinky_mcp = landmark_to_np(hand_landmarks[17])

    # local hand axes
    x_axis = normalize(index_mcp - pinky_mcp)     # across palm
    y_temp = normalize(middle_mcp - wrist)        # toward fingers
    z_axis = normalize(np.cross(x_axis, y_temp))  # palm normal
    y_axis = normalize(np.cross(z_axis, x_axis))  # corrected orthogonal y

    # rotation matrix
    R = np.column_stack((x_axis, y_axis, z_axis))

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    return {
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "roll": math.degrees(roll),
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    }
    
def count_extended_fingers(hand_landmarks):
    """
    Simple heuristic finger counter.
    Counts index/middle/ring/pinky by checking if tip is above pip in image space.
    Counts thumb by checking horizontal separation from thumb IP joint.
    Good enough for quick gesture mode switching.
    """
    # Landmark indices
    THUMB_TIP = 4
    THUMB_IP = 3

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    count = 0

    # Index/middle/ring/pinky:
    # In webcam image coordinates, smaller y = higher on screen
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        if hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y:
            count += 1

    # Thumb:
    # crude heuristic: thumb tip significantly left/right of thumb IP
    if abs(hand_landmarks[THUMB_TIP].x - hand_landmarks[THUMB_IP].x) > 0.04:
        count += 1

    return count


def get_mode_from_fingers(finger_count):
    """
    0 fingers -> pause
    1 finger  -> scale
    2 fingers -> rotate
    anything else -> pause
    """
    if finger_count == 1:
        return "scale"
    elif finger_count == 2:
        return "rotate"
    else:
        return "pause"

def load_ply(filename):
    with open(filename, "rb") as f:
        header_lines = []
        is_binary = False
        little_endian = True
        vertex_count = 0
        face_count = 0

        vertex_properties = []
        current_element = None

        # -------- read header --------
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")

            line_str = line.decode("ascii").strip()
            header_lines.append(line_str)

            if line_str == "ply":
                continue

            elif line_str.startswith("format"):
                if "ascii" in line_str:
                    is_binary = False
                elif "binary_little_endian" in line_str:
                    is_binary = True
                    little_endian = True
                elif "binary_big_endian" in line_str:
                    is_binary = True
                    little_endian = False
                else:
                    raise ValueError(f"Unsupported PLY format: {line_str}")

            elif line_str.startswith("element"):
                parts = line_str.split()
                current_element = parts[1]
                count = int(parts[2])

                if current_element == "vertex":
                    vertex_count = count
                elif current_element == "face":
                    face_count = count

            elif line_str.startswith("property"):
                parts = line_str.split()
                if current_element == "vertex":
                    # e.g. property float x
                    #      property uchar red
                    if parts[1] == "list":
                        raise ValueError("List properties in vertex section are not supported")
                    prop_type = parts[1]
                    prop_name = parts[2]
                    vertex_properties.append((prop_type, prop_name))

            elif line_str == "end_header":
                break

        # -------- helpers --------
        import struct

        type_map = {
            "char":   "b",
            "uchar":  "B",
            "int8":   "b",
            "uint8":  "B",
            "short":  "h",
            "ushort": "H",
            "int16":  "h",
            "uint16": "H",
            "int":    "i",
            "uint":   "I",
            "int32":  "i",
            "uint32": "I",
            "float":  "f",
            "float32":"f",
            "double": "d",
            "float64":"d",
        }

        endian = "<" if little_endian else ">"

        vertices = []
        vertex_colors = []
        faces = []

        prop_names = [name for _, name in vertex_properties]

        # -------- read vertices --------
        if not is_binary:
            # ASCII PLY
            for _ in range(vertex_count):
                line = f.readline().decode("ascii").strip()
                parts = line.split()

                values = {}
                for (prop_type, prop_name), raw in zip(vertex_properties, parts):
                    if prop_type in ("float", "float32", "double", "float64"):
                        values[prop_name] = float(raw)
                    else:
                        values[prop_name] = int(raw)

                x = float(values["x"])
                y = float(values["y"])
                z = float(values["z"])
                vertices.append((x, y, z))

                if all(c in values for c in ("red", "green", "blue")):
                    vertex_colors.append((values["red"], values["green"], values["blue"]))

            for _ in range(face_count):
                line = f.readline().decode("ascii").strip()
                parts = line.split()
                n = int(parts[0])
                idxs = list(map(int, parts[1:1+n]))

                if len(idxs) == 3:
                    faces.append(tuple(idxs))
                elif len(idxs) > 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append((idxs[0], idxs[i], idxs[i + 1]))

        else:
            # Binary PLY
            vertex_fmt = endian + "".join(type_map[prop_type] for prop_type, _ in vertex_properties)
            vertex_size = struct.calcsize(vertex_fmt)

            for _ in range(vertex_count):
                data = f.read(vertex_size)
                if len(data) != vertex_size:
                    raise ValueError("Unexpected EOF while reading binary vertex data")

                unpacked = struct.unpack(vertex_fmt, data)
                values = {name: val for (_, name), val in zip(vertex_properties, unpacked)}

                x = float(values["x"])
                y = float(values["y"])
                z = float(values["z"])
                vertices.append((x, y, z))

                if all(c in values for c in ("red", "green", "blue")):
                    vertex_colors.append((
                        int(values["red"]),
                        int(values["green"]),
                        int(values["blue"])
                    ))

            # Assume standard face property:
            # property list uchar int vertex_indices
            for _ in range(face_count):
                count_data = f.read(1)
                if not count_data:
                    raise ValueError("Unexpected EOF while reading binary face list count")

                n = struct.unpack(endian + "B", count_data)[0]

                idx_data = f.read(4 * n)
                if len(idx_data) != 4 * n:
                    raise ValueError("Unexpected EOF while reading binary face indices")

                idxs = list(struct.unpack(endian + ("i" * n), idx_data))

                if len(idxs) == 3:
                    faces.append(tuple(idxs))
                elif len(idxs) > 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append((idxs[0], idxs[i], idxs[i + 1]))

        if len(vertex_colors) != len(vertices):
            vertex_colors = None

        return vertices, faces, vertex_colors


def normalize_model(vertices, target_size=2.0):
    arr = np.array(vertices, dtype=np.float32)

    min_v = arr.min(axis=0)
    max_v = arr.max(axis=0)
    center = (min_v + max_v) / 2.0
    size = (max_v - min_v).max()

    if size < 1e-9:
        size = 1.0

    arr = (arr - center) * (target_size / size)
    return [tuple(v) for v in arr]


def face_normal(v0, v1, v2):
    a = np.array(v1) - np.array(v0)
    b = np.array(v2) - np.array(v0)
    n = np.cross(a, b)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return n / norm


def shade_color(normal, light_dir=np.array([0.4, -0.5, -1.0], dtype=np.float32)):
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Lambert shading
    brightness = np.dot(normal, -light_dir)
    brightness = max(0.15, min(1.0, brightness))

    base = np.array([180, 210, 255], dtype=np.float32)
    color = np.clip(base * brightness, 0, 255).astype(np.uint8)
    return tuple(map(int, color))
        
def triangle_normal(v0, v1, v2):
    a = np.array(v1, dtype=np.float32) - np.array(v0, dtype=np.float32)
    b = np.array(v2, dtype=np.float32) - np.array(v0, dtype=np.float32)
    n = np.cross(a, b)
    mag = np.linalg.norm(n)
    if mag < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return n / mag


def triangle_screen_bounds(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def draw_obj_model_numpy(screen, vertices_np, faces_np, display_R, scale, fov, camera_distance, vertex_colors_np=None):    # Transform all vertices at once
    transformed = (vertices_np * scale) @ display_R.T   # shape: (N, 3)

    # Project all vertices at once
    z = transformed[:, 2] + camera_distance
    valid_z = z > 1.0

    z_safe = np.maximum(z, 1.0)
    factor = fov / z_safe

    screen_x = transformed[:, 0] * factor + WIDTH / 2
    screen_y = -transformed[:, 1] * factor + HEIGHT / 2
    projected = np.stack([screen_x, screen_y], axis=1)  # shape: (N, 2)

    light = np.array([0.4, -0.5, -1.0], dtype=np.float32)
    light /= np.linalg.norm(light)

    draw_calls = []

    for i0, i1, i2 in faces_np[::FACE_STEP]:
        # Near-plane cull
        if not (valid_z[i0] and valid_z[i1] and valid_z[i2]):
            continue

        v0 = transformed[i0]
        v1 = transformed[i1]
        v2 = transformed[i2]

        # Normal
        a = v1 - v0
        b = v2 - v0
        n = np.cross(a, b)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm

        # Backface cull
        if n[2] >= 0:
            continue

        polyf = projected[[i0, i1, i2]]

        # Off-screen cull
        min_x = np.min(polyf[:, 0])
        max_x = np.max(polyf[:, 0])
        min_y = np.min(polyf[:, 1])
        max_y = np.max(polyf[:, 1])

        if max_x < 0 or min_x > WIDTH or max_y < 0 or min_y > HEIGHT:
            continue

        brightness = float(np.dot(n, -light))
        brightness = max(0.18, min(1.0, brightness))

        if vertex_colors_np is not None:
            base = (
                vertex_colors_np[i0].astype(np.float32) +
                vertex_colors_np[i1].astype(np.float32) +
                vertex_colors_np[i2].astype(np.float32)
            ) / 3.0
        else:
            base = np.array([180, 210, 255], dtype=np.float32)

        color = tuple(np.clip(base * brightness, 0, 255).astype(np.uint8))

        poly = [(int(polyf[0, 0]), int(polyf[0, 1])),
                (int(polyf[1, 0]), int(polyf[1, 1])),
                (int(polyf[2, 0]), int(polyf[2, 1]))]

        avg_z = float((v0[2] + v1[2] + v2[2]) / 3.0)
        draw_calls.append((avg_z, color, poly))

    draw_calls.sort(key=lambda item: item[0], reverse=True)

    for _, color, poly in draw_calls:
        pygame.draw.polygon(screen, color, poly)
        pygame.draw.polygon(screen, WHITE, poly, 1)

# OBJ model
MODEL_PATH = "mesh.ply"   # change to your .ply file
vertices, faces, vertex_colors = load_ply(MODEL_PATH)
vertices = normalize_model(vertices, target_size=2.0)

vertices_np = np.array(vertices, dtype=np.float32)
faces_np = np.array(faces, dtype=np.int32)

if vertex_colors is not None:
    vertex_colors_np = np.array(vertex_colors, dtype=np.uint8)
else:
    vertex_colors_np = None

scale = 120.0
target_scale = 120.0
neutral_R = None
display_R = np.eye(3, dtype=np.float32)
target_R = np.eye(3, dtype=np.float32)
recenter_requested = False
hand_present_last_frame = False

# smoothing tuning
rotation_follow_speed = 2.5
max_rotation_speed = 100.0
angle_deadzone = 2.5
scale_follow_speed = 6.0
fov = 800
viewer_distance = 5
yaw_sensitivity = 1.5     # left/right
pitch_sensitivity = 2.75   # up/down
roll_sensitivity = 1.2  # side/side

# combined loop
with HandLandmarker.create_from_options(options) as landmarker:
    # Pygame loop
    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    recenter_requested = True

        keys = pygame.key.get_pressed()
            
        viewer_distance = max(1.5, viewer_distance)
        
        # Run cv2 in pygame
        ok, frame = cap.read()
        if not ok:
            print("cannot receive frame, exiting...")
            break

        frame = cv2.flip(frame, 1)
        ts = int((time.monotonic() - t0) * 1000)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        res = landmarker.detect_for_video(mp_img, ts)
        
        h, w, _ = frame.shape
        
        hands = res.hand_landmarks if res.hand_landmarks else []
        hand_count = len(hands)

        control_hand = None
        mode_hand = None
        mode = "pause"

        if hand_count >= 2:
            control_hand = hands[0]
            mode_hand = hands[1]

            finger_count = count_extended_fingers(mode_hand)
            mode = get_mode_from_fingers(finger_count)

            cv2.putText(
                frame,
                f"Mode hand fingers: {finger_count}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Mode: {mode}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if mode != "pause" else (0, 0, 255),
                2
            )

            # Recenter when switching back into rotation control
            if mode == "rotate":
                current_R = hand_rotation_matrix(control_hand)

                if (not hand_present_last_frame) or (neutral_R is None) or recenter_requested:
                    neutral_R = current_R.copy()
                    recenter_requested = False
                    print("Recentered neutral pose")

                rel_R = current_R @ neutral_R.T

                # Convert to Euler angles
                roll, yaw, pitch = rotation_matrix_to_euler(rel_R)

                # Apply per-axis sensitivity scaling
                yaw *= yaw_sensitivity
                pitch *= pitch_sensitivity
                roll *= roll_sensitivity
                
                def apply_deadzone(val, threshold=2.0):
                    return 0 if abs(val) < threshold else val

                yaw = apply_deadzone(yaw)
                pitch = apply_deadzone(pitch)
                roll = apply_deadzone(roll)

                # Convert back to rotation matrix
                def euler_to_matrix(yaw, pitch, roll):
                    yaw = math.radians(yaw)
                    pitch = math.radians(pitch)
                    roll = math.radians(roll)

                    Rx = np.array([
                        [1, 0, 0],
                        [0, math.cos(pitch), -math.sin(pitch)],
                        [0, math.sin(pitch), math.cos(pitch)]
                    ], dtype=np.float32)

                    Ry = np.array([
                        [math.cos(yaw), 0, math.sin(yaw)],
                        [0, 1, 0],
                        [-math.sin(yaw), 0, math.cos(yaw)]
                    ], dtype=np.float32)

                    Rz = np.array([
                        [math.cos(roll), -math.sin(roll), 0],
                        [math.sin(roll), math.cos(roll), 0],
                        [0, 0, 1]
                    ], dtype=np.float32)

                    return Rz @ Ry @ Rx

                target_R = euler_to_matrix(yaw, pitch, roll).T

            elif mode == "scale":
                thumb_tip = control_hand[4]
                index_tip = control_hand[8]
                wrist = control_hand[0]
                middle_mcp = control_hand[9]

                pinch_dist = distance2d(thumb_tip, index_tip, w, h)
                ref_dist = distance2d(wrist, middle_mcp, w, h)

                if ref_dist > 0:
                    pinch_ratio = pinch_dist / ref_dist

                    ratio_min = 0.35
                    ratio_max = 2.2
                    normalized = (pinch_ratio - ratio_min) / (ratio_max - ratio_min)
                    normalized = clamp(normalized, 0.0, 1.0)

                    min_scale = 50
                    max_scale = 360
                    target_scale = min_scale + normalized * (max_scale - min_scale)

                    cv2.putText(
                        frame,
                        f"pinch ratio: {pinch_ratio:.2f}",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"cube target scale: {target_scale:.1f}",
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                x1 = int(thumb_tip.x * w)
                y1 = int(thumb_tip.y * h)
                x2 = int(index_tip.x * w)
                y2 = int(index_tip.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            else:
                # pause mode: do not update target_R or target_scale
                pass

            cv2.putText(
                frame,
                "Mode hand: 1 finger = scale, 2 fingers = rotate, 0 = pause",
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Press R to recenter",
                (30, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            for hand in hands:
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            hand_present_last_frame = (mode == "rotate")

        elif hand_count == 1:
            # One hand alone = pause
            control_hand = hands[0]

            cv2.putText(
                frame,
                "One hand detected: waiting for mode hand",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            cv2.putText(
                frame,
                "Raise 1 finger on other hand = scale | 2 fingers = rotate | 0 = pause",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Press R to recenter",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            for hand in hands:
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            hand_present_last_frame = False

        else:
            hand_present_last_frame = False
               
        scale_alpha = smooth_factor(scale_follow_speed, dt)
        scale = lerp(scale, target_scale, scale_alpha)
         
        rot_alpha = smooth_factor(rotation_follow_speed, dt)
        display_R = smooth_rotation_matrix(display_R, target_R, rot_alpha)

        # Draw OBJ model
        screen.fill(BG)
        camera_distance = 600
        draw_obj_model_numpy(
            screen,
            vertices_np,
            faces_np,
            display_R,
            scale,
            fov,
            camera_distance,
            vertex_colors_np
        )
          
        text1 = font.render(f"Cube scale: {scale:.1f}", True, WHITE)
        text_rot = font.render("Palm camera rotation: matrix mode", True, WHITE)
        text_target = font.render("2 hands: other hand selects mode | 1 finger scale | 2 rotate | 0 pause", True, WHITE)
        screen.blit(text_rot, (20, 80))
        screen.blit(text_target, (20, 110))
        screen.blit(text1, (20, 20))

        pygame.display.flip()
    
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()