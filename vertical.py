# Proof of concept with vertical hand motion scrolling

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

pygame.init()
pygame.font.init()
font = pygame.font.SysFont(None, 28)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Wireframe Viewer")
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

def rotation_matrix_x(deg):
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float32)
    
def rotation_matrix_y(deg):
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)

def rotation_matrix_z(deg):
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float32)
    
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

# Model cube
vertices = [
    (-1, -1, -1),
    ( 1, -1, -1),
    ( 1,  1, -1),
    (-1,  1, -1),
    (-1, -1,  1),
    ( 1, -1,  1),
    ( 1,  1,  1),
    (-1,  1,  1),
]
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

scale = 120.0
target_scale = 120.0
neutral_palm_y = None
neutral_R = None
display_R = np.eye(3, dtype=np.float32)
target_R = np.eye(3, dtype=np.float32)
vertical_tilt_deg = 0.0
target_vertical_tilt_deg = 0.0
recenter_requested = False
hand_present_last_frame = False

# smoothing tuning
rotation_follow_speed = 2.5
max_rotation_speed = 100.0
angle_deadzone = 2.5
scale_follow_speed = 6.0
vertical_tilt_gain = 260.0      # how strong up/down movement affects tilt
vertical_tilt_limit = 80.0      # max tilt from vertical movement
fov = 400
viewer_distance = 5

# combined loop
with HandLandmarker.create_from_options(options) as landmarker:
    # Pygame loop
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

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
        
        hands = res.hand_landmarks
        rotation_hand = None
        scale_hand = None
        if res.hand_landmarks:
            rotation_hand = hands[0]
            scale_hand = hands[1] if len(hands) > 1 else None
            
            current_R = hand_rotation_matrix(rotation_hand)
            
            # y axis hand position for pitch
            wrist = rotation_hand[0]
            index_mcp = rotation_hand[5]
            middle_mcp = rotation_hand[9]
            pinky_mcp = rotation_hand[17]

            palm_center_y = (wrist.y + index_mcp.y + middle_mcp.y + pinky_mcp.y) / 4.0

            # Recenter when hand first appears or R is pressed
            if (not hand_present_last_frame) or (neutral_R is None) or recenter_requested:
                neutral_R = current_R.copy()
                neutral_palm_y = palm_center_y
                recenter_requested = False
                print("Recentered neutral pose")

            # Relative hand rotation from neutral pose
            # Maps neutral-hand space -> current-hand space
            rel_R = current_R @ neutral_R.T
            vertical_offset = neutral_palm_y - palm_center_y
            target_vertical_tilt_deg = clamp(
                vertical_offset * vertical_tilt_gain,
                -vertical_tilt_limit,
                vertical_tilt_limit
            )

            # Palm acts like camera, so object should move opposite
            camera_like_R = rel_R.T

            # Convert to Euler only so we can keep 2 axes from orientation
            yaw_deg, pitch_deg, roll_deg = rotation_matrix_to_euler(camera_like_R)

            # Use hand up/down for screen-style tilt
            vertical_offset = neutral_palm_y - palm_center_y
            target_vertical_tilt_deg = clamp(
                vertical_offset * vertical_tilt_gain,
                -vertical_tilt_limit,
                vertical_tilt_limit
            )

            # Keep 2 axes from palm orientation, replace the "scroll tilt" axis with vertical motion
            target_R = (
                rotation_matrix_x(vertical_tilt_deg) @
                rotation_matrix_y(pitch_deg) @
                rotation_matrix_z(roll_deg)
            ).astype(np.float32)

            # Optional debug
            orientation = get_hand_orientation(rotation_hand)
            print(
                f"hand yaw={orientation['yaw']:.1f}, "
                f"pitch={orientation['pitch']:.1f}, "
                f"roll={orientation['roll']:.1f}"
            )
            print("palm normal:", orientation["z_axis"])

            if scale_hand is not None:
                thumb_tip = scale_hand[4]
                index_tip = scale_hand[8]

                wrist = scale_hand[0]
                middle_mcp = scale_hand[9]

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

                    print(
                        f"pinch_ratio={pinch_ratio:.2f}, "
                        f"target_scale={target_scale:.1f}, scale={scale:.1f}"
                    )

                    cv2.putText(
                        frame, f"pinch ratio: {pinch_ratio:.2f}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"cube target scale: {target_scale:.1f}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                    )

                x1 = int(thumb_tip.x * w)
                y1 = int(thumb_tip.y * h)
                x2 = int(index_tip.x * w)
                y2 = int(index_tip.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.putText(
                frame,
                "Palm camera mode active",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Press R to recenter",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            for hand in res.hand_landmarks:
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            hand_present_last_frame = True
        else:
            hand_present_last_frame = False
               
        scale_alpha = smooth_factor(scale_follow_speed, dt)
        scale = lerp(scale, target_scale, scale_alpha)
        
        tilt_alpha = smooth_factor(rotation_follow_speed, dt)
        vertical_tilt_deg = lerp(vertical_tilt_deg, target_vertical_tilt_deg, tilt_alpha)
         
        rot_alpha = smooth_factor(rotation_follow_speed, dt)
        display_R = smooth_rotation_matrix(display_R, target_R, rot_alpha)

        # Draw cube
        screen.fill(BG)

        transformed = []
        for vertex in vertices:
            x, y, z = vertex
            point = (x * scale, y * scale, z * scale)

            point = mat_rotate_point(display_R, point)

            transformed.append(point)

        camera_distance = 600
        projected = [project(p, fov, camera_distance) for p in transformed]

        for start, end in edges:
            pygame.draw.line(screen, WHITE, projected[start], projected[end], 2)

        for px, py in projected:
            pygame.draw.circle(screen, BLUE, (px, py), 5)
            
        text1 = font.render(f"Cube scale: {scale:.1f}", True, WHITE)
        text_rot = font.render("Palm camera rotation: matrix mode", True, WHITE)
        text_target = font.render("Hand 1 rotates, Hand 2 pinch scales, R recenters", True, WHITE)
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