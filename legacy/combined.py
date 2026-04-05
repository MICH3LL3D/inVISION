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

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return v / norm

def landmark_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

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
angle_x = 0.0
angle_y = 0.0
angle_z = 0.0
target_angle_x = 0.0
target_angle_y = 0.0
target_angle_z = 0.0
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

        keys = pygame.key.get_pressed()
        
        # Rotate object with arrow keys
        #if keys[pygame.K_LEFT]:
        #    angle_y -= 1.5 * dt
        #if keys[pygame.K_RIGHT]:
        #    angle_y += 1.5 * dt
        #if keys[pygame.K_UP]:
        #    angle_x -= 1.5 * dt
        #if keys[pygame.K_DOWN]:
        #    angle_x += 1.5 * dt
        #if keys[pygame.K_q]:
        #    angle_z -= 1.5 * dt
        #if keys[pygame.K_e]:
        #    angle_z += 1.5 * dt
        
        # Test constant rotation, slight birds eye FUUUUUUUUUU
        #angle_x = 60 / (180 * math.pi)
        #angle_y += 0.5 * dt

        # Zoom in/out by changing viewer distance (still works but don't use)
        if keys[pygame.K_w]:
            viewer_distance -= 2.0 * dt
        if keys[pygame.K_s]:
            viewer_distance += 2.0 * dt
            
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
        
        if res.hand_landmarks:
            landmarks1 = res.hand_landmarks[0]

            orientation = get_hand_orientation(landmarks1)

            # Keep everything in DEGREES here
            target_angle_x = -orientation["pitch"]
            target_angle_y = orientation["yaw"]
            target_angle_z = -orientation["roll"]

            print(
                f"yaw={target_angle_y:.1f}, "
                f"pitch={-target_angle_x:.1f}, "
                f"roll={-target_angle_z:.1f}"
            )
            print("palm normal:", orientation["z_axis"])

            thumb_tip = landmarks1[4]
            index_tip = landmarks1[8]

            # Points for normalization
            wrist = landmarks1[0]
            middle_mcp = landmarks1[9]

            # Pinch calculations
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

            # draw thumb-index line ONLY for visualization
            x1 = int(thumb_tip.x * w)
            y1 = int(thumb_tip.y * h)
            x2 = int(index_tip.x * w)
            y2 = int(index_tip.y * h)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.putText(
                frame,
                f"cube rot deg: x={target_angle_x:.1f} y={target_angle_y:.1f} z={target_angle_z:.1f}",
                (30, 120),
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
                
        # Smooth cube resizing
        scale = lerp(scale, target_scale, min(1.0, 8.0 * dt))
        angle_x = lerp_angle(angle_x, target_angle_x, 0.10)
        angle_y = lerp_angle(angle_y, target_angle_y, 0.10)
        angle_z = lerp_angle(angle_z, target_angle_z, 0.10)

        angle_x = wrap_angle(angle_x)
        angle_y = wrap_angle(angle_y)
        angle_z = wrap_angle(angle_z)

        # convert degrees -> radians ONLY for drawing
        draw_angle_x = math.radians(angle_x)
        draw_angle_y = math.radians(angle_y)
        draw_angle_z = math.radians(angle_z)

        # Draw cube
        screen.fill(BG)

        transformed = []
        for vertex in vertices:
            x, y, z = vertex
            point = (x * scale, y * scale, z * scale)

            point = rotate_x(point, draw_angle_x)
            point = rotate_y(point, draw_angle_y)
            point = rotate_z(point, draw_angle_z)

            transformed.append(point)

        camera_distance = 600
        projected = [project(p, fov, camera_distance) for p in transformed]

        for start, end in edges:
            pygame.draw.line(screen, WHITE, projected[start], projected[end], 2)

        for px, py in projected:
            pygame.draw.circle(screen, BLUE, (px, py), 5)
            
        text1 = font.render(f"Cube scale: {scale:.1f}", True, WHITE)
        text2 = font.render("Pinch thumb + index to grow/shrink cube", True, WHITE)
        screen.blit(text1, (20, 20))
        screen.blit(text2, (20, 50))
        
        text_rot = font.render(
            f"Rot X:{angle_x:.1f}  Y:{angle_y:.1f}  Z:{angle_z:.1f}",
            True,
            WHITE
        )
        screen.blit(text_rot, (20, 80))

        pygame.display.flip()
    
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()