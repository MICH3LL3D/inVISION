import math
import time

import cv2
import mediapipe as mp
import pygame

# ------------------------------------------
# Hand Tracker Config 
# ------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

CAMERA_INDEX = 0
MODEL_PATH = "hand_landmarker.task"
MAX_HANDS = 2

# ------------------------------------------
# Visualizer Config 
# ------------------------------------------
WIDTH, HEIGHT = 1000, 700
BG = (20, 20, 30)
WHITE = (240, 240, 240)
BLUE = (100, 200, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

MIN_SCALE = 50
MAX_SCALE = 360
INITIAL_SCALE = 120.0

PINCH_RATIO_MIN = 0.35
PINCH_RATIO_MAX = 2.2

ROTATE_SPEED = 1.5
ZOOM_SPEED = 2.0
SCALE_LERP_SPEED = 8.0

FOV = 400
CAMERA_DISTANCE = 600

# ------------------------------------------
# Helper Functions 
# ------------------------------------------
def distance2d(p1, p2, width, height):
    x1, y1 = p1.x * width, p1.y * height
    x2, y2 = p2.x * width, p2.y * height
    return math.hypot(x1 - x2, y1 - y2)

def clamp(value, low, high):
    return max(low, min(value, high))

def lerp(a, b, t):
    return a + (b - a) * t

def get_pixel_coords(lm, width, height):
    return int(lm.x * width), int(lm.y * height)

# ------------------------------------------
# 3D Math Helper Functions (3d math is hell on earth)
# ------------------------------------------
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

# ------------------------------------------
# Cube
# ------------------------------------------
VERTICES = [
    (-1, -1, -1),
    ( 1, -1, -1),
    ( 1,  1, -1),
    (-1,  1, -1),
    (-1, -1,  1),
    ( 1, -1,  1),
    ( 1,  1,  1),
    (-1,  1,  1),
]
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# ------------------------------------------
# Drawing Helper Functions
# ------------------------------------------
def draw_hand_landmarks(frame, hand_landmarks, width, height):
    for lm in hand_landmarks:
        x, y = get_pixel_coords(lm, width, height)
        cv2.circle(frame, (x, y), 5, GREEN, -1)


def draw_pinch_line(frame, thumb_tip, index_tip, width, height):
    x1, y1 = get_pixel_coords(thumb_tip, width, height)
    x2, y2 = get_pixel_coords(index_tip, width, height)
    cv2.line(frame, (x1, y1), (x2, y2), YELLOW, 3)


def draw_overlay_text(frame, pinch_ratio=None, target_scale=None, hand_angle_deg=None):
    y = 40

    if pinch_ratio is not None:
        cv2.putText(
            frame,
            f"pinch ratio: {pinch_ratio:.2f}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            GREEN,
            2,
        )
        y += 40

    if target_scale is not None:
        cv2.putText(
            frame,
            f"cube target scale: {target_scale:.1f}",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            GREEN,
            2,
        )
        y += 40

    # if hand_angle_deg is not None:
    #     cv2.putText(
    #         frame,
    #         f"hand angle: {hand_angle_deg:.1f}",
    #         (30, y),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.9,
    #         GREEN,
    #         2,
    #     )


def draw_cube(screen, font, scale, angle_x, angle_y, angle_z):
    screen.fill(BG)

    transformed = []
    for vertex in VERTICES:
        x, y, z = vertex
        point = (x * scale, y * scale, z * scale)
        point = rotate_x(point, angle_x)
        point = rotate_y(point, angle_y)
        point = rotate_z(point, angle_z)
        transformed.append(point)

    projected = [project(p, FOV, CAMERA_DISTANCE) for p in transformed]

    for start, end in EDGES:
        pygame.draw.line(screen, WHITE, projected[start], projected[end], 2)

    for px, py in projected:
        pygame.draw.circle(screen, BLUE, (px, py), 5)

    text1 = font.render(f"Cube scale: {scale:.1f}", True, WHITE)
    text2 = font.render("Pinch thumb + index to grow/shrink cube", True, WHITE)
    text3 = font.render(f"X angle: {math.degrees(angle_x):.1f}", True, WHITE)
    text4 = font.render(f"Y angle: {math.degrees(angle_y):.1f}", True, WHITE)
    text5 = font.render(f"Z angle: {math.degrees(angle_z):.1f}", True, WHITE)

    screen.blit(text1, (20, 20))
    screen.blit(text2, (20, 50))
    screen.blit(text3, (20, 80))
    screen.blit(text4, (20, 110))
    screen.blit(text5, (20, 130))

    pygame.display.flip()

# ------------------------------------------
# Hand Control Helper Functions
# ------------------------------------------
def compute_pinch_target_scale(hand_landmarks, width, height):
    thumb_tip = hand_landmarks[4]
    pinky_tip = hand_landmarks[20]
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    pinch_dist = distance2d(thumb_tip, pinky_tip, width, height)
    ref_dist = distance2d(wrist, middle_mcp, width, height)

    if ref_dist <= 0:
        return None, None, thumb_tip, pinky_tip

    pinch_ratio = pinch_dist / ref_dist
    normalized = (pinch_ratio - PINCH_RATIO_MIN) / (PINCH_RATIO_MAX - PINCH_RATIO_MIN)
    normalized = clamp(normalized, 0.0, 1.0)

    target_scale = MIN_SCALE + normalized * (MAX_SCALE - MIN_SCALE)
    return pinch_ratio, target_scale, thumb_tip, pinky_tip


def compute_hand_x_angle(hand_landmarks):
    """
    uses index and pinky points
    """
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    angle_x = math.atan2(-dx, dy)
    return angle_x

def compute_hand_y_angle(hand_landmarks):
    """
    uses index and pinky points
    """
    index = hand_landmarks[5]
    pinky = hand_landmarks[17]

    dx = pinky.x - index.x
    dy = pinky.y - index.y
    dz = pinky.z - index.z

    # mag = math.sqrt(dx*dx + dy*dy + dz*dz)
    # if mag == 0:
    #     return 0.0

    # dx /= mag
    # dz /= mag

    angle_y = math.atan2(-dz,dx)
    return angle_y

def compute_hand_z_angle(hand_landmarks):
    """
    uses middle and wrist points
    """
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    angle_z = math.atan2(-dy, dx)
    return angle_z

# ------------------------------------------
# Main Setup
# ------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("cannot open camera")
    raise SystemExit

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=MAX_HANDS,
    running_mode=VisionRunningMode.VIDEO,
)

pygame.init()
pygame.font.init()
font = pygame.font.SysFont(None, 28)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Wireframe Viewer")
clock = pygame.time.Clock()

t0 = time.monotonic()

scale = INITIAL_SCALE
target_scale = INITIAL_SCALE
angle_x = 0.0
angle_y = 0.0
angle_z = 0.0
target_angle_z = 0.0
viewer_distance = 5.0

def fingers_up(hand_landmarks):
    return {
        "index": hand_landmarks[8].y  < hand_landmarks[6].y,
        "middle": hand_landmarks[12].y < hand_landmarks[10].y,
        "ring": hand_landmarks[16].y < hand_landmarks[14].y,
        "pinky": hand_landmarks[20].y < hand_landmarks[18].y,
    }

def get_rotation_mode(left_hand):
    s = fingers_up(left_hand)

    if s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]:
        return "pitch"
    if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
        return "yaw"
    if s["index"] and s["middle"] and s["ring"] and not s["pinky"]:
        return "roll"

    return "idle"

# ------------------------------------------
# MAIN LOOP !!!!!
# ------------------------------------------
with HandLandmarker.create_from_options(options) as landmarker:
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        # ---- Pygame events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        viewer_distance = max(1.5, viewer_distance)

        # ---- Cam ----
        ok, frame = cap.read()
        if not ok:
            print("cannot receive frame, exiting...")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        ts = int((time.monotonic() - t0) * 1000)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = landmarker.detect_for_video(mp_img, ts)

        pinch_ratio = None
        hand_angle_deg = None

        # ---- Draw all landmarks ----
        if res.hand_landmarks:
            for hand_landmarks in res.hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks, w, h)

            mode = "idle"
            if len(res.hand_landmarks) == 2:
                scale_hand = res.hand_landmarks[1]

                pinch_ratio, target_scale_candidate, thumb_tip, pinky_tip = compute_pinch_target_scale(
                    scale_hand, w, h
                )

                if target_scale_candidate is not None:
                    target_scale = target_scale_candidate
                    scale = lerp(scale, target_scale, min(1.0, SCALE_LERP_SPEED * dt))
                    draw_pinch_line(frame, thumb_tip, pinky_tip, w, h)

                mode = get_rotation_mode(scale_hand)
                print(mode)

            rotate_hand = res.hand_landmarks[0]
            if mode == "pitch":
                target_angle_x = compute_hand_x_angle(rotate_hand)
                angle_x = target_angle_x
            if mode == "yaw":
                target_angle_y = compute_hand_y_angle(rotate_hand)
                angle_y = target_angle_y
            if mode == "roll":
                target_angle_z = compute_hand_z_angle(rotate_hand)
                angle_z = target_angle_z - math.pi/2
            # hand_angle_deg = math.degrees(angle_z)

        # ---- Draw UI / cube ----
        draw_overlay_text(
            frame,
            pinch_ratio=pinch_ratio,
            target_scale=target_scale if pinch_ratio is not None else None,
            hand_angle_deg=hand_angle_deg,
        )

        draw_cube(screen, font, scale, angle_x, angle_y, angle_z)

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
        
cap.release()
cv2.destroyAllWindows()
pygame.quit()