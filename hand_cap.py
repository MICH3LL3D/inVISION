import cv2
import mediapipe as mp
import time
import math

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
    running_mode=VisionRunningMode.VIDEO)

t0 = time.monotonic()

def distance2d(p1, p2, width, height):
    x1, y1 = p1.x * width, p1.y * height
    x2, y2 = p2.x * width, p2.y * height
    return math.hypot(x1 - x2, y1 - y2)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("cannot receive frame, exiting...")
            break
        
        ts = int((time.monotonic() - t0) * 1000)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        res = landmarker.detect_for_video(mp_img, ts)

        h, w, _ = frame.shape

        if res.hand_landmarks:
            landmarks = res.hand_landmarks[0]

            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            x1 = int(thumb_tip.x * w)
            y1 = int(thumb_tip.y * h)
            x2 = int(index_tip.x * w)
            y2 = int(index_tip.y * h)
            x3 = int(middle_tip.x * w)
            y3 = int(middle_tip.y * h)

            dist1 = distance2d(thumb_tip, index_tip, w, h)
            dist2 = distance2d(thumb_tip, middle_tip, w, h)
            
            # how to get relative distance working?
            scale = 1 # distance2d(landmarks[0], landmarks[1], w, h)
            norm_dist1 = dist1 / scale
            norm_dist2 = dist2 / scale
            
            cv2.putText(frame, f"relative distance between index and thumb: {norm_dist1:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, f"relative distance between middle and thumb: {norm_dist2:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(frame, (x1, y1), (x3, y3), (0, 255, 0), 3)
            cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 0), 3)
                        
            for hand in res.hand_landmarks:
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
          
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()