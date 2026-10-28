"""
الرسم في الهوا 🎨
    ☝️  إصبع واحد (السبابة)  = ترسم
    ✌️  إصبعين (السبابة + الوسطى) = تمسح
    ✊  أي شي ثاني = توقف (ترفع القلم)

الأزرار:
    1 أحمر | 2 أخضر | 3 أزرق | 4 أصفر
    c تمسح كل شي | s تحفظ الرسمة صورة | q خروج

التشغيل:
    python3 air_drawing.py
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ===== الإعدادات (تقدر تغيرها) =====
PEN_SIZE = 8
ERASER_SIZE = 50
SMOOTHING = 0.5           # 0 = بدون تنعيم، كل ما زاد الرقم صار الخط أنعم بس أبطأ
CAMERA_INDEX = 0

COLORS = {                # الألوان بنظام BGR
    ord("1"): ((0, 0, 255), "Red"),
    ord("2"): ((0, 255, 0), "Green"),
    ord("3"): ((255, 0, 0), "Blue"),
    ord("4"): ((0, 255, 255), "Yellow"),
}

# ===== نموذج الذكاء الاصطناعي اللي يعرف اليد =====
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# أرقام النقاط في اليد: (رأس الإصبع، المفصل اللي تحته)
INDEX = (8, 6)
MIDDLE = (12, 10)
RING = (16, 14)
PINKY = (20, 18)


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("ننزّل نموذج اليد (مرة وحدة بس)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("تم ✅")


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def finger_up(hand, finger):
    """الإصبع مرفوع لو رأسه أعلى من المفصل اللي تحته."""
    tip, joint = finger
    return hand[tip].y < hand[joint].y


def get_mode(hand):
    index = finger_up(hand, INDEX)
    middle = finger_up(hand, MIDDLE)
    others = finger_up(hand, RING) or finger_up(hand, PINKY)
    if index and not middle and not others:
        return "draw"
    if index and middle and not others:
        return "erase"
    return "idle"


def to_pixels(point, w, h):
    return int(point.x * w), int(point.y * h)


def main():
    download_model()
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ ما قدرت أفتح الكاميرا. تأكد إنك عطيت Terminal أو VS Code صلاحية الكاميرا:")
        print("   System Settings ← Privacy & Security ← Camera")
        return

    canvas = None
    color, color_name = COLORS[ord("1")]
    prev_point = None     # آخر نقطة رسمنا عندها
    smooth_point = None
    start = time.time()

    print("🎨 شغال! ارفع السبابة وارسم، إصبعين تمسح، q للخروج")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # زي المراية
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, int((time.time() - start) * 1000))

        mode = "idle"
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            mode = get_mode(hand)
            tip = to_pixels(hand[INDEX[0]], w, h)

            if mode == "draw":
                # تنعيم الحركة عشان الخط ما يرجف
                if smooth_point is None:
                    smooth_point = tip
                else:
                    smooth_point = (
                        int(smooth_point[0] * SMOOTHING + tip[0] * (1 - SMOOTHING)),
                        int(smooth_point[1] * SMOOTHING + tip[1] * (1 - SMOOTHING)),
                    )
                if prev_point is not None:
                    cv2.line(canvas, prev_point, smooth_point, color, PEN_SIZE)
                prev_point = smooth_point
                cv2.circle(frame, smooth_point, PEN_SIZE, color, -1)

            elif mode == "erase":
                # الممحاة بين رأس السبابة والوسطى
                mid = to_pixels(hand[MIDDLE[0]], w, h)
                center = ((tip[0] + mid[0]) // 2, (tip[1] + mid[1]) // 2)
                cv2.circle(canvas, center, ERASER_SIZE, (0, 0, 0), -1)
                cv2.circle(frame, center, ERASER_SIZE, (255, 255, 255), 2)

        if mode != "draw":
            prev_point = None   # نرفع القلم عشان ما يوصل خط بين رسمتين
            smooth_point = None

        # نحط الرسمة فوق صورة الكاميرا
        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
        frame[mask] = canvas[mask]

        labels = {"draw": "DRAW", "erase": "ERASE", "idle": "..."}
        cv2.putText(frame, f"{labels[mode]}  |  {color_name}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, "1-4 color  c clear  s save  q quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Air Drawing", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key == ord("s"):
            name = f"drawing_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(name, canvas)
            print(f"💾 انحفظت: {name}")
        elif key in COLORS:
            color, color_name = COLORS[key]

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
