"""
كاشف النعاس 😴
يراقبك بكاميرا الماك، ولو غمضت عيونك أكثر من ثانيتين يصفّر لك.

التشغيل:
    python3 drowsiness_detector.py
للخروج: اضغط حرف q
"""

import os
import subprocess
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ===== الإعدادات (تقدر تغيرها) =====
CLOSED_THRESHOLD = 0.5    # فوق هالرقم = العين مغمضة (من 0 إلى 1)
SECONDS_TO_ALARM = 2.0    # كم ثانية مغمض قبل ما يصفّر
ALARM_SOUND = "/System/Library/Sounds/Sosumi.aiff"
CAMERA_INDEX = 0          # 0 = كاميرا الماك

# ===== نموذج الذكاء الاصطناعي اللي يعرف ملامح الوجه =====
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("ننزّل نموذج الذكاء الاصطناعي (مرة وحدة بس)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("تم ✅")


def create_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,  # يعطينا درجة غمضة كل عين
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def eyes_closed_score(result):
    """ترجع رقم من 0 (مفتوحة) إلى 1 (مغمضة)، أو None لو ما فيه وجه."""
    if not result.face_blendshapes:
        return None
    scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
    return (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2


def draw_ui(frame, score, closed_for, alarm_on, naps):
    h, w = frame.shape[:2]

    if alarm_on:
        # إطار أحمر يرمش
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 25)
        cv2.putText(frame, "WAKE UP!", (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 6)

    if score is None:
        status, color = "No face", (0, 200, 255)
    elif score > CLOSED_THRESHOLD:
        status, color = f"Eyes CLOSED  {closed_for:.1f}s", (0, 0, 255)
    else:
        status, color = "Eyes open", (0, 200, 0)

    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"Naps: {naps}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # شريط يوضح كم العين مغمضة
    if score is not None:
        bar_w = int(300 * score)
        cv2.rectangle(frame, (20, 115), (320, 140), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, 115), (20 + bar_w, 140), color, -1)
        x = 20 + int(300 * CLOSED_THRESHOLD)
        cv2.line(frame, (x, 110), (x, 145), (255, 255, 255), 2)


def main():
    download_model()
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ ما قدرت أفتح الكاميرا. تأكد إنك عطيت Terminal أو VS Code صلاحية الكاميرا:")
        print("   System Settings ← Privacy & Security ← Camera")
        return

    closed_since = None   # متى بدت العين تغمض
    alarm_process = None  # صوت التنبيه الشغال
    naps = 0              # كم مرة نعست
    alarm_on = False
    start = time.time()

    print("👀 شغال! اضغط q للخروج")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # زي المراية

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, int((time.time() - start) * 1000))
        score = eyes_closed_score(result)

        now = time.time()
        if score is not None and score > CLOSED_THRESHOLD:
            if closed_since is None:
                closed_since = now
            closed_for = now - closed_since
        else:
            closed_since = None
            closed_for = 0.0
            alarm_on = False

        if closed_for >= SECONDS_TO_ALARM:
            if not alarm_on:
                naps += 1
                alarm_on = True
            # يعيد الصوت لين تفتح عيونك
            if alarm_process is None or alarm_process.poll() is not None:
                alarm_process = subprocess.Popen(["afplay", ALARM_SOUND])

        draw_ui(frame, score, closed_for, alarm_on, naps)
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"انتهى. نعست {naps} مرة 😴")


if __name__ == "__main__":
    main()
