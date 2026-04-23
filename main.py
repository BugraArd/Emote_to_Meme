"""
main.py — Emote to Meme | Ana Program

Gerçek zamanlı kamera görüntüsünden yüz ifadeleri ve vücut hareketlerini
tespit ederek eşleşen videoyu kamera görüntüsüne bindirerek oynatır.

Çalışma akışı:
  1. Her kare kameradan okunur ve ayna gibi çevrilir (daha doğal his için)
  2. Aynı kare hem yüz hem vücut analizine gönderilir (MediaPipe)
  3. İfade/hareket skorları hesaplanır (expressions.py, pose.py)
  4. Smoothing sayacı: aynı jest SMOOTHING_FRAMES kez üst üste tespit edilmesi
     gerekir — bu anlık yanlış tetiklemeleri önler
  5. Eşiği geçen jest için ilgili video oynatılır (animator.py)

Klavye kısayolları:
  [S] — sol üstteki skor panelini göster/gizle (kalibrasyonda kullanışlı)
  [L] — yüz ve vücut landmark noktalarını göster/gizle
  [Q] — uygulamayı kapat

Yeni jest eklemek için:
  1. assets/ klasörüne video dosyasını koy
  2. animator.py _load_all() içine jest_adi: "dosya.mp4" ekle
  3. counters listesine jest adını ekle
  4. determine_animation() veya pose.py detect_gesture() içine koşulunu yaz
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import expressions as expr
import pose as pose_mod
from animator import AnimationManager

# ─── Ayarlar ──────────────────────────────────────────────────────────────────

CAMERA_INDEX = 0                    # Birden fazla kameran varsa 1, 2 dene
FACE_MODEL   = "face_landmarker.task"
POSE_MODEL   = "pose_landmarker.task"

# Yüz ifadesi eşikleri — bu değerlerin altı "yok", üstü "var" sayılır.
# Blendshape skorları kişiden kişiye çok az değişir; genellikle değiştirmen gerekmez.
FACE_THRESHOLDS = {
    "smile":         0.45,   # 0.45 altı nötr/hafif gülümseme, üstü belirgin gülümseme
    "mouth_open":    0.50,   # 0.50 altı kapalı/hafif açık, üstü belirgin açık
    "eyebrow_raise": 0.50,   # 0.50 altı normal, üstü kaşlar kalkık
    "blink":         0.60,   # 0.60 altı açık/yarı kapalı, üstü göz kapalı (kırpma)
    "head_tilt_deg": 15,     # ±15 derece altı dik, üstü belirgin eğiklik
}

# Vücut hareketi eşikleri — pose.py içindeki detect_gesture'a iletilir.
POSE_THRESHOLDS = {
    "head_down":  0.5,
    "hand_raise": 0.6,
}

# Kaç üst üste kare aynı jesti göstermeli ki animasyon tetiklensin?
# 6 kare ≈ 30fps kamerada 0.2 saniye — çok hızlı ve kararlı hareketler için ideal.
# Daha "ağır" tetikleme istersen 10-12 yap, daha hızlı istersen 3-4 yap.
SMOOTHING_FRAMES = 6

# ─── MediaPipe kurulumu ───────────────────────────────────────────────────────

face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
        output_face_blendshapes=True,   # 52 kas hareketi skoru istiyoruz
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# ─── Uygulama durumu ──────────────────────────────────────────────────────────

anim_manager = AnimationManager(asset_dir="assets")

# Her jest için bağımsız sayaç. Sayaç SMOOTHING_FRAMES'e ulaşınca jest tetiklenir,
# jest bozulunca sayaç birer birer azalır — ani sıçramaları önler.
counters = {k: 0 for k in [
    "smile", "mouth_open", "eyebrow_raise", "blink",
    "tilt_left", "tilt_right",
    "salud", "hand_raise_left", "hand_raise_right", "i_dont_know", "peace",
]}

show_scores    = True
show_landmarks = False

# EMA (Exponential Moving Average) ile oynak skorları yumuşat.
# Formül: yeni = alpha * ham_değer + (1-alpha) * önceki_değer
# alpha = 0.25: her kare %25 yeni, %75 geçmiş — stabil ama gecikmeli tepki
# alpha = 0.50: daha hızlı ama daha sallantılı
ema_scores = {}
EMA_ALPHA  = 0.25


def determine_animation(face_scores, pose_scores):
    """
    Yüz ve vücut skorlarına bakarak tetiklenecek jest adını döndürür.
    Vücut hareketleri (pose) yüz ifadelerine göre daha önce kontrol edilir —
    çünkü vücut jestleri genellikle daha bilinçli ve kasıtlı hareketlerdir.
    """
    FT = FACE_THRESHOLDS

    # Vücut jestlerini önce kontrol et (pose.py içinde kendi öncelik sırası var)
    pose_gesture = pose_mod.detect_gesture(pose_scores, POSE_THRESHOLDS)
    if pose_gesture:
        return pose_gesture

    # Yüz ifadelerini kontrol et
    if face_scores["smile"] > FT["smile"]:
        return "smile"

    # Şaşırma: kaş kaldırma VE ağız açma birlikte olmalı
    if face_scores["eyebrow_raise"] > FT["eyebrow_raise"] and face_scores["mouth_open"] > FT["mouth_open"]:
        return "surprise"

    if face_scores["mouth_open"] > FT["mouth_open"]:
        return "mouth_open"

    # Kırpma: gülümseme yokken kontrol edilir çünkü gülerken yanaklar gözü
    # biraz kapatır ve yanlış kırpma tespitine yol açabilir
    if face_scores["blink"] > FT["blink"] and face_scores["smile"] < 0.2:
        return "blink"

    tilt = face_scores["head_tilt"]
    if tilt < -FT["head_tilt_deg"]:
        return "tilt_left"
    if tilt > FT["head_tilt_deg"]:
        return "tilt_right"

    return None


def draw_score_panel(frame, face_scores, pose_scores, active_anim):
    """
    Sol üst köşeye yarı saydam skor paneli çizer.
    Geliştirme ve kalibrasyon sırasında hangi jestlerin ne kadar güçlü
    algılandığını görmek için [S] tuşuyla açılıp kapatılabilir.
    """
    panel_w, panel_h = 280, 305
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    face_items = [
        ("Gulumse",    face_scores["smile"],         (50, 220, 50)),
        ("Agiz acik",  face_scores["mouth_open"],    (220, 150, 50)),
        ("Kas kaldir", face_scores["eyebrow_raise"], (50, 150, 220)),
        ("Kirpma",     face_scores["blink"],         (150, 50, 220)),
    ]
    for i, (label, val, color) in enumerate(face_items):
        y = 22 + i * 30
        cv2.putText(frame, f"{label}:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        bx, by = 110, y - 11
        cv2.rectangle(frame, (bx, by), (bx + 110, by + 13), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(val * 110), by + 13), color, -1)
        cv2.putText(frame, f"{val:.2f}", (bx + 114, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.line(frame, (5, 135), (panel_w - 5, 135), (80, 80, 80), 1)

    if pose_scores:
        pose_items = [
            ("Sol el",    pose_scores["left_hand_raise"],  (50, 220, 180)),
            ("Sag el",    pose_scores["right_hand_raise"], (220, 180, 50)),
            ("Yana acik", pose_scores["side_open"],        (50, 200, 220)),
            ("Bas asagi", pose_scores["head_down"],        (180, 50, 220)),
            ("Peace",     pose_scores["peace"],            (220, 220, 50)),
        ]
        for i, (label, val, color) in enumerate(pose_items):
            y = 152 + i * 28
            cv2.putText(frame, f"{label}:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            bx, by = 110, y - 11
            cv2.rectangle(frame, (bx, by), (bx + 110, by + 13), (50, 50, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(val * 110), by + 13), color, -1)
            cv2.putText(frame, f"{val:.2f}", (bx + 114, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    else:
        cv2.putText(frame, "Vucut: kamera uzak?", (8, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 200), 1)

    return frame


def main():
    global show_scores, show_landmarks, counters

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"HATA: Kamera {CAMERA_INDEX} acilamadi! Farkli bir indeks deneyin.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Kamera acildi!")
    print("Tuslar: [S] skor paneli, [L] landmark noktalari, [Q] cikis\n")

    active_anim = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # ayna görüntüsü — daha doğal hissettiriyor
        h, w  = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_result = face_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)

        face_scores = None
        pose_scores = None

        if face_result.face_landmarks:
            face_lm     = face_result.face_landmarks[0]
            blendshapes = face_result.face_blendshapes[0]
            face_scores = expr.analyze(face_lm, blendshapes, w, h)
            if show_landmarks:
                for lm in face_lm:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

        if pose_result.pose_landmarks:
            pose_lm     = pose_result.pose_landmarks[0]
            pose_scores = pose_mod.analyze(pose_lm)

            # Oynak vücut skorlarını EMA filtresiyle yumuşat
            for key in ("peace", "side_open", "any_hand_raise", "head_down",
                        "left_hand_raise", "right_hand_raise"):
                raw = pose_scores.get(key, 0.0)
                prev = ema_scores.get(key, raw)
                ema_scores[key] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * prev
                pose_scores[key] = ema_scores[key]

            if show_landmarks:
                for lm in pose_lm:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 100, 255), -1)

        if face_scores or pose_scores:
            fs       = face_scores or {"smile": 0, "mouth_open": 0, "eyebrow_raise": 0, "blink": 0, "head_tilt": 0}
            triggered = determine_animation(fs, pose_scores)

            for key in counters:
                if triggered == key:
                    counters[key] = min(counters[key] + 1, SMOOTHING_FRAMES * 2)
                else:
                    counters[key] = max(counters[key] - 1, 0)

            best = max(counters, key=lambda k: counters[k])
            if counters[best] >= SMOOTHING_FRAMES:
                active_anim = best
                anim_manager.trigger(best)
            else:
                active_anim = None
                anim_manager.clear()
        else:
            anim_manager.clear()
            active_anim = None

        frame = anim_manager.render(frame)

        if show_scores and face_scores:
            frame = draw_score_panel(frame, face_scores, pose_scores, active_anim)

        if active_anim:
            cv2.putText(frame, f">> {active_anim}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

        cv2.imshow("Emote to Meme", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            show_scores = not show_scores
        elif key == ord("l"):
            show_landmarks = not show_landmarks

    cap.release()
    anim_manager.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()


if __name__ == "__main__":
    main()
