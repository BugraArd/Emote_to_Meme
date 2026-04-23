"""
expressions.py — Yüz ifadesi analiz modülü

MediaPipe FaceLandmarker'ın "blendshape" çıktılarını kullanır.
Blendshape, MediaPipe'ın kendi sinir ağının yüzdeki 52 farklı kas hareketini
0.0 ile 1.0 arasında puanlamasıdır. 0.0 tamamen hareketsiz, 1.0 tam hareket demektir.

Neden koordinat matematiği yerine blendshape?
  Eski yöntemde 468 landmark noktasının koordinatlarından elle hesaplama yapılırdı
  (ör. ağız köşelerinin Y konumu). Bu yöntem kişiden kişiye ve kamera açısına göre
  büyük farklılıklar gösterirdi. Blendshape ise modelin içindeki derin öğrenme
  katmanlarından geldiği için kişi, ışık ve açıdan bağımsız çok daha tutarlı sonuç verir.

Baş eğikliği (head_tilt) ise blendshape'de bulunmadığı için hâlâ iki landmark
noktası arasındaki trigonometrik açıdan hesaplanır.
"""

import math


def get_blendshape(blendshapes, name):
    """
    Blendshape listesinden belirtilen ismi arar ve skorunu döndürür.
    İsim bulunamazsa 0.0 döner — bu MediaPipe versiyonları arası farkı güvenli atlatır.
    """
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return 0.0


def analyze(face_landmarks, blendshapes, image_width, image_height):
    """
    Tek bir karedeki yüz için tüm ifade skorlarını hesaplar ve dict olarak döndürür.
    Tüm skorlar 0.0-1.0 arasındadır (head_tilt derece cinsindendir, farklıdır).

    Parametreler:
        face_landmarks  : MediaPipe'ın döndürdüğü normalize landmark listesi
        blendshapes     : MediaPipe'ın 52 kas hareketi skoru listesi
        image_width/height: Piksel koordinatına çevirmek için kare boyutları
    """

    # Gülümseme skoru: sol ve sağ ağız köşesinin yukarı çekilme miktarının ortalaması.
    # 0.0 = düz/nötr yüz, 0.45+ = belirgin gülümseme, 0.80+ = büyük sırıtma.
    # İki tarafın ortalaması alınır çünkü asimetrik gülümsemeler (tek taraflı) de
    # yakalanabilsin diye her iki köşe ayrı ayrı puanlanır.
    smile = (
        get_blendshape(blendshapes, "mouthSmileLeft") +
        get_blendshape(blendshapes, "mouthSmileRight")
    ) / 2

    # Ağız açıklığı: alt çenenin aşağı inmesini ölçer.
    # 0.0 = ağız kapalı, 0.50+ = ağız belirgin açık, 0.80+ = çok geniş açık.
    # Şaşırma ifadesinde kaş kaldırmayla birlikte kullanılır.
    mouth_open = get_blendshape(blendshapes, "jawOpen")

    # Kaş kaldırma: iç kaş (browInnerUp) ve dış kaşların (browOuterUp) ortalaması.
    # 0.0 = kaşlar normal, 0.50+ = kaşlar belirgin yukarıda.
    # Tek başına şaşkınlık değil, ağız açıklığıyla birleşince "şaşırma" jest olur.
    eyebrow_raise = (
        get_blendshape(blendshapes, "browInnerUp") +
        get_blendshape(blendshapes, "browOuterUpLeft") +
        get_blendshape(blendshapes, "browOuterUpRight")
    ) / 3

    # Göz kırpma: her iki göz kapağının kapanma miktarının ortalaması.
    # 0.0 = gözler tam açık, 0.60+ = göz büyük olasılıkla kapalı (kırpma).
    # Gülümserken yanaklar gözü biraz kapatabilir; bu yüzden kırpma kontrolünde
    # gülümseme skoru düşük olmalı şartı da aranır (bkz. main.py determine_animation).
    blink = (
        get_blendshape(blendshapes, "eyeBlinkLeft") +
        get_blendshape(blendshapes, "eyeBlinkRight")
    ) / 2

    # Baş eğikliği (sola/sağa): burun ucu (nokta 1) ile burun kökü (nokta 168)
    # arasındaki ekseni dikey referansa göre ölçeriz.
    # Pozitif derece = sağa eğik, negatif = sola eğik.
    # 15 derece üzeri belirgin eğiklik olarak kabul edilir (bkz. FACE_THRESHOLDS).
    lm = face_landmarks
    w, h = image_width, image_height
    nose_tip  = (lm[1].x * w,   lm[1].y * h)
    nose_base = (lm[168].x * w, lm[168].y * h)
    dx = nose_tip[0] - nose_base[0]
    dy = nose_tip[1] - nose_base[1]
    head_tilt = math.degrees(math.atan2(dx, -dy))

    return {
        "smile":         smile,
        "mouth_open":    mouth_open,
        "eyebrow_raise": eyebrow_raise,
        "blink":         blink,
        "head_tilt":     head_tilt,
    }
