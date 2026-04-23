"""
pose.py — Vücut hareketi analiz ve jest tespit modülü

MediaPipe PoseLandmarker'ın 33 vücut iskelet noktasını kullanır.
Koordinatlar normalize (0.0-1.0) formattadır: x soldan sağa, y yukarıdan aşağıya büyür.

Kullanılan kritik nokta indeksleri:
  0  = burun ucu (baş pozisyonu referansı)
  11 = sol omuz,  12 = sağ omuz
  15 = sol bilek, 16 = sağ bilek

Tüm mesafe hesaplamaları "omuz genişliği"ne normalize edilir.
Bu sayede kameraya yakın duran biri ile uzak duran biri için
eşikler otomatik olarak ölçeklenir.

Tespit edilen jestler:
  - peace       : İki el birleşik yüz önünde
  - salud       : Baş öne eğik + tek el yukarıda
  - i_dont_know : Baş dik + iki el yana açık
  - hand_raise  : Baş dik + tek el düz yukarıda
"""


def analyze(pose_landmarks):
    """
    Bir karedeki vücut landmark listesinden tüm hareket skorlarını hesaplar.
    Landmark yoksa (vücut kamerada görünmüyorsa) None döner.
    """
    if not pose_landmarks:
        return None

    lm = pose_landmarks
    nose       = lm[0]
    l_shoulder = lm[11]
    r_shoulder = lm[12]
    l_wrist    = lm[15]
    r_wrist    = lm[16]

    shoulder_y    = (l_shoulder.y + r_shoulder.y) / 2
    shoulder_x    = (l_shoulder.x + r_shoulder.x) / 2
    shoulder_span = abs(r_shoulder.x - l_shoulder.x) + 1e-6  # sıfıra bölmeyi önler

    # ── El yukarı kaldırma skoru ────────────────────────────────────────────────
    # Bileğin omuz hizasına göre dikey yüksekliğini ölçeriz.
    # Y ekseni aşağı doğru büyüdüğü için bilek.y < omuz.y ise el yukarıdadır.
    # Formül: (omuz_y - bilek_y) / 0.3
    #   → Bilek omuzla aynı hizada (0 fark): skor = 0.0
    #   → Bilek omuzun tam 0.3 birim yukarısında: skor = 1.0
    #   → 0.30 skoru kabaca "el omzun yarı yukarısında" demektir
    #   → 0.65+ skoru "el açıkça omuzun üstünde, salud/selamlama pozisyonu"
    l_hand_raise = min(1.0, max(0.0, (shoulder_y - l_wrist.y) / 0.3))
    r_hand_raise = min(1.0, max(0.0, (shoulder_y - r_wrist.y) / 0.3))

    # ── Elleri yana açma skoru ──────────────────────────────────────────────────
    # Kamera görüntüsü ayna gibi çevrildiğinde (flip) sol/sağ koordinatları yer
    # değiştirebilir. Bu yüzden yön yerine omuz merkezine olan mutlak yatay
    # uzaklığı kullanırız — flip'ten bağımsız çalışır.
    #
    # l_dist: sol bileğin omuz merkezinden normalize uzaklığı (omuz genişliği birimi)
    #   1.0 = bilek omuzla aynı genişlikte dışarıda (orta açıklık)
    #   1.5+ = bilek omuzdan belirgin daha dışarıda (geniş açıklık)
    #
    # l_side skoru formülü: (l_dist - 0.5) / 0.5
    #   → l_dist = 0.5 (bilek omuz ucunda): skor = 0.0  — henüz yana açılmamış
    #   → l_dist = 1.0 (bilek omuz genişliği kadar dışarda): skor = 1.0
    #   → 0.35+ skor eşiği: her iki el de omuzdan belirgin dışarıda = yana açık jest
    l_dist = abs(l_wrist.x - shoulder_x) / shoulder_span
    r_dist = abs(r_wrist.x - shoulder_x) / shoulder_span
    l_side = min(1.0, max(0.0, (l_dist - 0.5) / 0.5))
    r_side = min(1.0, max(0.0, (r_dist - 0.5) / 0.5))

    # Her iki el de 0.35 üzerindeyse "iki el yana açık" sayarız.
    # Sadece bir el açıksa bu jest değil, o yüzden min() alırız:
    # min(l_side, r_side) ile her ikisinin de yeterince açık olması şartı koşulur.
    both_side_open  = l_side > 0.35 and r_side > 0.35
    side_open_score = min(l_side, r_side) if both_side_open else 0.0

    # ── Baş öne eğme skoru ──────────────────────────────────────────────────────
    # Burnun omuz hizasına ne kadar yaklaştığını ölçeriz.
    # Normal dik duruşta burun omuzların çok üzerindedir (nose.y << shoulder_y).
    # Baş öne eğildikçe burun aşağı iner ve omuz Y değerine yaklaşır.
    #
    # Formül: (nose.y - (shoulder_y - 0.15)) / 0.15
    #   → Burun omuzun 0.15 birim üzerinde (dik duruş): skor ≈ 0.0
    #   → Burun omuz hizasında: skor ≈ 1.0
    #   → 0.30 skor eşiği: belirgin ama abartılı olmayan bir eğilme (salud için yeterli)
    #   → 0.65+ skor: çok belirgin öne eğilme
    head_down = min(1.0, max(0.0, (nose.y - (shoulder_y - 0.15)) / 0.15))

    # ── Peace jesti skoru ───────────────────────────────────────────────────────
    # Üç koşulun çarpımıdır; herhangi biri sıfırsa toplam skor sıfır olur:
    #
    # 1. wrist_close: İki bilek birbirine ne kadar yakın?
    #    wrist_dist = bilek arası mesafe / omuz genişliği
    #    0.25-0.35 arası: eller birleşik (iyi peace pozisyonu)
    #    0.42+: eller çok açık, muhtemelen peace değil
    #    Formül: (0.42 - wrist_dist) / 0.18 → mesafe azaldıkça skor artar
    #
    # 2. centered_x: Bileklerin ortası vücudun yatay merkezinde mi?
    #    ±0.01-0.05 fark: mükemmel ortalama → skor ≈ 1.0
    #    ±0.06+ fark: eller bir yana kaymış → skor düşer
    #
    # 3. at_face_y: Eller yüzün önünde mi (dikey konum)?
    #    Gerçek test verileri: bilek omuzdan yaklaşık 0.22 birim aşağıda
    #    (yani göğüs üstü / çene altı bölgesi). Bu değerden uzaklaştıkça skor düşer.
    #    ±0.12 tolerans: eller hafif yukarı/aşağı olsa da geçer
    wrist_mid_x = (l_wrist.x + r_wrist.x) / 2
    wrist_mid_y = (l_wrist.y + r_wrist.y) / 2

    wrist_dist  = abs(l_wrist.x - r_wrist.x) / shoulder_span
    wrist_close = min(1.0, max(0.0, (0.42 - wrist_dist) / 0.18))
    centered_x  = 1.0 - min(1.0, abs(wrist_mid_x - shoulder_x) / 0.06)
    diff_from_shoulder = wrist_mid_y - shoulder_y
    at_face_y   = 1.0 - min(1.0, max(0.0, abs(diff_from_shoulder - 0.22) / 0.12))

    peace_score = wrist_close * centered_x * at_face_y

    return {
        "left_hand_raise":  l_hand_raise,
        "right_hand_raise": r_hand_raise,
        "any_hand_raise":   max(l_hand_raise, r_hand_raise),
        "side_open":        side_open_score,
        "head_down":        head_down,
        "peace":            peace_score,
    }


def detect_gesture(pose_scores, thresholds):
    """
    Hesaplanan vücut skorlarından hangi jestin yapıldığını belirler.

    Her jestin matematiksel olarak çakışmaması için eşikler birbirini dışlar:
      - Salud: baş > 0.30 ZORUNLU, tek el yukarıda, eller yana değil
      - I don't know: baş < 0.20 ZORUNLU, her iki el yana açık
      - Hand raise: baş < 0.35, tek el yukarı, yana açık değil
      - Peace: her zaman önce kontrol edilir (baş dik + eller birleşik önde)

    Öncelik sırası: peace > salud > i_dont_know > hand_raise
    Hiçbiri tetiklenmezse None döner.
    """
    if pose_scores is None:
        return None

    head_down = pose_scores["head_down"]
    any_raise = pose_scores["any_hand_raise"]
    l_raise   = pose_scores["left_hand_raise"]
    r_raise   = pose_scores["right_hand_raise"]
    side_open = pose_scores["side_open"]
    peace     = pose_scores["peace"]

    # ── PEACE ───────────────────────────────────────────────────────────────────
    # İki el yüz önünde birleşik durmalı (peace skoru ≥ 0.30) ve baş dik olmalı.
    # Peace skoru 0.30 altına düşerse jest bitti sayılır, animasyon durur.
    is_peace = (
        peace > 0.30 and
        head_down < 0.35
    )

    # ── SALUD ───────────────────────────────────────────────────────────────────
    # Bir eliyle selam verme/veda jesti.
    # Baş hafifçe öne eğik (0.30+): tam dik duruşta 0.0, belirgin eğilişte 0.60+
    # Sadece bir el yukarıda: != operatörü XOR mantığı sağlar
    #   → sol el 0.30 üzeri, sağ el 0.30 altı: True (sadece sol)
    #   → her ikisi 0.30 üzeri: False (ikisi birden = salud değil)
    # Eller yana açık değil (side_open < 0.35): elleri yana değil, düz yukarı
    one_hand_up = (l_raise > 0.30) != (r_raise > 0.30)
    is_salud = (
        head_down > 0.30 and
        one_hand_up and
        side_open < 0.35
    )

    # ── I DON'T KNOW ────────────────────────────────────────────────────────────
    # "Bilmiyorum" / omuz silkme jesti.
    # Baş kesinlikle dik olmalı (0.20 altı): 0.20 üstü baş eğikliği salud ile karışır
    # Her iki el yana açık (side_open > 0.45): tek el açıksa jest sayılmaz
    # Eller hafifçe yukarıda da olabilir (any_raise > 0.10): tamamen aşağı sarkık
    # eller de kabul edilir, biraz yukarı kalkık olan da
    is_i_dont_know = (
        head_down < 0.20 and
        side_open > 0.45 and
        any_raise > 0.10
    )

    # ── TEK EL KALDIRMA ─────────────────────────────────────────────────────────
    # Düz yukarı uzanmış tek el jesti (salud'dan farkı: baş dik).
    # 0.65+ el yüksekliği: el açıkça omuzun üzerinde, hafif kaldırma sayılmaz
    # side_open < 0.35: el yana değil, düz yukarı
    is_hand_raise = (
        any_raise > 0.65 and
        head_down < 0.35 and
        side_open < 0.35
    )

    if is_peace:
        return "peace"
    if is_salud:
        return "salud"
    if is_i_dont_know:
        return "i_dont_know"
    if is_hand_raise:
        return "hand_raise_left" if l_raise > r_raise else "hand_raise_right"

    return None
