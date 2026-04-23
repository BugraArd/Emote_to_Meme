# Emote to Meme

Kameradan yüz ifadelerini ve vücut hareketlerini gerçek zamanlı olarak tanıyıp, her harekete karşılık gelen videoyu ekrana yansıtan bir proje.

---

## Kurulum

**1. Gerekli kütüphaneleri yükle:**
```bash
pip install opencv-python mediapipe pillow numpy pygame moviepy
```

**2. MediaPipe model dosyalarını indir:**
```bash
python setup.py
```

**3. Çalıştır:**
```bash
python main.py
```

---

## Hareketler ve Tetikleyiciler

| Video | Hareket | Açıklama |
|---|---|---|
| **Smile.mp4** | Gülümsemek | Ağız köşelerini yukarı çekecek şekilde belirgin bir şekilde gülümsemek yeterli. Hafif gülümseme tetiklemeyebilir, biraz abartmak gerekebilir. |
| **Salud.mp4** | Başı öne eğmek + bir eli kaldırmak | Hafifçe selam verir gibi başını öne eğ ve aynı anda sağ ya da sol elini omzunun üstüne kaldır. İki el birden kalkmamalı. |
| **I dont know.mp4** | İki eli yana açmak | Başı dik tutarken her iki kolu yana açık bir şekilde tut. Ellerin omuz hizasında ya da biraz üstünde olması yeterli. |
| **Peace.mp4** | İki eli yüzün önünde birleştirmek | İki elini düz bir şekilde birbirinin üstüne koy ve çenenin altına, yüzünün önüne getir. Ellerin birbirine yakın ve vücudun ortasında olması gerekiyor. |

---

## Klavye Kısayolları

| Tuş | İşlev |
|---|---|
| `S` | Sol üstteki skor panelini aç / kapat |
| `L` | Yüz ve vücut landmark noktalarını göster / gizle |
| `Q` | Uygulamayı kapat |

---

## Yeni Hareket Eklemek

1. `assets/` klasörüne MP4 dosyasını koy
2. `animator.py` içindeki `_load_all()` fonksiyonuna şu satırı ekle:
   ```python
   "jest_adi": "dosyaadi.mp4",
   ```
3. `main.py` içindeki `counters` listesine jest adını ekle
4. Yüz ifadesi ekleyeceksen `main.py` → `determine_animation()` fonksiyonuna,  
   vücut hareketi ekleyeceksen `pose.py` → `detect_gesture()` fonksiyonuna koşulunu yaz

---

## Proje Yapısı

```
├── main.py          # Ana döngü: kamera, analiz, animasyon koordinasyonu
├── expressions.py   # Yüz ifadesi skorları (MediaPipe blendshape)
├── pose.py          # Vücut hareketi skorları ve jest tespiti
├── animator.py      # Video oynatma ve ses yönetimi
├── setup.py         # Model dosyalarını indirme scripti
└── assets/          # MP4 video dosyaları buraya konur
```

---

## Teknik Altyapı

- **MediaPipe FaceLandmarker** — 52 yüz kası hareketi (blendshape) ile gülümseme, göz kırpma, kaş kaldırma gibi ifadeleri tespit eder
- **MediaPipe PoseLandmarker** — 33 vücut iskelet noktasından el kaldırma, baş eğme, yana açılma gibi hareketleri hesaplar
- **EMA Filtresi** — Vücut skorlarındaki anlık sıçramaları yumuşatmak için üssel hareketli ortalama uygulanır
- **Smoothing Sayacı** — Yanlış tetiklemeleri önlemek için aynı jest 6 üst üste kare boyunca görülmesi gerekir
