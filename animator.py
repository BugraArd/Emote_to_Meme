"""
animator.py — Video oynatma ve ses yönetimi

Her jest için assets/ klasöründeki bir MP4 dosyasını kamera görüntüsünün
sağ üst köşesine bindirerek oynatır. Ses pygame ile eş zamanlı çalınır.

Çalışma mantığı:
  1. Başlangıçta tüm video dosyaları yüklenir ve sesleri temp klasörüne çıkarılır.
  2. Bir jest tespit edilince o jest için VideoOverlay.trigger() çağrılır:
       - Video başa alınır (ilk kareye döner)
       - Ses baştan çalmaya başlar
  3. Jest biterken opacity yavaşça 0'a düşer (fade-out efekti).
  4. Video dosyası yoksa (assets/ içinde bulunmuyorsa) sessizce atlanır.

Yeni jest eklemek için:
  - assets/ klasörüne MP4 dosyasını koy
  - _load_all() içindeki mapping sözlüğüne jest_adı: "dosyaadi.mp4" satırı ekle
  - main.py'deki counters ve determine_animation'a da jest adını ekle
"""

import cv2
import numpy as np
import os
import tempfile
import pygame

pygame.mixer.init()


def extract_audio(video_path):
    """
    Videodan ses parçasını bir geçici MP3 dosyasına çıkarır.
    moviepy kütüphanesi bu işlem için kullanılır (içinde ffmpeg bulunur).
    Videoda ses yoksa veya hata oluşursa None döner — uygulama sessiz devam eder.
    """
    try:
        from moviepy import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        clip.audio.write_audiofile(tmp.name, logger=None)
        clip.close()
        return tmp.name
    except Exception as e:
        print(f"Ses cikarilamadi ({os.path.basename(video_path)}): {e}")
        return None


class VideoOverlay:
    """
    Tek bir video dosyasını temsil eder.

    Özellikler:
      - trigger() çağrılınca video başa döner ve ses başlar
      - stop() çağrılınca ses durur, opacity yavaş yavaş azalır (fade-out)
      - render() her karede bir sonraki video karesini kamera görüntüsüne karıştırır
      - opacity: 0.0 = tamamen görünmez, 1.0 = tam opak
        Her render çağrısında 0.15 adımla artar/azalır (yaklaşık 7 karede tam geçiş)
    """

    def __init__(self, path, size=(300, 300), position="top-right"):
        self.path     = path
        self.size     = size
        self.position = position
        self.cap      = None
        self.active   = False
        self.opacity  = 0.0
        self.audio_path = None
        self.channel  = None

        if os.path.exists(path):
            self.cap = cv2.VideoCapture(path)
            self.audio_path = extract_audio(path)
            status = "video + ses" if self.audio_path else "video (ses yok)"
            print(f"  Yuklendi ({status}): {os.path.basename(path)}")
        else:
            print(f"  Atlandi (dosya yok): {os.path.basename(path)}")

    def trigger(self):
        """Videoyu ve sesi en baştan başlatır."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.active = True
        if self.audio_path:
            sound = pygame.mixer.Sound(self.audio_path)
            self.channel = sound.play()

    def stop(self):
        """Videoyu durdurur. Görsel fade-out render() içinde gerçekleşir."""
        self.active = False
        if self.channel:
            self.channel.stop()
            self.channel = None

    def render(self, frame):
        """
        Aktif durumdaysa kamera karesinin üstüne video karesi bindirme yapar.

        Opacity artış/azalış:
          - active=True  → her karede +0.15 (yaklaşık 7 karede tam görünür)
          - active=False → her karede -0.15 (yaklaşık 7 karede tamamen kaybolur)

        Video bitince başa döner (döngülü oynatma), jest devam ettiği sürece oynar.
        """
        if self.active:
            self.opacity = min(1.0, self.opacity + 0.15)
        else:
            self.opacity = max(0.0, self.opacity - 0.15)

        if self.opacity <= 0 or self.cap is None:
            return frame

        ret, vid_frame = self.cap.read()
        if not ret:
            # Video sona erdi, başa sar ve tekrar dene
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, vid_frame = self.cap.read()
            if not ret:
                return frame

        vid_frame = cv2.resize(vid_frame, self.size)

        h, w = frame.shape[:2]
        vw, vh = self.size
        margin = 20
        positions = {
            "top-right":    (w - vw - margin, margin),
            "top-left":     (margin, margin),
            "center":       (w // 2 - vw // 2, h // 2 - vh // 2),
            "bottom-right": (w - vw - margin, h - vh - margin),
            "bottom-left":  (margin, h - vh - margin),
        }
        x, y = positions.get(self.position, positions["top-right"])

        # Ekran kenarını taşan kısmı kırp
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + vw, w), min(y + vh, h)
        if x2 <= x1 or y2 <= y1:
            return frame

        vid_crop = vid_frame[(y1 - y):(y1 - y) + (y2 - y1), (x1 - x):(x1 - x) + (x2 - x1)]
        roi      = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(vid_crop, self.opacity, roi, 1 - self.opacity, 0)

        return frame

    def release(self):
        """Kaynakları serbest bırakır. Uygulama kapanırken çağrılır."""
        if self.cap:
            self.cap.release()
        if self.channel:
            self.channel.stop()
        if self.audio_path and os.path.exists(self.audio_path):
            os.remove(self.audio_path)


class AnimationManager:
    """
    Tüm jest-video eşleştirmelerini yönetir.

    Aynı anda yalnızca bir video aktif olabilir.
    Jest değiştiğinde eski video durdurulur, yeni video baştan başlatılır.
    """

    def __init__(self, asset_dir="assets"):
        self.asset_dir   = asset_dir
        self.videos      = {}
        self.active_name = None
        print("Videolar yukleniyor...")
        self._load_all()
        print("Yukleme tamamlandi.\n")

    def _load_all(self):
        # Jest adı → MP4 dosya adı eşleştirmesi.
        # Dosya yoksa VideoOverlay sessizce atlar, uygulama çalışmaya devam eder.
        mapping = {
            "smile":            "Smile.mp4",
            "surprise":         "surprise.mp4",
            "blink":            "blink.mp4",
            "mouth_open":       "mouth_open.mp4",
            "tilt_left":        "tilt_left.mp4",
            "tilt_right":       "tilt_right.mp4",
            "salud":            "Salud.mp4",
            "hand_raise_left":  "hand_raise.mp4",
            "hand_raise_right": "hand_raise.mp4",
            "i_dont_know":      "I dont know.mp4",
            "peace":            "Peace.mp4",
        }
        for name, filename in mapping.items():
            path = os.path.join(self.asset_dir, filename)
            self.videos[name] = VideoOverlay(path, size=(300, 300), position="top-right")

    def trigger(self, name):
        """
        Belirtilen jesti aktif eder.
        Farklı bir jest gelirse öncekini durdurur ve yenisini baştan başlatır.
        Aynı jest devam ediyorsa tekrar başlatmaz (zaten oynuyor).
        """
        if name not in self.videos:
            return
        if self.active_name != name:
            if self.active_name and self.active_name in self.videos:
                self.videos[self.active_name].stop()
            self.videos[name].trigger()
            self.active_name = name

    def clear(self):
        """Aktif jesti durdurur (jest bitti, fade-out başlar)."""
        if self.active_name and self.active_name in self.videos:
            self.videos[self.active_name].stop()
        self.active_name = None

    def render(self, frame, position=None):
        """Aktif olan veya henüz fade-out tamamlanmamış videoları çizer."""
        for vid in self.videos.values():
            if vid.active or vid.opacity > 0:
                frame = vid.render(frame)
        return frame

    def release(self):
        """Tüm video ve ses kaynaklarını serbest bırakır."""
        for vid in self.videos.values():
            vid.release()
