"""
setup.py — Gerekli model dosyalarını indirir.

Projeyi ilk kez çalıştırmadan önce bu scripti bir kere çalıştır:
    python setup.py

Ne indirir?
  - face_landmarker.task : Yüz ifadesi tespiti için MediaPipe modeli (~30MB)
  - pose_landmarker.task : Vücut iskelet tespiti için MediaPipe modeli (~5MB)
"""

import urllib.request
import os

models = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

for filename, url in models.items():
    if os.path.exists(filename):
        print(f"Zaten mevcut: {filename}")
    else:
        print(f"Indiriliyor: {filename} ...")
        urllib.request.urlretrieve(url, filename)
        print(f"Tamam: {filename}")

print("\nKurulum tamamlandi! Simdi 'python main.py' ile calistirabilirsin.")
