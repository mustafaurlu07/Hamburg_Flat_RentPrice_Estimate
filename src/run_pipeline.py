import subprocess

scripts = [
    "veri_temizleme.py",
    "feature_engineering.py",
    "model_gelistirme.py",
    "best_model_optimize.py",
]

for script in scripts:
    print(f"🔹 Çalıştırılıyor: {script}")
    subprocess.run(["python", script], check=True)

print("✅ Tüm pipeline başarıyla tamamlandı!")
