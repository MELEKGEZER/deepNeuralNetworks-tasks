import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Videoyu aç
    frames = []  # Kareleri saklamak için boş liste
    while True:
        ret, frame = cap.read()  # Bir sonraki kareyi oku
        if not ret:  # Kare okunamazsa (video bittiğinde)
            break
        frames.append(frame)  # Kareyi listeye ekle
    cap.release()  # Videoyu kapat
    return frames  # Tüm kareleri döndür

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Video codec belirle (Motion-JPEG)
    out = cv2.VideoWriter(
        output_video_path,  # Çıktı dosyası yolu
        fourcc,
        24,  # FPS (saniyedeki kare sayısı)
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])  # Genişlik x Yükseklik
    )
    for frame in output_video_frames:
        out.write(frame)  # Kareleri videoya yaz
    out.release()  # Videoyu kaydet ve kapat