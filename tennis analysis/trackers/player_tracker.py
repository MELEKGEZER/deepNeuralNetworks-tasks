from ultralytics import YOLO  # YOLOv8 modelini kullanmak için
import cv2  # OpenCV kütüphanesi (görüntü işleme)
import pickle  # Nesneleri dosyaya kaydetmek/okumak için
import sys  # Sistem işlemleri için

sys.path.append('../')  # Üst dizindeki modüllere erişim için yol ekler
from utils import measure_distance, get_center_of_bbox  # Özel yardımcı fonksiyonlar


class PlayerTracker:
    def __init__(self, model_path):
        # YOLO modelini belirtilen yoldan yükler
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # İlk karedeki oyuncu tespitlerini al
        player_detections_first_frame = player_detections[0]
        # Kort çizgilerine en yakın 2 oyuncuyu seç
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)

        # Tüm karelerde sadece seçilen oyuncuları sakla
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if
                                    track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        # Oyuncuların kort çizgilerine olan mesafelerini hesapla
        distances = []
        for track_id, bbox in player_dict.items():
            # Bounding box'ın merkez noktasını al
            player_center = get_center_of_bbox(bbox)

            # En yakın kort çizgisi noktasını bul
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Mesafelere göre sırala (en yakından en uzağa)
        distances.sort(key=lambda x: x[1])
        # İlk 2 oyuncuyu seç (en yakın olanlar)
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # Eğer önbellek okuma modu aktifse ve dosya varsa
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                # Önbellekten tespit verilerini yükle
                player_detections = pickle.load(f)
            return player_detections

        # Tüm karelerde oyuncu tespiti yap
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # Eğer önbellek dosya yolu verilmişse
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                # Tespit verilerini dosyaya kaydet
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        # YOLO ile nesne tespiti ve takibi yap
        results = self.model.track(frame, persist=True)[0]
        # Sınıf isimlerini al (0: person, 1: ball vb.)
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            # Takip ID'sini al
            track_id = int(box.id.tolist()[0])
            # Bounding box koordinatlarını al [x1,y1,x2,y2]
            result = box.xyxy.tolist()[0]
            # Nesne sınıf ID'sini al
            object_cls_id = box.cls.tolist()[0]
            # Sınıf ismini al
            object_cls_name = id_name_dict[object_cls_id]
            # Sadece "person" sınıfını al
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict  # {track_id: [x1,y1,x2,y2]}

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        # Her kare ve oyuncu tespiti için
        for frame, player_dict in zip(video_frames, player_detections):
            # Bounding box'ları çiz
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Oyuncu ID'sini yaz
                cv2.putText(frame, f"Player ID: {track_id}",
                            (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # Bounding box çiz
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames  # Çizim yapılmış kareler