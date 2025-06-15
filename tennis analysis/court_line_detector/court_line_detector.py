import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np


class CourtLineDetector:
    def __init__(self, model_path):
        # MobileNetV2 modelini yükle (önceden eğitilmiş)
        self.model = models.mobilenet_v2(pretrained=True)

        # Son katmanı (classifier) değiştir: 14 nokta × 2 koordinat (x,y)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),  # MobileNetV2'de varsayılan dropout oranı
            torch.nn.Linear(self.model.last_channel, 14 * 2)  # 28 çıkışlı lineer katman
        )

        # Eğitilmiş ağırlıkları yükle
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()  # Çıkarım moduna al

        # Görüntü ön işleme pipeline'ı (MobileNetV2 için standart normalizasyon)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # MobileNetV2'nin beklediği giriş boyutu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Görüntüyü RGB'ye çevir ve ön işle
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)  # Batch boyutu ekle

        # Çıkarım yap
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Tahminleri numpy array'e çevir
        keypoints = outputs.squeeze().cpu().numpy()

        # Koordinatları orijinal görüntü boyutuna ölçekle
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0  # x koordinatları
        keypoints[1::2] *= original_h / 224.0  # y koordinatları

        return keypoints

    # Aynı çizim fonksiyonları (değişiklik gerekmez)
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        return [self.draw_keypoints(frame, keypoints) for frame in video_frames]