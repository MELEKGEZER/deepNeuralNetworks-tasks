from ultralytics import YOLO
import cv2
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) # YOLO modelini belirtilen yoldan yükler.

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate() # DataFrame'deki eksik değerleri doğrusal interpolasyon ile doldurur.
        df_ball_positions = df_ball_positions.bfill() # İnterpolasyonun dolduramadığı eksik değerleri sonraki geçerli değerle doldurur.

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff() # Topun dikey hareketindeki değişimi hesaplar.

        minimum_change_frames_for_hit = 25
        df_ball_positions = df_ball_positions.copy()

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    neg_following = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    pos_following = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and neg_following:
                        change_count += 1
                    elif positive_position_change and pos_following:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[df_ball_positions.index[i], 'ball_hit'] = 1 # Vuruş olarak değerlendirilen frame'i işaretler.

        return df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist() # Vuruş frame'lerinin indekslerini döndürür.

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f) # Önbellekten top tespitlerini yükler.
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame) # Her bir frame için top tespiti yapar.
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f) # Top tespitlerini önbelleğe kaydeder.

        return ball_detections

    def detect_frame(self, frame):
        results = self.model(frame, conf=0.15) # YOLO modelini kullanarak nesne tespiti yapar.
        if isinstance(results, list):
            results = results[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result # Tespit edilen topun bounding box koordinatlarını saklar.

        return ball_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) # Tespit edilen topun etrafına bounding box çizer.
            output_video_frames.append(frame)
        return output_video_frames