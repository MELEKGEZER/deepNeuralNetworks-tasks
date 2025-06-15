import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats):

    for index, row in player_stats.iterrows(): # Player istatistikleri DataFrame'inin satırları üzerinde döngü kurar.
        player_1_shot_speed = row['player_1_last_shot_speed'] # Oyuncu 1'in son şut hızını alır.
        player_2_shot_speed = row['player_2_last_shot_speed'] # Oyuncu 2'nin son şut hızını alır.
        player_1_speed = row['player_1_last_player_speed'] # Oyuncu 1'in son hareket hızını alır.
        player_2_speed = row['player_2_last_player_speed'] # Oyuncu 2'nin son hareket hızını alır.

        avg_player_1_shot_speed = row['player_1_average_shot_speed'] # Oyuncu 1'in ortalama şut hızını alır.
        avg_player_2_shot_speed = row['player_2_average_shot_speed'] # Oyuncu 2'nin ortalama şut hızını alır.
        avg_player_1_speed = row['player_1_average_player_speed'] # Oyuncu 1'in ortalama hareket hızını alır.
        avg_player_2_speed = row['player_2_average_player_speed'] # Oyuncu 2'nin ortalama hareket hızını alır.

        frame = output_video_frames[index] # Mevcut frame'i alır.
        shapes = np.zeros_like(frame, np.uint8) # Frame ile aynı boyutta siyah bir maske oluşturur (şu anda kullanılmıyor).

        width=350 # Bilgi kutusunun genişliğini tanımlar.
        height=230 # Bilgi kutusunun yüksekliğini tanımlar.

        start_x = frame.shape[1]-400 # Bilgi kutusunun sol üst köşesinin x koordinatını (sağ alt köşeye yakın) tanımlar.
        start_y = frame.shape[0]-500 # Bilgi kutusunun sol üst köşesinin y koordinatını (alt kısma yakın) tanımlar.
        end_x = start_x+width # Bilgi kutusunun sağ alt köşesinin x koordinatını hesaplar.
        end_y = start_y+height # Bilgi kutusunun sağ alt köşesinin y koordinatını hesaplar.

        overlay = frame.copy() # Mevcut frame'in bir kopyasını oluşturur (overlay için).
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1) # Overlay üzerine siyah bir dikdörtgen çizer (bilgi kutusu arka planı).
        alpha = 0.5 # Overlay'in şeffaflık değerini tanımlar.
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) # Orijinal frame ile şeffaf arkaplanı birleştirir.
        output_video_frames[index] = frame # İşlenmiş frame'i çıktı listesinde günceller.

        text = "     Player 1     Player 2" # Oyuncu isimleri başlığını tanımlar.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # Başlığı frame üzerine yazar.

        text = "Shot Speed" # Şut hızı etiketini tanımlar.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1) # Şut hızı etiketini frame üzerine yazar.
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h" # Oyuncuların son şut hızlarını biçimlendirir.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Oyuncuların son şut hızlarını frame üzerine yazar.

        text = "Player Speed" # Oyuncu hızı etiketini tanımlar.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1) # Oyuncu hızı etiketini frame üzerine yazar.
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h" # Oyuncuların son hareket hızlarını biçimlendirir.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Oyuncuların son hareket hızlarını frame üzerine yazar.


        text = "avg. S. Speed" # Ortalama şut hızı etiketini tanımlar.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1) # Ortalama şut hızı etiketini frame üzerine yazar.
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h" # Oyuncuların ortalama şut hızlarını biçimlendirir.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Oyuncuların ortalama şut hızlarını frame üzerine yazar.

        text = "avg. P. Speed" # Ortalama oyuncu hızı etiketini tanımlar.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1) # Ortalama oyuncu hızı etiketini frame üzerine yazar.
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h" # Oyuncuların ortalama hareket hızlarını biçimlendirir.
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Oyuncuların ortalama hareket hızlarını frame üzerine yazar.

    return output_video_frames # İşlenmiş (üzerinde istatistikler çizilmiş) video frame'lerinin listesini döndürür.