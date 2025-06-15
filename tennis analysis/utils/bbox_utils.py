def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox # Bounding box koordinatlarını açar.
    center_x = int((x1 + x2) / 2) # Bounding box'ın yatay merkez noktasını hesaplar.
    center_y = int((y1 + y2) / 2) # Bounding box'ın dikey merkez noktasını hesaplar.
    return (center_x, center_y) # Hesaplanan merkez noktasının (x, y) koordinatlarını döndürür.

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 # İki nokta arasındaki Öklid mesafesini hesaplar.

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox # Bounding box koordinatlarını açar.
    return (int((x1 + x2) / 2), y2) # Bounding box'ın alt orta noktasını (ayak pozisyonu tahmini) hesaplar ve döndürür.

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   closest_distance = float('inf') # Başlangıçta en yakın mesafeyi sonsuz olarak ayarlar.
   key_point_ind = keypoint_indices[0] # Başlangıçta ilk anahtar nokta indeksini en yakın olarak varsayar.
   for keypoint_indix in keypoint_indices: # Belirtilen anahtar nokta indeksleri üzerinde döngü kurar.
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1] # Mevcut anahtar noktanın (x, y) koordinatlarını alır.
       distance = abs(point[1]-keypoint[1]) # Verilen nokta ile mevcut anahtar noktanın dikey mesafesini hesaplar.

       if distance<closest_distance: # Eğer hesaplanan mesafe mevcut en yakın mesafeden daha küçükse.
           closest_distance = distance # Yeni en yakın mesafeyi günceller.
           key_point_ind = keypoint_indix # En yakın anahtar noktanın indeksini günceller.

   return key_point_ind # En yakın anahtar noktanın indeksini döndürür.

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1] # Bounding box'ın yüksekliğini (alt y - üst y) hesaplar.

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1]) # İki nokta arasındaki yatay ve dikey mesafeleri ayrı ayrı hesaplar.

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2)) # Bounding box'ın merkez noktasının (x, y) koordinatlarını hesaplar ve döndürür.