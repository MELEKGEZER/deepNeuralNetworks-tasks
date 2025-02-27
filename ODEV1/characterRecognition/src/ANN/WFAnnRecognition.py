import tkinter as tk  # Tkinter kütüphanesini GUI oluşturmak için içe aktar
from tkinter import ttk, messagebox  # Tkinter'ın ttk ve messagebox modüllerini içe aktar
import numpy as np  # Numpy kütüphanesini matematiksel işlemler için içe aktar
from sklearn.neural_network import MLPClassifier  # Yapay sinir ağı modeli için MLPClassifier'ı içe aktar
from sklearn.metrics import mean_squared_error  # Ortalama kare hatasını hesaplamak için mean_squared_error'ı içe aktar

class WFAnnRecognitionUI:
    def __init__(self, root):
        self.root = root  # Tkinter penceresini root olarak ata
        self.root.title("ANN (Artificial Neural Network) Character Recognition - Melek Gezer")  # Pencere başlığını ayarla
        self.root.geometry("1154x397")  # Pencere boyutunu ayarla
        self.root.resizable(False, False)  # Pencerenin boyutunun değiştirilemez olmasını sağla

        # Initialize ANN model
        self.network = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', solver='sgd', max_iter=8000)  # Yapay sinir ağı modelini başlat
        self.input_layer_size = 35  # Giriş katmanı boyutunu 35 olarak ayarla
        self.output_layer_size = 5  # Çıkış katmanı boyutunu 5 olarak ayarla

        # Panel for matrix buttons (INPUT VALUE)
        self.panelInputValue = tk.Frame(root, bg="#FFFFC0")  # Sarı arka planlı bir frame oluştur
        self.panelInputValue.place(x=7, y=9, width=215, height=346)  # Frame'in konumunu ve boyutunu ayarla

        # Create 7x5 matrix buttons for INPUT VALUE
        self.inputButtons = []  # Giriş butonlarını saklamak için boş bir liste oluştur
        for i in range(7):  # 7 satır için döngü
            for j in range(5):  # 5 sütun için döngü
                button = tk.Button(self.panelInputValue, text=" ", width=4, height=2, bg="white", relief="flat")  # Buton oluştur
                button.grid(row=i, column=j, padx=2, pady=2)  # Butonu grid'e yerleştir
                button.bind("<Button-1>", self.btnInputValue_Click)  # Butona tıklama olayını bağla
                self.inputButtons.append(button)  # Butonu listeye ekle

        # Panel for matrix display (MATRIX)
        self.panelMatrisContainer = tk.Frame(root, bg="white")  # Beyaz arka planlı bir frame oluştur
        self.panelMatrisContainer.place(x=228, y=9, width=386, height=375)  # Frame'in konumunu ve boyutunu ayarla

        # Create 7x5 labels for MATRIX
        self.matrixLabels = []  # Matris etiketlerini saklamak için boş bir liste oluştur
        for i in range(7):  # 7 satır için döngü
            for j in range(5):  # 5 sütun için döngü
                label = tk.Label(self.panelMatrisContainer, text="0", width=4, height=2, bg="white", relief="flat")  # Etiket oluştur
                label.grid(row=i, column=j, padx=2, pady=2)  # Etiketi grid'e yerleştir
                self.matrixLabels.append(label)  # Etiketi listeye ekle

        # Clear button
        self.btnClear = tk.Button(root, text="CLEAR", command=self.btnClear_Click)  # Temizle butonu oluştur
        self.btnClear.place(x=7, y=359, width=67, height=30)  # Butonun konumunu ve boyutunu ayarla

        # GroupBox for output
        self.groupBox1 = ttk.LabelFrame(root, text="OUTPUT")  # Çıkış için bir grup kutusu oluştur
        self.groupBox1.place(x=629, y=9, width=513, height=375)  # Grup kutusunun konumunu ve boyutunu ayarla

        # ListBox for outputs
        self.listBoxOutputs = tk.Listbox(self.groupBox1, font=("Microsoft Sans Serif", 14))  # Çıkışları göstermek için bir liste kutusu oluştur
        self.listBoxOutputs.place(x=11, y=20, width=258, height=340)  # Liste kutusunun konumunu ve boyutunu ayarla

        # GroupBox for information
        self.groupBox4 = ttk.LabelFrame(self.groupBox1, text="Information")  # Bilgi için bir grup kutusu oluştur
        self.groupBox4.place(x=275, y=12, width=232, height=348)  # Grup kutusunun konumunu ve boyutunu ayarla

        # Labels and text boxes for learning rate and iteration
        self.label3 = tk.Label(self.groupBox4, text="Set Iteration:")  # İterasyon etiketi oluştur
        self.label3.place(x=43, y=164)  # Etiketin konumunu ayarla
        self.txtIteration = tk.Entry(self.groupBox4)  # İterasyon için metin kutusu oluştur
        self.txtIteration.place(x=116, y=163, width=89)  # Metin kutusunun konumunu ve boyutunu ayarla
        self.txtIteration.insert(0, "5000")  # Metin kutusuna varsayılan değer ekle

        self.label2 = tk.Label(self.groupBox4, text="Set Learning Rate:")  # Öğrenme oranı etiketi oluştur
        self.label2.place(x=14, y=140)  # Etiketin konumunu ayarla
        self.txtLearningRate = tk.Entry(self.groupBox4)  # Öğrenme oranı için metin kutusu oluştur
        self.txtLearningRate.place(x=116, y=137, width=89)  # Metin kutusunun konumunu ve boyutunu ayarla
        self.txtLearningRate.insert(0, "0.3")  # Metin kutusuna varsayılan değer ekle

        # Labels for network information
        self.lblError = tk.Label(self.groupBox4, text="Mean Squared Error: ")  # Ortalama kare hatası etiketi oluştur
        self.lblError.place(x=4, y=103)  # Etiketin konumunu ayarla

        self.lblOutputLayer = tk.Label(self.groupBox4, text="Output Layer: 5")  # Çıkış katmanı etiketi oluştur
        self.lblOutputLayer.place(x=38, y=84)  # Etiketin konumunu ayarla

        self.lblHidden = tk.Label(self.groupBox4, text="Hidden Layer: 3")  # Gizli katman etiketi oluştur
        self.lblHidden.place(x=38, y=62)  # Etiketin konumunu ayarla

        self.lblInputLayer = tk.Label(self.groupBox4, text="Input Layer: 35")  # Giriş katmanı etiketi oluştur
        self.lblInputLayer.place(x=46, y=39)  # Etiketin konumunu ayarla

        self.txtLearning = tk.Label(self.groupBox4, text="Learning Rate: 3")  # Öğrenme oranı etiketi oluştur
        self.txtLearning.place(x=35, y=16)  # Etiketin konumunu ayarla

        # Train button
        self.btnTrain = tk.Button(root, text="TRAIN", command=self.btnTrain_Click)  # Eğitim butonu oluştur
        self.btnTrain.place(x=79, y=359, width=64, height=29)  # Butonun konumunu ve boyutunu ayarla

        # Classify button
        self.btnClassify = tk.Button(root, text="CLASSIFY", state=tk.DISABLED, command=self.btnClassify_Click)  # Sınıflandırma butonu oluştur
        self.btnClassify.place(x=147, y=358, width=75, height=30)  # Butonun konumunu ve boyutunu ayarla

        # Log listbox
        self.lstLog = tk.Listbox(self.groupBox4)  # Logları göstermek için bir liste kutusu oluştur
        self.lstLog.place(x=7, y=196, width=219, height=147)  # Liste kutusunun konumunu ve boyutunu ayarla

    def btnInputValue_Click(self, event):
        button = event.widget  # Tıklanan butonu al
        if button.cget("text") == " ":  # Eğer butonun metni boşsa
            button.config(text="1", bg="black")  # Butonun metnini "1" yap ve arka planını siyah yap
        else:
            button.config(text=" ", bg="white")  # Butonun metnini boş yap ve arka planını beyaz yap
        self.updateMatrixDisplay()  # Matris ekranını güncelle

    def updateMatrixDisplay(self):
        for i, button in enumerate(self.inputButtons):  # Tüm giriş butonları üzerinde döngü
            self.matrixLabels[i].config(text="1" if button.cget("text") == "1" else "0")  # Butonun durumuna göre matris etiketini güncelle

    def btnClear_Click(self):
        for button in self.inputButtons:  # Tüm giriş butonları üzerinde döngü
            button.config(text=" ", bg="white")  # Butonun metnini boş yap ve arka planını beyaz yap
        self.updateMatrixDisplay()  # Matris ekranını güncelle

    def btnTrain_Click(self):
        # Training data
        X = np.array([MyDataSet.A, MyDataSet.B, MyDataSet.C, MyDataSet.D, MyDataSet.E])  # Eğitim verilerini numpy dizisine dönüştür
        y = np.array([
            [1, 0, 0, 0, 0],  # A
            [0, 1, 0, 0, 0],  # B
            [0, 0, 1, 0, 0],  # C
            [0, 0, 0, 1, 0],  # D
            [0, 0, 0, 0, 1]   # E
        ])  # Hedef çıkışları numpy dizisine dönüştür

        self.network.fit(X, y)  # Modeli eğit
        
        # Calculate Mean Squared Error
        y_pred = self.network.predict(X)  # Modelin tahminlerini al
        mse = mean_squared_error(y, y_pred)  # Ortalama kare hatasını hesapla
        self.lblError.config(text=f"Mean Squared Error: {mse:.4f}")  # Hata değerini ekranda göster

        self.lstLog.insert(0, "Training Completed")  # Log'a eğitimin tamamlandığını yaz
        self.lstLog.insert(0, "TRAINED: E")  # Log'a E harfinin eğitildiğini yaz
        self.lstLog.insert(0, "TRAINED: D")  # Log'a D harfinin eğitildiğini yaz
        self.lstLog.insert(0, "TRAINED: C")  # Log'a C harfinin eğitildiğini yaz
        self.lstLog.insert(0, "TRAINED: B")  # Log'a B harfinin eğitildiğini yaz
        self.lstLog.insert(0, "TRAINED: A")  # Log'a A harfinin eğitildiğini yaz
        self.lstLog.insert(0, "INITIALIZE ANN MODEL")  # Log'a modelin başlatıldığını yaz
        self.btnClassify.config(state=tk.NORMAL)  # Sınıflandırma butonunu etkinleştir

    def btnClassify_Click(self):
        input_matrix = self.getInputs()  # Kullanıcının girdiği matrisi al
        output = self.network.predict_proba([input_matrix])[0]  # Modelin tahmin olasılıklarını al
        self.lstLog.insert(0, "Classification Completed!")  # Log'a sınıflandırmanın tamamlandığını yaz

        # Display output
        self.listBoxOutputs.delete(0, tk.END)  # Liste kutusunu temizle
        for i, value in enumerate(output):  # Tahmin olasılıkları üzerinde döngü
            self.listBoxOutputs.insert(tk.END, f"{i + 1}-{value}")  # Her bir çıkışı liste kutusuna ekle

    def getInputs(self):
        inputs = np.zeros(35)  # 35 elemanlı bir numpy dizisi oluştur
        for i, button in enumerate(self.inputButtons):  # Tüm giriş butonları üzerinde döngü
            if button.cget("text") == "1":  # Eğer butonun metni "1" ise
                inputs[i] = 1  # Dizideki ilgili indeksi 1 yap
        return inputs  # Giriş matrisini döndür


class MyDataSet:
    A = np.array([  # A harfi için örnek veri
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1
    ])

    B = np.array([  # B harfi için örnek veri
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1
    ])

    C = np.array([  # C harfi için örnek veri
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    D = np.array([  # D harfi için örnek veri
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 0
    ])

    E = np.array([  # E harfi için örnek veri
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])


if __name__ == "__main__":
    root = tk.Tk()  # Tkinter penceresini oluştur
    app = WFAnnRecognitionUI(root)  # Uygulama sınıfını başlat
    root.mainloop()  # Pencereyi açık tut