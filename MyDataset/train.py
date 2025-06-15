import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


import pandas as pd
train_df = pd.read_csv('train.csv')
eval_df = pd.read_csv('eval.csv')


train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

eval_texts = eval_df['text'].tolist()
eval_labels = eval_df['label'].tolist()

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

train_encodings = tokenize_function(train_texts)
eval_encodings = tokenize_function(eval_texts)


class ThemeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ThemeDataset(train_encodings, train_labels)
eval_dataset = ThemeDataset(eval_encodings, eval_labels)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)


training_args = TrainingArguments(
    output_dir='./results',          # Çıktı dosyalarının kaydedileceği dizin
    num_train_epochs=3,              # Epoch sayısı
    per_device_train_batch_size=16,  # Eğitimdeki batch boyutu
    per_device_eval_batch_size=64,   # Değerlendirme batch boyutu
    warmup_steps=500,                # Öğrenme oranı 500 adımda ısınma
    weight_decay=0.01,               # Ağırlık azalması
    logging_dir='./logs',            # Log dosyaları
    logging_steps=10,                # Log kaydı atma sıklığı
)

trainer = Trainer(
    model=model,                         # Eğitilecek model
    args=training_args,                  # Eğitim parametreleri
    train_dataset=train_dataset,         # Eğitim veri kümesi
    eval_dataset=eval_dataset            # Değerlendirme veri kümesi
)


trainer.train()


model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
