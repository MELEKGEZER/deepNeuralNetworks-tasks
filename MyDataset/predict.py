from transformers import BertTokenizer, BertForSequenceClassification
import torch


model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model')


sentence = "I gave him a right hook then a left jab"


inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)


with torch.no_grad():
    logits = model(**inputs).logits


scores = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()


theme_list = ["friendship", "hope", "sacrifice", "battle", "self development", "betrayal", "love", "dialogue"]


labels_scores = list(zip(theme_list, scores))


sorted_labels_scores = sorted(labels_scores, key=lambda x: x[1], reverse=True)


sorted_labels = [label for label, score in sorted_labels_scores]
sorted_scores = [score for label, score in sorted_labels_scores]


output = {
    'sequence': sentence,
    'labels': sorted_labels,
    'scores': sorted_scores
}

print(output)
