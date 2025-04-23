from transformers import RobertaTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = RobertaTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

EMOTION_LABELS = ['negative', 'neutral', 'positive']

def analyze_emotion_with_wangchanberta(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return EMOTION_LABELS[pred]

def get_default_suggestions(emotion):
    suggestions = {
        "positive": "เยี่ยมเลย! รักษาอารมณ์ดีแบบนี้ไว้นะ",
        "neutral": "อารมณ์กลาง ๆ ลองพักผ่อนหรือทำสิ่งที่ชอบดูนะ",
        "negative": "ถ้ารู้สึกไม่ดี ลองฟังเพลงเบา ๆ หรือคุยกับเพื่อนดูนะ"
    }
    return suggestions.get(emotion, "ลองหากิจกรรมที่ทำให้คุณสบายใจดูนะ")
