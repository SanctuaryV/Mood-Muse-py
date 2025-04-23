from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

# โมเดลที่ใช้
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3,torch_dtype=torch.float16)

# โหลด stopwords ภาษาไทย
stop_words = set(thai_stopwords())

# ฟังก์ชัน preprocess สำหรับข้อความภาษาไทย
def preprocess(text):
    # ลบอักขระพิเศษและตัวเลขออกจากข้อความ
    text = re.sub(r'[^ก-ฮะ-์\s]', '', text)
    # ใช้ pythainlp ในการ tokenize ข้อความ
    tokens = word_tokenize(text)
    # ลบ stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ฟังก์ชันวิเคราะห์อารมณ์
EMOTION_LABELS = ['negative', 'neutral', 'positive']

def analyze_emotion_with_wangchanberta(text):
    # Preprocess ข้อความก่อน
    processed_text = preprocess(text)
    
    # ใช้ torch สำหรับการประมวลผลโมเดล
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True)
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
