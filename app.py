# === emotion_utils.py ===
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
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


# === gemini_utils.py ===
import google.generativeai as genai

def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def get_gemini_suggestions(emotion_type, text, api_key):
    model = initialize_gemini(api_key)

    # สร้าง prompt ตามประเภทอารมณ์
    if "negative" in emotion_type:
        prompt = f"""
        ฉันรู้สึกไม่ดีและเขียนข้อความนี้: "{text}"

        วิเคราะห์ว่าข้อความนี้แสดงถึงอารมณ์ใดเป็นพิเศษ (เช่น เศร้า โกรธ เหนื่อย กังวล เหงา หรืออื่นๆ)
        จากนั้นให้คำแนะนำที่เป็นประโยชน์และเฉพาะเจาะจงกับอารมณ์นั้น 3-4 ข้อ
        คำแนะนำควรเป็นภาษาไทย กระชับ และปฏิบัติได้จริง
        ตอบในรูปแบบข้อความที่มีจุดนำหน้าแต่ละคำแนะนำ
        """
    elif "positive" in emotion_type:
        prompt = f"""
        ฉันรู้สึกดีและเขียนข้อความนี้: "{text}"

        วิเคราะห์ว่าข้อความนี้แสดงถึงอารมณ์ใดเป็นพิเศษ (เช่น มีความสุข สงบ พึงพอใจ ตื่นเต้น หรืออื่นๆ)
        จากนั้นให้คำแนะนำที่เป็นประโยชน์เพื่อรักษาหรือต่อยอดความรู้สึกดีนั้น 3-4 ข้อ
        คำแนะนำควรเป็นภาษาไทย กระชับ และปฏิบัติได้จริง
        ตอบในรูปแบบข้อความที่มีจุดนำหน้าแต่ละคำแนะนำ
        """
    else:  # เป็นกลาง
        prompt = f"""
        ฉันรู้สึกเป็นกลางและเขียนข้อความนี้: "{text}"

        วิเคราะห์ความรู้สึกที่อาจซ่อนอยู่ในข้อความนี้
        จากนั้นให้คำแนะนำที่เป็นประโยชน์เพื่อเสริมสร้างความรู้สึกเชิงบวก 3-4 ข้อ
        คำแนะนำควรเป็นภาษาไทย กระชับ และเฉพาะเจาะจงกับเนื้อหาข้อความ
        ตอบในรูปแบบข้อความที่มีจุดนำหน้าแต่ละคำแนะนำ
        """

    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else "ไม่สามารถตอบกลับจาก Gemini ได้"


# === summarize.py ===
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# === main.py ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from gemini_utils import initialize_gemini, get_gemini_suggestions
from emotion_utils import analyze_emotion_with_wangchanberta, get_default_suggestions
from summarize import summarize_text

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    use_gemini: Optional[bool] = True
    gemini_api_key: Optional[str] = None

@app.post("/analyze")
async def analyze(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")

    summarized = summarize_text(request.text)
    emotion = analyze_emotion_with_wangchanberta(summarized)

    if request.use_gemini and request.gemini_api_key:
        suggestion = get_gemini_suggestions(emotion, summarized, request.gemini_api_key)
    else:
        suggestion = get_default_suggestions(emotion)

    return {
        "summarized": summarized,
        "emotion": emotion,
        "suggestion": suggestion
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)


