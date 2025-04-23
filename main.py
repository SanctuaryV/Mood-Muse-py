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
