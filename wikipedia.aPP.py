import gradio as gr
from transformers import pipeline
import wikipedia
import re

# --- Modeller ---
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
qa_model = pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")  # Türkçe destekli güçlü model

# --- Wikipedia içeriğini getir ---
def get_article(title, lang):
    wikipedia.set_lang(lang)
    try:
        page = wikipedia.page(title)
        return page.title, page.content
    except Exception as e:
        return None, f"Hata: {str(e)}"

# --- Özetleme (maks. 3 cümle) ---
def summarize(text):
    if len(text) < 100:
        return "Metin özetlemek için çok kısa."
    summary = summarizer(text[:1000])[0]['summary_text']
    sentences = re.split(r'(?<=[.!?]) +', summary)
    return " ".join(sentences[:3])

# --- Paragraf Tabanlı Akıllı Soru-Cevap ---
def smart_qa(content, question):
    paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 50]
    best_answer = {"answer": "", "score": 0}
    for para in paragraphs:
        try:
            result = qa_model(question=question, context=para)
            if result["score"] > best_answer["score"] and result["answer"].strip() != "":
                best_answer = result
        except:
            continue
    if best_answer["score"] > 0.25:
        return best_answer["answer"]
    return "❌ Cevap bulunamadı. Daha açık bir soru deneyin."

# --- Ana fonksiyon ---
def run_assistant(title, lang_code, question, context_choice):
    real_title, content = get_article(title, lang_code)
    if real_title is None:
        return "", "", "", content
    summary = summarize(content)
    selected_context = summary if context_choice == "Özet" else content[:3000]
    answer = smart_qa(selected_context, question)
    return real_title, content[:3000], summary, answer

# --- Arayüz ---
iface = gr.Interface(
    fn=run_assistant,
    inputs=[
        gr.Textbox(label="🔎 Wikipedia Başlığı"),
        gr.Radio(["en", "tr"], label="🌍 Dil", value="tr"),
        gr.Textbox(label="❓ Soru"),
        gr.Radio(["Özet", "Tam İçerik"], label="💬 Cevap Bağlamı", value="Özet")
    ],
    outputs=[
        gr.Textbox(label="📄 Gerçek Başlık"),
        gr.Textbox(label="📚 İçerik (ilk 3000 karakter)", lines=10),
        gr.Textbox(label="✂️ Özet (maks. 3 cümle)", lines=3),
        gr.Textbox(label="💬 Cevap", lines=2)
    ],
    title="🧠 Gelişmiş Wikipedia Soru-Cevap Asistanı",
    description="Wikipedia'dan içerik çeker, özet üretir ve soruları akıllı şekilde yanıtlar. Türkçe desteklidir.",
)

iface.launch()
