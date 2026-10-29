import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./trained_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def clean_response(text: str) -> str:
    # 1. Отрезаем новый блок диалога, если модель начала писать следующий вопрос/ответ
    for marker in ["\n", "Вопрос:", "Ответ:"]:
        if marker in text:
            text = text.split(marker)[0]
    
    text = text.strip()

    # 2. Отрезаем незавершенный хвост по последнему знаку препинания
    punctuation_marks = [".", "!", "?"]
    last_idx = max(text.rfind(p) for p in punctuation_marks)
    
    if last_idx != -1:
        # Оставляем текст строго до последней точки/восклицательного/вопросительного знака
        text = text[:last_idx + 1]
    
    return text.strip()

def chat_with_ai(user_input):
    if not user_input.strip():
        return ""
    
    prompt = f"Вопрос: {user_input}\nОтвет:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            repetition_penalty=1.2
        )
    
    # Извлекаем только сгенерированные токены (без входного промпта)
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Применяем фильтрацию
    response = clean_response(raw_response)
    
    # Запасной вариант: если знаков препинания не было вообще
    return response if response else raw_response.strip()

gr.Interface(
    fn=chat_with_ai,
    inputs=gr.Textbox(lines=2, placeholder="Введите сообщение..."),
    outputs="text",
    title="Мой кастомный AI",
    description="Обученный AI-помощник на DistilGPT-2"
).launch()