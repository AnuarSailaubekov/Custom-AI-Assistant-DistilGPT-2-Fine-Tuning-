import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"
DATA_PATH = "data/data.jsonl"
OUTPUT_DIR = "./trained_model"

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Для GPT-2 назначаем eos_token в качестве pad_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Загрузка данных
def load_data(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)

dataset = load_data(DATA_PATH)

# Токенизация с шаблоном диалога и маскированием паддингов
def tokenize_function(example):
    # Добавляем разделитель между вопросом и ответом, а в конце — токен завершения
    full_text = f"Вопрос: {example['prompt']}\nОтвет: {example['response']}{tokenizer.eos_token}"
    tokens = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # Игнорируем pad-токены при подсчете ошибки (loss)
    labels = [
        label if label != tokenizer.pad_token_id else -100
        for label in tokens["input_ids"]
    ]
    tokens["labels"] = labels
    return tokens

tokenized_datasets = dataset.map(tokenize_function, remove_columns=["prompt", "response"])

# Настройки обучения
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=20,
    learning_rate=5e-5,
    logging_steps=5,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

print("Начинаем обучение модели...")
trainer.train()

# Сохранение весов и токенизатора
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Модель успешно сохранена в {OUTPUT_DIR}!")