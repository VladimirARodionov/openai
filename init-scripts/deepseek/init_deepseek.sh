#!/bin/bash

echo "Запуск Ollama сервера..."
ollama serve &

# Ждем пока Ollama запустится
echo "Ожидание запуска Ollama сервера..."
sleep 10

# Проверяем доступность API
while ! curl -f http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "Ожидание доступности Ollama API..."
    sleep 5
done

echo "Ollama API доступен, загружаем модель DeepSeek..."

# Загружаем модель DeepSeek R1 (8B параметров - оптимальная для большинства задач)
ollama pull ${DEEPSEEK_MODEL}

# Также загружаем модель для эмбеддингов (более легкая)
ollama pull ${DEEPSEEK_EMBED_MODEL}

echo "Модели DeepSeek загружены успешно!"
echo "API доступен по адресу: http://localhost:11434"

# Держим контейнер запущенным
wait
