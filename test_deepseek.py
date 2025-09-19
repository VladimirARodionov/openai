#!/usr/bin/env python3
"""
Скрипт для тестирования интеграции DeepSeek
"""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from services.deepseek_client import DeepSeekClient, DeepSeekLLM, DeepSeekEmbedding
from llama_index.core.base.llms.types import ChatMessage
from llama_cloud import MessageRole


async def test_deepseek_connection():
    """Тестирует подключение к DeepSeek API"""
    print("🔍 Тестирование подключения к DeepSeek...")
    
    client = DeepSeekClient(base_url="http://localhost:11434")
    
    try:
        # Тест простого чата
        messages = [
            ChatMessage(
                content="Привет! Ты работаешь?",
                role=MessageRole.USER
            )
        ]
        
        response = await client.chat_completion(messages)
        print(f"✅ Чат работает! Ответ: {response['choices'][0]['message']['content'][:100]}...")
        
        # Тест эмбеддингов
        embeddings_response = await client.embeddings(["Тестовый текст для эмбеддинга"])
        embedding_dim = len(embeddings_response['data'][0]['embedding'])
        print(f"✅ Эмбеддинги работают! Размерность: {embedding_dim}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения к DeepSeek: {str(e)}")
        await client.close()
        return False


async def test_llama_index_integration():
    """Тестирует интеграцию с llama-index"""
    print("🔍 Тестирование интеграции с llama-index...")
    
    try:
        # Тест LLM
        llm = DeepSeekLLM(base_url="http://localhost:11434")
        
        messages = [
            ChatMessage(
                content="Напиши короткий ответ о том, что такое искусственный интеллект",
                role=MessageRole.USER
            )
        ]
        
        response = await llm.achat(messages)
        print(f"✅ LLM интеграция работает! Ответ: {str(response)[:100]}...")
        
        # Тест эмбеддингов
        embed_model = DeepSeekEmbedding(base_url="http://localhost:11434")
        embedding = await embed_model.aget_text_embedding("Тестовый текст")
        print(f"✅ Embedding интеграция работает! Размерность: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка интеграции с llama-index: {str(e)}")
        return False


async def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестирования DeepSeek интеграции...\n")
    
    # Проверяем доступность API
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                print("✅ DeepSeek API доступен")
                models = response.json()
                print(f"📋 Доступные модели: {[m['name'] for m in models.get('models', [])]}")
            else:
                print("❌ DeepSeek API недоступен")
                return
    except Exception as e:
        print(f"❌ Не удается подключиться к DeepSeek API: {str(e)}")
        print("💡 Убедитесь, что контейнер deepseek запущен: docker-compose up deepseek")
        return
    
    print()
    
    # Тестируем базовое подключение
    connection_ok = await test_deepseek_connection()
    print()
    
    # Тестируем интеграцию с llama-index
    if connection_ok:
        integration_ok = await test_llama_index_integration()
        print()
        
        if integration_ok:
            print("🎉 Все тесты прошли успешно! DeepSeek готов к использованию.")
        else:
            print("⚠️ Проблемы с интеграцией llama-index")
    else:
        print("⚠️ Проблемы с базовым подключением к DeepSeek")


if __name__ == "__main__":
    asyncio.run(main())
