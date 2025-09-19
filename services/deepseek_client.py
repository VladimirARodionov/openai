"""
Клиент для работы с DeepSeek через Ollama API
Обеспечивает совместимость с OpenAI API
"""

import httpx
import logging
from typing import List, Dict, Any, AsyncIterable
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_cloud import MessageRole

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Клиент для работы с DeepSeek через Ollama API"""
    
    def __init__(self, base_url: str = "http://deepseek:11434", model: str = "deepseek-r1:8b"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def chat_completion(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """
        Создает chat completion запрос к DeepSeek через Ollama API
        Совместимо с форматом OpenAI
        """
        try:
            # Конвертируем ChatMessage в формат Ollama
            ollama_messages = []
            for msg in messages:
                role_map = {
                    MessageRole.SYSTEM: "system",
                    MessageRole.USER: "user", 
                    MessageRole.ASSISTANT: "assistant"
                }
                ollama_messages.append({
                    "role": role_map.get(msg.role, "user"),
                    "content": msg.content
                })
            
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 20480),  # Увеличиваем лимит токенов
                    "stop": [],  # Убираем стоп-слова
                    "repeat_penalty": 1.1,  # Против повторений
                    "seed": -1,  # Случайность
                    "penalize_newline": False  # Разрешаем переносы строк
                }
            }
            
            logger.info(f"Отправка запроса к DeepSeek: {self.base_url}/api/chat")
            
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка DeepSeek API: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek API error: {response.status_code}")
            
            result = response.json()
            
            # Форматируем ответ в стиле OpenAI
            formatted_response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": result.get("message", {}).get("content", "")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
            
            logger.info(f"Получен ответ от DeepSeek: {len(formatted_response['choices'][0]['message']['content'])} символов")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Ошибка при обращении к DeepSeek: {str(e)}")
            raise
    
    async def embeddings(self, texts: List[str], model: str = "nomic-embed-text") -> Dict[str, Any]:
        """
        Создает эмбеддинги через Ollama API
        Совместимо с форматом OpenAI
        """
        try:
            all_embeddings = []
            
            for text in texts:
                payload = {
                    "model": model,
                    "prompt": text
                }
                
                response = await self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Ошибка эмбеддинга DeepSeek API: {response.status_code} - {response.text}")
                    raise Exception(f"DeepSeek embeddings API error: {response.status_code}")
                
                result = response.json()
                all_embeddings.append({
                    "object": "embedding",
                    "embedding": result.get("embedding", []),
                    "index": len(all_embeddings)
                })
            
            # Форматируем ответ в стиле OpenAI
            formatted_response = {
                "object": "list",
                "data": all_embeddings,
                "model": model,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                }
            }
            
            logger.info(f"Создано {len(all_embeddings)} эмбеддингов")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов DeepSeek: {str(e)}")
            raise
    
    async def close(self):
        """Закрывает HTTP клиент"""
        await self.client.aclose()


class DeepSeekLLM(LLM):
    """Обертка для интеграции DeepSeek с llama-index"""
    
    def __init__(self, model: str = "deepseek-r1:8b", base_url: str = "http://deepseek:11434", **kwargs):
        # Инициализируем базовый класс с model_name
        super().__init__(model_name=model, **kwargs)
        # Используем object.__setattr__ для обхода Pydantic валидации
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_base_url', base_url)
        object.__setattr__(self, 'client', DeepSeekClient(base_url, model))
        object.__setattr__(self, '_kwargs', kwargs)
    
    def _clean_response(self, content: str) -> str:
        """Очищает ответ от служебных тегов DeepSeek"""
        import re
        
        # Удаляем служебные токены DeepSeek
        content = re.sub(r'<｜begin▁of▁sentence｜>', '', content)
        content = re.sub(r'<｜end▁of▁sentence｜>', '', content)
        content = re.sub(r'<｜.*?｜>', '', content)  # Удаляем любые другие служебные токены
        
        # Очищаем от лишних пробелов и переносов в начале
        content = content.strip()
        
        return content
    
    @property
    def metadata(self):
        """Метаданные LLM в формате, совместимом с llama-index"""
        # Создаем объект с атрибутами вместо словаря
        class LLMMetadata:
            def __init__(self, model_name: str, context_window: int = 8192, num_output: int = 2048):
                self.model_name = model_name
                self.context_window = context_window
                self.num_output = num_output
                self.is_chat_model = True
                self.is_function_calling_model = False
        
        return LLMMetadata(
            model_name=self._model,
            context_window=32768,  # Реальное окно контекста для DeepSeek R1
            num_output=20480       # Максимальная длина ответа
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Синхронный completion через requests"""
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        chat_response = self.chat(messages, **kwargs)
        
        # Очищаем ответ от служебных тегов
        clean_text = self._clean_response(chat_response.message.content)
        
        return CompletionResponse(
            text=clean_text,
            raw=chat_response.raw
        )
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Асинхронный completion"""
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        merged_kwargs = {**self._kwargs, **kwargs}
        response = await self.client.chat_completion(messages, **merged_kwargs)
        
        return CompletionResponse(
            text=response["choices"][0]["message"]["content"],
            raw=response
        )
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Синхронный чат с DeepSeek через requests"""
        import requests

        try:
            # Конвертируем ChatMessage в формат Ollama
            ollama_messages = []
            for msg in messages:
                role_map = {
                    MessageRole.SYSTEM: "system",
                    MessageRole.USER: "user", 
                    MessageRole.ASSISTANT: "assistant"
                }
                ollama_messages.append({
                    "role": role_map.get(msg.role, "user"),
                    "content": msg.content
                })
            
                payload = {
                    "model": self._model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "num_predict": kwargs.get("max_tokens", 20480),  # Увеличиваем лимит токенов
                        "stop": [],  # Убираем стоп-слова
                        "repeat_penalty": 1.1,  # Против повторений
                        "seed": -1,  # Случайность
                        "typical_p": 1.0,  # Typical sampling
                        "repeat_last_n": 64  # Контекст для repeat_penalty
                    }
                }
            
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=900
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка DeepSeek chat API: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek chat API error: {response.status_code}")
            
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            # Диагностика ответа DeepSeek
            logger.info(f"DeepSeek raw response: {result}")
            logger.info(f"DeepSeek content length: {len(content)} chars")
            logger.info(f"DeepSeek content preview: {content[:200]}...")
            
            # Очищаем ответ от служебных токенов DeepSeek
            cleaned_content = self._clean_response(content)
            logger.info(f"DeepSeek cleaned content preview: {cleaned_content[:200]}...")
            
            # Создаем объект ответа совместимый с llama-index
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=cleaned_content
                ),
                raw=result
            )
            
        except Exception as e:
            logger.error(f"Ошибка при чате с DeepSeek: {str(e)}")
            raise
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Асинхронный чат с DeepSeek"""
        merged_kwargs = {**self._kwargs, **kwargs}
        response = await self.client.chat_completion(messages, **merged_kwargs)
        
        # Очищаем ответ от служебных токенов DeepSeek
        content = response["choices"][0]["message"]["content"]
        cleaned_content = self._clean_response(content)
        
        # Создаем объект ответа совместимый с llama-index
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=cleaned_content
            ),
            raw=response
        )
    
    async def astream_complete(self, prompt: str, **kwargs) -> AsyncIterable:
        """Асинхронный streaming completion - заглушка"""
        # Возвращаем пустой асинхронный итератор
        async def empty_generator():
            return
            yield  # Недостижимый код, но нужен для создания генератора
        return empty_generator()
    
    def stream_complete(self, prompt: str, **kwargs):
        """Синхронный streaming completion - заглушка"""
        # Возвращаем пустой итератор
        return iter([])
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncIterable:
        """Асинхронный streaming chat - заглушка"""
        # Возвращаем пустой асинхронный итератор
        async def empty_generator():
            return
            yield  # Недостижимый код, но нужен для создания генератора
        return empty_generator()
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        """Синхронный streaming chat - заглушка"""
        # Возвращаем пустой итератор
        return iter([])


class DeepSeekEmbedding(BaseEmbedding):
    """Обертка для эмбеддингов DeepSeek"""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://deepseek:11434", **kwargs):
        super().__init__(**kwargs)
        # Используем object.__setattr__ для обхода возможных ограничений
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_base_url', base_url)
        object.__setattr__(self, 'client', DeepSeekClient(base_url))
        object.__setattr__(self, '_kwargs', kwargs)
    
    @classmethod
    def class_name(cls) -> str:
        return "DeepSeekEmbedding"
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Получить эмбеддинг для запроса"""
        response = await self.client.embeddings([query], self._model)
        return response["data"][0]["embedding"]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Синхронное получение эмбеддинга запроса через requests"""
        import requests

        try:
            payload = {
                "model": self._model,
                "prompt": query
            }
            
            response = requests.post(
                f"{self._base_url}/api/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка эмбеддинга запроса DeepSeek API: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek query embeddings API error: {response.status_code}")
            
            result = response.json()
            return result.get("embedding", [])
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга запроса DeepSeek: {str(e)}")
            raise
    
    # Для совместимости с разными версиями llama-index
    def get_query_embedding(self, query: str) -> List[float]:
        """Публичный метод для получения эмбеддинга запроса"""
        return self._get_query_embedding(query)
    
    async def aget_query_embedding(self, query: str) -> List[float]:
        """Публичный асинхронный метод для получения эмбеддинга запроса"""
        return await self._aget_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Получить эмбеддинг для текста"""
        response = await self.client.embeddings([text], self._model)
        return response["data"][0]["embedding"]
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Синхронное получение эмбеддинга через requests"""
        import requests

        try:
            payload = {
                "model": self._model,
                "prompt": text
            }
            
            response = requests.post(
                f"{self._base_url}/api/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Ошибка эмбеддинга DeepSeek API: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek embeddings API error: {response.status_code}")
            
            result = response.json()
            return result.get("embedding", [])
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга DeepSeek: {str(e)}")
            raise
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получить эмбеддинги для списка текстов"""
        response = await self.client.embeddings(texts, self._model)
        return [item["embedding"] for item in response["data"]]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Синхронное получение эмбеддингов через requests"""
        import requests

        try:
            all_embeddings = []
            
            for text in texts:
                payload = {
                    "model": self._model,
                    "prompt": text
                }
                
                response = requests.post(
                    f"{self._base_url}/api/embeddings",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"Ошибка эмбеддинга DeepSeek API: {response.status_code} - {response.text}")
                    raise Exception(f"DeepSeek embeddings API error: {response.status_code}")
                
                result = response.json()
                embedding = result.get("embedding", [])
                all_embeddings.append(embedding)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов DeepSeek: {str(e)}")
            raise
    
    # Основные методы для llama-index
    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False, **kwargs) -> List[List[float]]:
        """Получить эмбеддинги для батча текстов (основной метод для llama-index)"""
        return self._get_text_embeddings(texts)
    
    # Для обратной совместимости
    async def aget_text_embedding(self, text: str) -> List[float]:
        """Получить эмбеддинг для текста"""
        return await self._aget_text_embedding(text)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Синхронное получение эмбеддинга"""
        return self._get_text_embedding(text)
    
    async def aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получить эмбеддинги для списка текстов"""
        return await self._aget_text_embeddings(texts)
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Синхронное получение эмбеддингов"""
        return self._get_text_embeddings(texts)
