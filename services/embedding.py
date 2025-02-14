import logging
from pathlib import Path

import openai
import tiktoken
from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.couchbase import CouchbaseVectorStore
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from llama_index.llms.openai import OpenAI

from create_bot import env_config

logger = logging.getLogger(__name__)

class EmbeddingsSearch:
    def __init__(self, embedding_model, model, user, password):
        """Инициализация с API ключом OpenAI и подключением к Couchbase"""
        self.EMBEDDING_MODEL = embedding_model
        self.GPT_MODEL = model
        
        # Устанавливаем API ключ OpenAI
        openai_api_key = env_config.get('OPEN_AI_TOKEN')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Инициализация llama-index settings с API ключом
        Settings.llm = OpenAI(
            model=model,
            api_key=openai_api_key,
            max_retries=2,
            timeout=30,
            request_timeout=30
        )
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=openai_api_key,
            max_retries=2,
            timeout=30,
            request_timeout=30
        )

        # Получаем параметры подключения из переменных окружения
        couchbase_host = env_config.get('COUCHBASE_HOST')
        
        # Используем connection string с указанием всех необходимых портов
        connection_string = f"couchbase://{couchbase_host}"
        self.cluster = Cluster.connect(
            connection_string,
            ClusterOptions(PasswordAuthenticator(user, password))
        )
        
        try:
            # Проверяем доступные индексы
            mgr = self.cluster.search_indexes()
            indexes = mgr.get_all_indexes()
            logger.info(f"Available indexes: {[idx.name for idx in indexes]}")
            
            # Проверяем GSI индексы
            result = self.cluster.query(
                "SELECT * FROM system:indexes;"
            )
            gsi_indexes = [row for row in result]
            logger.info(f"Available GSI indexes in _default scope: {gsi_indexes}")
            
            # Создаем векторное хранилище
            self.vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name="vector_store",
                scope_name="_default",
                collection_name="_default",
                index_name="vector-index"
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
        
        # Инициализация storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.node_parser = SimpleNodeParser.from_defaults()

    def num_tokens(self, text):
        """Подсчет токенов в тексте"""
        encoding = tiktoken.encoding_for_model(self.GPT_MODEL)
        return len(encoding.encode(text))

    def _split_text_into_paragraphs(self, text: str) -> list[str]:
        """Разбивает текст на параграфы
        
        Args:
            text (str): Исходный текст

        Returns:
            list[str]: Список параграфов
        """
        # Разбиваем по двойным переносам строк
        paragraphs = [p.strip() for p in text.split('\n\n')]
        
        return paragraphs

    def load_documents_from_directory(self, directory_path):
        """Загрузка документов из директории, создание индекса и сохранение в Couchbase
        
        Поддерживаемые форматы:
        - Текстовые: .txt, .md, .json, .csv, .html, .xml, .pdf, .doc, .docx, .ppt, .pptx
        - Аудио: .mp3, .mp4, .mpeg, .mpga, .m4a, .wav, .webm
        - Изображения: .jpg, .jpeg, .png, .gif, .webp
        - Код: .py, .js, .java, .cpp, .h, .c, .cs, .php, .rb, .swift, .go
        
        Args:
            directory_path (str): Путь к директории с документами
        """
        try:
            file_count = 0
            # Используем SimpleDirectoryReader для загрузки всех поддерживаемых файлов
            documents = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=True,
                filename_as_id=True,
                required_exts=[
                    # Текстовые форматы
                    ".txt", ".md", ".json", ".csv", ".html", ".xml",
                    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
                    # Аудио форматы
                    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm",
                    # Форматы изображений
                    ".jpg", ".jpeg", ".png", ".gif", ".webp",
                    # Форматы кода
                    ".py", ".js", ".java", ".cpp", ".h", ".c", ".cs", ".php", ".rb", ".swift", ".go"
                ],
                exclude_hidden=True
            ).load_data()
            
            if documents:
                # Разбиваем на узлы
                nodes = self.node_parser.get_nodes_from_documents(documents)
                
                # Создаем эмбеддинги для всех узлов сразу
                embeddings = Settings.embed_model.get_text_embedding_batch(
                    [node.text for node in nodes]
                )
                
                # Добавляем метаданные и эмбеддинги к узлам
                for node, embedding in zip(nodes, embeddings):
                    file_path = node.metadata.get('file_path', '')
                    if file_path:
                        file_path = Path(file_path)
                        node.metadata.update({
                            "file_path": str(file_path),
                            "file_type": file_path.suffix.lower().lstrip('.'),
                            "file_name": file_path.name,
                            "source": str(file_path),
                            "type": "vector",  # Добавляем тип для индекса
                            "embedding": embedding  # Сохраняем эмбеддинг
                        })
                
                # Сохраняем узлы в Couchbase
                self.vector_store.add_nodes(nodes)  # Используем add_nodes вместо add
                
                file_count = len(documents)
                logger.info(f"Обработано файлов: {file_count}")
                
                return f"Загружено {file_count} файлов"
            else:
                return "Не найдено поддерживаемых файлов в указанной директории"
            
        except Exception as e:
            logger.exception(f"Ошибка при загрузке документов: {str(e)}")
            return f"Произошла ошибка при загрузке документов: {str(e)}"

    def ask(self, query, print_message=False):
        """Ответ на вопрос с использованием GPT и релевантных текстов"""
        try:
            # Создаем индекс для поиска
            index = VectorStoreIndex.from_vector_store(
                self.vector_store
            )
            
            # Создаем движок для запросов с кастомным поиском
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.response_synthesizers import get_response_synthesizer
            from llama_index.core.prompts import PromptTemplate
            from llama_index.core.response_synthesizers import ResponseMode
            
            # Создаем эмбеддинг для запроса один раз
            query_embedding = Settings.embed_model.get_text_embedding(query)
            
            # Создаем кастомный retriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=2,
                vector_store=self.vector_store,
            )
            retriever._get_query_embedding = lambda _: query_embedding
            
            # Создаем кастомный текст промпта
            qa_template = PromptTemplate(
                "Используй приведенные ниже тексты для ответа на вопрос.\n"
                "Если ответ не найден в текстах, напиши 'Не могу найти ответ.'\n"
                "Тексты: {context_str}\n"
                "Вопрос: {query_str}\n"
                "Ответ: "
            )
            
            # Создаем response synthesizer с нашим промптом
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode.COMPACT,
                text_qa_template=qa_template
            )
            
            # Создаем query engine с нашими компонентами
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )
            
            # Настраиваем логирование запросов OpenAI
            original_post = openai.OpenAI.post
            def logging_post(self, *args, **kwargs):
                logger.info(f"OpenAI request args: {args} {kwargs}")
                response = original_post(self, *args, **kwargs)
                logger.info(f"OpenAI response: {response}")
                return response
            openai.OpenAI.post = logging_post
            
            # Получаем ответ
            response = query_engine.query(query)
            
            # Восстанавливаем оригинальный метод
            openai.OpenAI.post = original_post
            
            if print_message:
                logger.info(f"Query: {query}")
                logger.info(f"Response: {response}")
            
            return str(response)
            
        except Exception as e:
            logger.exception(str(e))
            return f"Произошла ошибка: {str(e)}"

    def clear_database(self):
        """Очистка базы данных"""
        try:
            # Удаляем все документы
            query = f"DELETE FROM `{self.vector_store._bucket_name}`.`{self.vector_store._scope_name}`.`{self.vector_store._collection_name}`"
            self.cluster.query(query)
            
            logger.info("База данных очищена")
            return "База данных очищена"
        except Exception as e:
            logger.exception(f"Ошибка при очистке базы данных: {str(e)}")
            return f"Ошибка при очистке базы данных: {str(e)}"
