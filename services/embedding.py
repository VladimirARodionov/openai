import logging
from pathlib import Path
import tiktoken
from llama_index.core import ServiceContext, StorageContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.couchbase import CouchbaseVectorStore
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

from create_bot import env_config

logger = logging.getLogger(__name__)

class EmbeddingsSearch:
    def __init__(self, embedding_model, model, user, password):
        """Инициализация с API ключом OpenAI и подключением к Couchbase"""
        self.EMBEDDING_MODEL = embedding_model
        self.GPT_MODEL = model
        
        # Инициализация Couchbase
        auth = PasswordAuthenticator(
            user,
            password
        )
        
        # Настройка подключения к кластеру
        options = ClusterOptions(
            auth
        )
        
        # Получаем параметры подключения из переменных окружения
        couchbase_host = env_config.get('COUCHBASE_HOST')
        couchbase_port = env_config.get('COUCHBASE_PORT')
        
        # Используем connection string с указанием всех необходимых портов
        connection_string = f"couchbase://{couchbase_host}"
        cluster = Cluster.connect(
            connection_string,
            ClusterOptions(PasswordAuthenticator(user, password))
        )
        
        # Создаем векторное хранилище
        self.vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name="vector_store",
            scope_name="_default",
            collection_name="_default",
            index_name="#primary"
        )
        
        # Инициализация llama-index
        self.service_context = ServiceContext.from_defaults(
            llm_model=model,
            embed_model=embedding_model
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.node_parser = SimpleNodeParser.from_defaults()

    def num_tokens(self, text):
        """Подсчет токенов в тексте"""
        encoding = tiktoken.encoding_for_model(self.GPT_MODEL)
        return len(encoding.encode(text))

    def query_message(self, query, token_budget=4000):
        """Создание сообщения для GPT с релевантными текстами"""
        strings, relatednesses = self.strings_ranked_by_relatedness(query)
        introduction = 'Используй приведенные ниже тексты для ответа на вопрос. Если ответ не найден в текстах, напиши "Не могу найти ответ."'
        question = f"\n\nВопрос: {query}"
        message = introduction
        
        for string in strings:
            next_article = f'\n\nТекст:\n"""\n{string}\n"""'
            if self.num_tokens(message + next_article + question) > token_budget:
                break
            message += next_article
        return message + question

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
                
                # Добавляем метаданные к узлам
                for node in nodes:
                    file_path = node.metadata.get('file_path', '')
                    if file_path:
                        file_path = Path(file_path)
                        node.metadata.update({
                            "file_path": str(file_path),
                            "file_type": file_path.suffix.lower().lstrip('.'),
                            "file_name": file_path.name,
                            "source": str(file_path),
                            "type": "vector"  # Добавляем тип для индекса
                        })
                
                # Создаем индекс и сохраняем в Couchbase
                index = VectorStoreIndex(
                    nodes,
                    storage_context=self.storage_context,
                    service_context=self.service_context,
                    show_progress=True  # Показываем прогресс индексации
                )
                
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
                self.vector_store,
                service_context=self.service_context
            )
            
            # Создаем движок для запросов
            query_engine = index.as_query_engine()
            
            # Получаем ответ
            response = query_engine.query(query)
            
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
            # Очищаем векторное хранилище
            self.vector_store.delete_all()
            logger.info("База данных очищена")
            return "База данных очищена"
        except Exception as e:
            logger.exception(f"Ошибка при очистке базы данных: {str(e)}")
            return f"Ошибка при очистке базы данных: {str(e)}"
