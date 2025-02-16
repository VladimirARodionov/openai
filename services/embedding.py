import logging
from datetime import timedelta
from pathlib import Path

import openai
import tiktoken
from llama_cloud import MessageRole
from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.couchbase import CouchbaseVectorStore
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session

from create_bot import env_config, db
from db.models.model import User
from locale_config import i18n

logger = logging.getLogger(__name__)

def read_from_file(file_path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return "".join(f.readlines())


# Настраиваем логирование запросов OpenAI
original_post = openai.OpenAI.post
def logging_post(self, *args, **kwargs):
    # Проверяем флаг логирования в текущем экземпляре
    if hasattr(self, '_disable_logging') and self._disable_logging:
        return original_post(self, *args, **kwargs)
    
    logger.info(f"OpenAI request args: {args} {kwargs}")
    response = original_post(self, *args, **kwargs)
    logger.info(f"OpenAI response: {response}")
    return response

openai.OpenAI.post = logging_post

# Определяем шаблоны сообщений для чата

TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "Ты - помощник, который отвечает на вопросы, используя только предоставленную информацию. "
        "Если в контексте нет информации для ответа, скажи 'Не могу найти ответ в предоставленных документах.'"
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Контекст:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Ответь на вопрос, используя только информацию из контекста выше. "
            "Вопрос: {query_str}\n"
            "Ответ: "
        ),
        role=MessageRole.USER,
    ),
]

# Создаем шаблон чата
CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

CITATION_QA_TEMPLATE = PromptTemplate(read_from_file("templates/citation_qa_template.txt"))
print(CITATION_QA_TEMPLATE)
CITATION_REFINE_TEMPLATE = PromptTemplate(read_from_file("templates/citation_refine_template.txt"))

CITATION_QA_TEMPLATE_INTERNET = PromptTemplate(read_from_file("templates/citation_qa_template_internet.txt"))

CITATION_REFINE_TEMPLATE_INTERNET = PromptTemplate(read_from_file("templates/citation_refine_template_internet.txt"))

class EmbeddingsSearch:
    def __init__(self):
        """Инициализация с API ключом OpenAI и подключением к Couchbase"""
        self.EMBEDDING_MODEL = env_config.get('EMBEDDING_MODEL')
        self.GPT_MODEL = env_config.get('MODEL')
        
        # Устанавливаем API ключ OpenAI
        openai_api_key = env_config.get('OPEN_AI_TOKEN')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Инициализация llama-index settings с API ключом
        Settings.llm = OpenAI(
            model=self.GPT_MODEL,
            api_key=openai_api_key,
            max_retries=2,
            timeout=30,
            request_timeout=30
        )
        Settings.embed_model = OpenAIEmbedding(
            model=self.EMBEDDING_MODEL,
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
            ClusterOptions(PasswordAuthenticator(env_config.get('COUCHBASE_ADMINISTRATOR_USERNAME'),
                                                 env_config.get('COUCHBASE_ADMINISTRATOR_PASSWORD')),
                           timeout_options=ClusterTimeoutOptions(
                               kv_timeout=timedelta(seconds=120),
                               query_timeout=timedelta(seconds=120),
                               search_timeout=timedelta(seconds=120)
                           ))
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
            # Отключаем логирование OpenAI запросов
            openai.OpenAI._disable_logging = True
            
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
                # Обновляем метаданные документов
                for doc in documents:
                    file_path = Path(doc.metadata.get('file_path', ''))
                    if file_path:
                        # Используем имя файла без пути и расширения как идентификатор документа
                        doc.doc_id = file_path.stem
                        doc.metadata.update({
                            "file_type": file_path.suffix.lower().lstrip('.'),
                            "file_name": file_path.name,
                            "source": file_path.stem,
                            "type": "vector"
                        })

                # Создаем индекс из документов целиком
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    show_progress=True
                )

                file_count = len(documents)
                logger.info(f"Обработано файлов: {file_count}")
                
                # Включаем логирование обратно
                openai.OpenAI._disable_logging = False
                
                return f"Загружено {file_count} файлов"
            else:
                # Включаем логирование обратно
                openai.OpenAI._disable_logging = False
                return "Не найдено поддерживаемых файлов в указанной директории"
            
        except Exception as e:
            # Включаем логирование обратно даже в случае ошибки
            openai.OpenAI._disable_logging = False
            logger.exception(f"Ошибка при загрузке документов: {str(e)}")
            return f"Произошла ошибка при загрузке документов: {str(e)}"

    def ask(self, query, user_id, print_message=False):
        """Ответ на вопрос с использованием GPT и релевантных текстов"""
        search_from_inet = False
        session = Session(db)
        try:
            user = session.get(User, user_id)
            if not user:
                return i18n.format_value("user_not_found")
            search_from_inet = user.search_from_inet
        except Exception:
            logger.exception('Ошибка при проверке режима поиска из интернета')
            session.rollback()
        finally:
            session.close()
        try:
            # Создаем индекс для поиска
            index = VectorStoreIndex.from_vector_store(
                self.vector_store
            )
            
            # Создаем движок для запросов с нашим промптом
            query_engine = CitationQueryEngine.from_args(
                index,
                citation_chunk_size=1024,
                similarity_top_k=env_config.get('SIMILARITY_TOP_K', 10),
                citation_qa_template=CITATION_QA_TEMPLATE_INTERNET if search_from_inet else CITATION_QA_TEMPLATE,
                citation_refine_template=CITATION_REFINE_TEMPLATE_INTERNET if search_from_inet else CITATION_REFINE_TEMPLATE,
                response_mode=ResponseMode.COMPACT
            )
            
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
            # Получаем доступ к коллекции
            bucket = self.cluster.bucket(self.vector_store._bucket_name)
            scope = bucket.scope(self.vector_store._scope_name)
            collection = scope.collection(self.vector_store._collection_name)
            
            # Сначала проверяем количество документов
            count_query = f"SELECT COUNT(*) as count FROM `{self.vector_store._bucket_name}`.`{self.vector_store._scope_name}`.`{self.vector_store._collection_name}`"
            result = self.cluster.query(count_query).rows()
            initial_count = next(result)['count']
            logger.info(f"Documents before deletion: {initial_count}")
            
            # Получаем все ID документов
            id_query = f"SELECT META().id FROM `{self.vector_store._bucket_name}`.`{self.vector_store._scope_name}`.`{self.vector_store._collection_name}`"
            result = self.cluster.query(id_query).rows()
            
            # Удаляем документы по одному
            deleted_count = 0
            for row in result:
                try:
                    collection.remove(row['id'])
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting document {row['id']}: {str(e)}")
            
            logger.info(f"Deleted {deleted_count} documents")
            
            # Проверяем, что документы удалены
            result = self.cluster.query(count_query).rows()
            final_count = next(result)['count']
            logger.info(f"Documents after deletion: {final_count}")
            
            if final_count == 0:
                logger.info("База данных успешно очищена")
                return "База данных очищена"
            else:
                error_msg = f"Не все документы были удалены. Осталось: {final_count}"
                logger.error(error_msg)
                return error_msg
            
        except Exception as e:
            error_msg = f"Ошибка при очистке базы данных: {str(e)}"
            logger.exception(error_msg)
            return error_msg
