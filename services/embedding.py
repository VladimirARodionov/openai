import logging
import time
from datetime import timedelta
from pathlib import Path
from multiprocessing import Process, Event
import telegram
from datetime import datetime

import openai
import tiktoken
from llama_cloud import MessageRole
from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, VectorStoreIndex, PromptTemplate, Document, SummaryIndex
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

from create_bot import env_config
from services.common import get_search_from_inet

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
CITATION_REFINE_TEMPLATE = PromptTemplate(read_from_file("templates/citation_refine_template.txt"))

CITATION_QA_TEMPLATE_INTERNET = PromptTemplate(read_from_file("templates/citation_qa_template_internet.txt"))

CITATION_REFINE_TEMPLATE_INTERNET = PromptTemplate(read_from_file("templates/citation_refine_template_internet.txt"))

INTERNET_QA_TEMPLATE = PromptTemplate(read_from_file("templates/internet_qa_template.txt"))
INTERNET_REPORT_TEMPLATE = PromptTemplate(read_from_file("templates/internet_report_template.txt"))

def _create_query_engine(index, top_k:int = 20):
    query_engine = CitationQueryEngine.from_args(
        index,
        citation_chunk_size=1024,
        similarity_top_k=top_k or env_config.get('SIMILARITY_TOP_K', 10),
        citation_qa_template=CITATION_QA_TEMPLATE,
        citation_refine_template=CITATION_REFINE_TEMPLATE,
        response_mode=ResponseMode.COMPACT
    )
    return query_engine


def _extract_topics(text: str) -> list[str]:
    """Извлечение основных тем из текста

    Args:
        text (str): Анализируемый текст

    Returns:
        list[str]: Список основных тем
    """
    try:
        # Создаем промпт для выделения тем
        prompt = PromptTemplate(
            "Проанализируй следующий текст и выдели 3-5 основных тем или ключевых понятий:\n"
            "{text}\n"
            "Темы:"
        )

        # Используем GPT для анализа
        response = Settings.llm.complete(prompt.format(text=text[:1000]))  # Ограничиваем длину текста

        # Разбираем ответ
        topics = [topic.strip().lower() for topic in str(response).split('\n') if topic.strip()]

        return topics

    except Exception as e:
        logger.warning(f"Ошибка при извлечении тем: {str(e)}")
        return []


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
            max_retries=3,
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

        self.bot = telegram.Bot(token=env_config.get('TELEGRAM_BOT_TOKEN'))
        self.loading_process = None
        self.stop_loading = Event()

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

    def _send_status_message(self, chat_id):
        """Отправка статуса загрузки в Telegram"""
        while not self.stop_loading.is_set():
            try:
                # Подсчет документов в базе
                count_query = f"SELECT COUNT(*) as count FROM `{self.vector_store._bucket_name}`.`{self.vector_store._scope_name}`.`{self.vector_store._collection_name}`"
                result = self.cluster.query(count_query).rows()
                doc_count = next(result)['count']
                
                message = f"📊 Статус загрузки документов:\nВсего документов в системе: {doc_count}\nВремя: {datetime.now().strftime('%H:%M:%S')}"
                self.bot.send_message(chat_id=chat_id, text=message)
                
                # Ждем 10 минут перед следующим обновлением
                for _ in range(600):  # 10 минут = 600 секунд
                    if self.stop_loading.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Ошибка при отправке статуса: {str(e)}")
                time.sleep(60)  # Подождем минуту при ошибке

    def _load_documents_process(self, directory_path, chat_id):
        """Процесс для загрузки документов"""
        try:
            # Создаем новое подключение к базе данных для процесса
            cluster = Cluster.connect(
                f"couchbase://{env_config.get('COUCHBASE_HOST')}",
                ClusterOptions(
                    PasswordAuthenticator(
                        env_config.get('COUCHBASE_ADMINISTRATOR_USERNAME'),
                        env_config.get('COUCHBASE_ADMINISTRATOR_PASSWORD')
                    ),
                    timeout_options=ClusterTimeoutOptions(
                        kv_timeout=timedelta(seconds=120),
                        query_timeout=timedelta(seconds=120),
                        search_timeout=timedelta(seconds=120)
                    )
                )
            )
            
            # Создаем новый экземпляр vector_store для процесса
            vector_store = CouchbaseVectorStore(
                cluster=cluster,
                bucket_name="vector_store",
                scope_name="_default",
                collection_name="_default",
                index_name="vector-index"
            )
            
            # Создаем новый storage_context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            
            result = self._load_documents_impl(directory_path, storage_context)
            # Отправляем финальное сообщение
            self.bot.send_message(chat_id=chat_id, text=f"✅ Загрузка завершена!\n{result}")
        except Exception as e:
            error_msg = f"❌ Ошибка при загрузке документов: {str(e)}"
            logger.exception(error_msg)
            self.bot.send_message(chat_id=chat_id, text=error_msg)
        finally:
            self.stop_loading.set()
            self.loading_process = None

    def load_documents_from_directory(self, directory_path, chat_id):
        """Асинхронная загрузка документов из директории"""
        if self.loading_process and self.loading_process.is_alive():
            return "Загрузка документов уже выполняется"

        self.stop_loading.clear()
        
        # Запускаем процесс для отправки статуса
        status_process = Process(
            target=self._send_status_message,
            args=(chat_id,)
        )
        status_process.daemon = True
        status_process.start()
        
        # Запускаем процесс для загрузки документов
        self.loading_process = Process(
            target=self._load_documents_process,
            args=(directory_path, chat_id)
        )
        self.loading_process.daemon = True
        self.loading_process.start()
        
        return "Загрузка документов начата в фоновом режиме"

    def _load_documents_impl(self, directory_path, storage_context=None):
        """Реализация загрузки документов"""
        try:
            openai.OpenAI._disable_logging = True
            logger.info("Начинаем загрузку документов")

            documents = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=True,
                filename_as_id=True,
                required_exts=[
                    ".txt", ".md", ".json", ".csv", ".html", ".xml",
                    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
                    ".py", ".js", ".java", ".cpp", ".h", ".c", ".cs", ".php", ".rb", ".swift", ".go"
                ],
                exclude_hidden=True
            ).load_data()

            if documents:
                # Обновляем метаданные документов
                logger.info("Обновляем метаданные документов")
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
                logger.info("Создаем индекс из документов целиком")
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context or self.storage_context,
                    show_progress=True
                )

                file_count = len(documents)
                logger.info(f"Обработано файлов: {file_count}")
                
                return f"Загружено {file_count} файлов"
            else:
                return "Не найдено поддерживаемых файлов в указанной директории"
            
        except Exception as e:
            logger.exception(f"Ошибка при загрузке документов: {str(e)}")
            raise
        finally:
            openai.OpenAI._disable_logging = False

    def ask(self, query, user_id, print_message=False):
        """Ответ на вопрос с использованием GPT и релевантных текстов"""
        try:
            search_from_inet = get_search_from_inet(user_id)
            response_parts = []
            
            # 1. Поиск в локальных документах
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            query_engine = _create_query_engine(index)
            local_response = query_engine.query(query)
            response_parts.append(str(local_response))
            
            # 2. Поиск в интернете через GPT, если включен
            if search_from_inet:
                try:
                    # Используем шаблон для поиска в интернете
                    internet_response = Settings.llm.complete(
                        INTERNET_QA_TEMPLATE.format(
                            query_str=query,
                            local_response=str(local_response)
                        )
                    )
                    
                    if str(internet_response).strip():
                        response_parts.append("\n\n🌐 Дополнительная информация из интернета:\n" + str(internet_response))
                
                except Exception as e:
                    logger.warning(f"Ошибка при поиске в интернете: {str(e)}")
                    response_parts.append("\n⚠️ Не удалось получить информацию из интернета")
            
            if print_message:
                logger.info(f"Query: {query}")
                logger.info(f"Response: {response_parts}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.exception(str(e))
            return f"Произошла ошибка: {str(e)}"

    def report(self, query: str, user_id, print_message=False):
        """Формирование детального отчета по запросу"""
        try:
            search_from_inet = get_search_from_inet(user_id)
            report_parts = []
            
            # Создаем индекс для поиска в локальных документах
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Получаем релевантные ноды с метаданными
            retriever = index.as_retriever(similarity_top_k=10)
            nodes = retriever.retrieve(query)
            
            # 1. Основной ответ из локальных документов
            query_engine = _create_query_engine(index)
            main_response = query_engine.query(query)
            report_parts.append(f"🔍 Основной ответ из документов:\n\n{str(main_response)}\n")
            
            # 2. Краткое саммари локальных документов
            if nodes:
                documents = [
                    Document(
                        text=node.text,
                        metadata=node.metadata
                    ) for node in nodes
                ]
                
                summary_index = SummaryIndex.from_documents(documents)
                summary = summary_index.as_query_engine().query(
                    "Создай краткое саммари найденной информации в 2-3 предложения"
                )
                report_parts.append(f"\n📝 Краткое саммари локальных документов:\n{str(summary)}\n")
            
            # 3. Поиск в интернете через GPT, если включен
            if search_from_inet:
                try:
                    # Используем шаблон для детального отчета
                    internet_response = Settings.llm.complete(
                        INTERNET_REPORT_TEMPLATE.format(
                            query_str=query,
                            local_response=str(main_response)
                        )
                    )
                    
                    if str(internet_response).strip():
                        report_parts.append("\n🌐 Информация из интернета:\n" + str(internet_response))
                
                except Exception as e:
                    logger.warning(f"Ошибка при поиске в интернете: {str(e)}")
                    report_parts.append("\n⚠️ Не удалось получить информацию из интернета")
            
            # 4. Источники информации из локальных документов
            sources = {}
            for node in nodes:
                source = node.metadata.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
            
            report_parts.append("\n📚 Основные локальные источники:")
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
                report_parts.append(f"- {source}: {count} релевантных фрагментов")
            
            return "\n".join(report_parts)
            
        except Exception as e:
            logger.exception(str(e))
            return f"Произошла ошибка при формировании отчета: {str(e)}"

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

            time.sleep(3)

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
