"""
Простая улучшенная система семантического поиска без кеширования
"""
import logging
import time
from datetime import timedelta, datetime
from pathlib import Path
from multiprocessing import Process, Event
from typing import List, Dict, Any
import asyncio

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode

from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.couchbase import CouchbaseVectorStore
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions

from services.common import get_search_from_inet, get_history, add_history
from create_bot import env_config
from locale_config import i18n

logger = logging.getLogger(__name__)

class SimpleEnhancedSearch:
    """
    Улучшенная система семантического поиска без кеширования
    Фокус на качестве поиска и простоте использования
    """
    
    def __init__(self):
        """Инициализация улучшенной системы поиска"""
        
        # Базовые настройки
        self.EMBEDDING_MODEL = env_config.get('EMBEDDING_MODEL', 'text-embedding-3-large')
        self.GPT_MODEL = env_config.get('MODEL', 'gpt-4o-mini')
        
        openai_api_key = env_config.get('OPEN_AI_TOKEN')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Инициализация LlamaIndex с улучшенными настройками
        Settings.llm = OpenAI(
            model=self.GPT_MODEL,
            api_key=openai_api_key,
            temperature=0.1,  # Более детерминированные ответы
            max_retries=3,
            timeout=60,
            request_timeout=60
        )
        
        # Используем более качественную модель эмбеддингов
        embedding_dimensions = 3072 if "3-large" in self.EMBEDDING_MODEL else 1536
        Settings.embed_model = OpenAIEmbedding(
            model=self.EMBEDDING_MODEL,
            api_key=openai_api_key,
            dimensions=embedding_dimensions,
            max_retries=3,
            timeout=120,  # Увеличиваем таймаут
            request_timeout=120,
            embed_batch_size=100  # Батчевая обработка
        )
        
        # Подключение к Couchbase
        self.cluster = self._get_cluster()
        self.vector_store = self._get_vector_store(self.cluster)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Настройки
        self.use_history = env_config.get('USE_HISTORY_IN_QUERIES', False)
        self.use_advanced_chunking = env_config.get('USE_ADVANCED_CHUNKING', True)
        self.use_query_enhancement = env_config.get('USE_QUERY_ENHANCEMENT', True)
        
        # Процессы загрузки
        self.loading_process = None
        self.stop_loading = Event()
        
        logger.info(f"Инициализирована улучшенная система поиска:")
        logger.info(f"- Модель эмбеддингов: {self.EMBEDDING_MODEL}")
        logger.info(f"- GPT модель: {self.GPT_MODEL}")
        logger.info(f"- Размерность эмбеддингов: {embedding_dimensions}")
        logger.info(f"- Продвинутый чанкинг: {self.use_advanced_chunking}")
        logger.info(f"- Улучшение запросов: {self.use_query_enhancement}")
    
    def _get_cluster(self):
        """Получение кластера Couchbase"""
        try:
            couchbase_host = env_config.get('COUCHBASE_HOST')
            connection_string = f"couchbase://{couchbase_host}"
            cluster = Cluster.connect(
                connection_string,
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
            return cluster
        except Exception as e:
            logger.exception(f"Ошибка подключения к Couchbase: {str(e)}")
            if "unambiguous_timeout" in str(e):
                raise Exception("Превышено время ожидания подключения к базе данных документов.")
            else:
                raise Exception(f"Ошибка подключения к базе данных: {str(e)}")
    
    def _get_vector_store(self, cluster):
        """Получение векторного хранилища"""
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name="vector_store",
            scope_name="_default",
            collection_name="_default",
            index_name="vector-index"
        )
        logger.info("Vector store initialized successfully")
        return vector_store
    
    def create_enhanced_pipeline(self, chunk_size: int = None) -> IngestionPipeline:
        """
        Создает улучшенный пайплайн обработки документов
        
        Args:
            chunk_size: Размер чанка (если None, используется адаптивный)
            
        Returns:
            IngestionPipeline: Настроенный пайплайн
        """
        transformations: List[Any] = []
        
        if self.use_advanced_chunking and chunk_size is None:
            # Семантический сплиттер - разбивает по смыслу
            semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model,
            )
            transformations.append(semantic_splitter)
            logger.info("Используется семантический чанкинг")
        else:
            # Обычный сплиттер с оптимизированными параметрами
            chunk_size = chunk_size or 512
            sentence_splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.1),  # 10% перекрытия
                paragraph_separator="\n\n",
                secondary_chunking_regex="[^,.;。]+[,.;。]?"
            )
            transformations.append(sentence_splitter)
            logger.info(f"Используется обычный чанкинг с размером {chunk_size}")
        
        # Добавляем экстракторы метаданных
        if env_config.get('USE_METADATA_EXTRACTION', True):
            try:
                title_extractor = TitleExtractor(nodes=3, llm=Settings.llm)
                transformations.append(title_extractor)
                logger.info("Добавлен экстрактор заголовков")
            except Exception as e:
                logger.warning(f"Не удалось добавить экстрактор заголовков: {str(e)}")
        
        # Добавляем эмбеддинг модель
        transformations.append(Settings.embed_model)
        
        pipeline = IngestionPipeline(transformations=transformations)
        return pipeline
    
    def enhance_query(self, query: str, num_variations: int = 2) -> List[str]:
        """
        Улучшает запрос, создавая дополнительные вариации
        
        Args:
            query: Исходный запрос
            num_variations: Количество вариаций
            
        Returns:
            List[str]: Список запросов включая оригинальный
        """
        if not self.use_query_enhancement:
            return [query]
        
        try:
            enhancement_prompt = f"""
            Исходный запрос: "{query}"
            
            Создай {num_variations} альтернативные формулировки этого запроса для улучшения поиска:
            1. Используй синонимы и близкие по смыслу слова
            2. Переформулируй вопрос по-другому
            3. Сделай запрос более конкретным или более общим
            
            Верни только альтернативные запросы, по одному на строке, без нумерации.
            """
            
            response = Settings.llm.complete(enhancement_prompt)
            variations = [line.strip() for line in str(response).split('\n') if line.strip()]
            
            # Возвращаем оригинальный запрос + вариации
            all_queries = [query] + variations[:num_variations]
            logger.info(f"Создано {len(all_queries)} вариантов запроса")
            return all_queries
            
        except Exception as e:
            logger.warning(f"Не удалось создать вариации запроса: {str(e)}")
            return [query]
    
    def create_advanced_query_engine(self, index: VectorStoreIndex, 
                                   similarity_top_k: int = 15,
                                   similarity_cutoff: float = 0.7,
                                   use_hyde: bool = False) -> RetrieverQueryEngine:
        """
        Создает продвинутый движок запросов
        
        Args:
            index: Векторный индекс
            similarity_top_k: Количество результатов
            similarity_cutoff: Порог схожести
            use_hyde: Использовать ли HyDE
            
        Returns:
            RetrieverQueryEngine: Настроенный движок
        """
        # Создаем ретривер
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )
        
        # Пост-процессоры для улучшения качества
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        ]
        
        # Синтезатор ответов
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            use_async=False,
            streaming=False
        )
        
        # Создаем движок запросов
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )
        
        # Добавляем HyDE если нужно
        if use_hyde:
            try:
                from llama_index.core.query_engine import TransformQueryEngine
                
                hyde_transform = HyDEQueryTransform(
                    llm=Settings.llm,
                    include_original=True
                )
                
                # Создаем query engine с HyDE трансформацией
                query_engine = TransformQueryEngine(
                    query_engine=query_engine,
                    query_transform=hyde_transform
                )
                logger.info("HyDE трансформация включена")
            except Exception as e:
                logger.warning(f"Не удалось включить HyDE: {str(e)}")
        
        return query_engine
    
    def ask(self, query: str, user_id: str, print_message: bool = False) -> str:
        """
        Улучшенный метод поиска
        
        Args:
            query: Поисковый запрос
            user_id: ID пользователя
            print_message: Выводить ли отладочную информацию
            
        Returns:
            str: Ответ на запрос
        """
        start_time = time.time()
        
        try:
            # 1. Получаем историю если включена
            query_with_history = query
            if self.use_history:
                history = get_history(user_id)
                if history:
                    history_context = "\n".join([
                        f"Предыдущий вопрос: {h.search_text}\nПредыдущий ответ: {h.answer_text}"
                        for h in history[-3:]  # Берем только последние 3 записи
                    ])
                    query_with_history = f"""
                    Контекст предыдущих вопросов:\n{history_context}\n\n
                    Текущий вопрос: {query}
                    """
            
            # 2. Создаем вариации запроса
            query_variations = self.enhance_query(query_with_history)
            
            # 3. Создаем индекс и движок запросов
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Определяем параметры поиска в зависимости от длины запроса
            words_count = len(query.split())
            if words_count <= 3:  # Короткий запрос
                similarity_top_k = 20
                similarity_cutoff = 0.6
            elif words_count >= 15:  # Длинный запрос
                similarity_top_k = 10
                similarity_cutoff = 0.8
            else:  # Средний запрос
                similarity_top_k = 15
                similarity_cutoff = 0.7
            
            query_engine = self.create_advanced_query_engine(
                index, 
                similarity_top_k=similarity_top_k,
                similarity_cutoff=similarity_cutoff,
                use_hyde=env_config.get('USE_HYDE', False)
            )
            
            # 4. Выполняем поиск по всем вариациям и выбираем лучший результат
            best_response = None
            best_score = 0
            
            for i, search_query in enumerate(query_variations):
                try:
                    response = query_engine.query(search_query)
                    response_text = str(response)
                    
                    # Простая оценка качества (длина + наличие источников)
                    score = len(response_text) * 0.1
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        score += len(response.source_nodes) * 10
                    
                    if score > best_score:
                        best_score = score
                        best_response = response
                    
                    if print_message:
                        logger.info(f"Вариант {i+1}: score={score:.1f}, длина={len(response_text)}")
                        
                except Exception as e:
                    logger.warning(f"Ошибка при поиске по варианту '{search_query[:50]}...': {str(e)}")
                    continue
            
            response_parts = []
            if best_response:
                response_parts.append(str(best_response))
            else:
                response_parts.append("Не удалось найти релевантную информацию в документах.")
            
            # 5. Поиск в интернете если включен
            search_from_inet = get_search_from_inet(user_id)
            if search_from_inet:
                try:
                    internet_prompt = f"""
                    Найди дополнительную информацию в интернете по запросу: {query}
                    
                    Информация из локальных документов: {response_parts[0][:500]}...
                    
                    Дополни и расширь информацию из надежных интернет-источников.
                    Укажи источники в формате (название сайта).
                    """
                    
                    internet_response = Settings.llm.complete(internet_prompt)
                    
                    if str(internet_response).strip():
                        response_parts.append(
                            "\n\n" + i18n.format_value('search_internet_title') + 
                            str(internet_response)
                        )
                
                except Exception as e:
                    logger.exception(f"Ошибка при поиске в интернете: {str(e)}")
                    response_parts.append("\n\n" + i18n.format_value('search_internet_error'))
            
            # 6. Формируем финальный ответ
            final_response = "".join(response_parts)
            
            # 7. Сохраняем в историю
            add_history(user_id, query, final_response, False, "enhanced_search")
            
            # 8. Логируем результаты
            query_time = time.time() - start_time
            if print_message:
                logger.info(f"Query: {query}")
                logger.info(f"Response time: {query_time:.2f}s")
                logger.info(f"Query variations: {len(query_variations)}")
                logger.info(f"Best score: {best_score:.1f}")
            
            return final_response
            
        except Exception as e:
            logger.exception(str(e))
            error_response = i18n.format_value('search_error', {'error': str(e)})
            add_history(user_id, query, error_response, True, "enhanced_search")
            return error_response
    
    def report(self, query: str, user_id: str, print_message: bool = False) -> str:
        """
        Создание детального отчёта по запросу с множественными источниками
        
        Args:
            query: Поисковый запрос
            user_id: ID пользователя
            print_message: Флаг вывода сообщений в лог
            
        Returns:
            str: Детальный отчёт
        """
        try:
            search_from_inet = get_search_from_inet(user_id)
            report_parts = []
            
            query_with_history = query
            
            # Получаем историю поиска только если включено в настройках
            if self.use_history:
                history = get_history(user_id)
                history_context = "\n".join([
                    f"Предыдущий вопрос: {h.search_text}\nПредыдущий ответ: {h.answer_text}"
                    for h in history
                ])
                
                # Добавляем историю в промпт
                if history_context:
                    query_with_history = f"""
                        История предыдущих вопросов и ответов:
                        {history_context}

                        Текущий вопрос с учетом контекста предыдущих вопросов:
                        {query}
                        """
            
            # Получаем кластер и создаём индекс
            cluster = self._get_cluster()
            vector_store = self._get_vector_store(cluster)
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            # 1. Основной ответ из локальных документов с улучшенным поиском
            enhanced_queries = self.enhance_query(query_with_history)
            all_nodes = []
            
            for enhanced_query in enhanced_queries:
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=8
                )
                nodes = retriever.retrieve(enhanced_query)
                all_nodes.extend(nodes)
            
            # Удаляем дубликаты и сортируем по релевантности
            unique_nodes = {}
            for node in all_nodes:
                node_id = node.node_id
                if node_id not in unique_nodes or node.score > unique_nodes[node_id].score:
                    unique_nodes[node_id] = node
            
            final_nodes = sorted(unique_nodes.values(), key=lambda x: x.score, reverse=True)[:15]
            
            # Создаём продвинутый query engine
            query_engine = self.create_advanced_query_engine(index, similarity_top_k=15, use_hyde=True)
            main_response = query_engine.query(query_with_history)
            report_parts.append(i18n.format_value('search_local_title') + '\n' + str(main_response) + '\n')
            
            # 2. Краткое саммари найденных документов
            if final_nodes:
                from llama_index.core import Document, SummaryIndex
                documents = [
                    Document(
                        text=node.text,
                        metadata=node.metadata
                    ) for node in final_nodes[:10]
                ]
                
                summary_index = SummaryIndex.from_documents(documents)
                summary = summary_index.as_query_engine().query(
                    "Создай краткое саммари найденной информации в 2-3 предложения"
                )
                report_parts.append(i18n.format_value('search_summary_title') + str(summary) + '\n')
            
            # 3. Поиск в интернете через GPT, если включен
            if search_from_inet:
                try:
                    # Читаем шаблон для интернет-отчёта
                    template_path = Path("templates/internet_report_template.txt")
                    if template_path.exists():
                        with open(template_path, 'r', encoding='utf-8') as f:
                            internet_template = f.read()
                        
                        internet_response = Settings.llm.complete(
                            internet_template.format(
                                query_str=query_with_history,
                                local_response=str(main_response)
                            )
                        )
                        
                        if str(internet_response).strip():
                            report_parts.append(i18n.format_value('search_internet_title') + str(internet_response))
                    else:
                        # Простой интернет-поиск без шаблона
                        internet_prompt = f"""
                        Локальный ответ: {main_response}
                        
                        Дополни этот ответ актуальной информацией из интернета по запросу: {query}
                        Укажи источники и дату информации, если возможно.
                        """
                        internet_response = Settings.llm.complete(internet_prompt)
                        report_parts.append(i18n.format_value('search_internet_title') + str(internet_response))
                
                except Exception as e:
                    logger.exception(f"Ошибка при поиске в интернете: {str(e)}")
                    report_parts.append(i18n.format_value('search_internet_error'))
            
            # 4. Источники информации из локальных документов
            sources = {}
            for node in final_nodes:
                source = node.metadata.get('file_name', node.metadata.get('source', 'Unknown'))
                sources[source] = sources.get(source, 0) + 1
            
            if sources:
                report_parts.append(i18n.format_value('search_sources_title'))
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report_parts.append(i18n.format_value('search_source_count', {
                        'source': source,
                        'count': count
                    }))
            
            final_report = "\n".join(report_parts)
            
            # Сохраняем запрос и ответ в историю
            add_history(user_id, query, final_report, False, "detailed_report")
            
            if print_message:
                logger.info(f"Generated detailed report for query: {query[:50]}...")
            
            return final_report
            
        except Exception as e:
            logger.exception(str(e))
            error_response = i18n.format_value('search_report_error', {'error': str(e)})
            # Сохраняем ошибку в историю
            add_history(user_id, query, error_response, True, "detailed_report")
            return error_response
    
    def load_documents_from_directory(self, directory_path: str, chat_id: str,
                                    chunk_size: int = None) -> str:
        """
        Загрузка документов с улучшенной обработкой
        
        Args:
            directory_path: Путь к директории с документами
            chat_id: ID чата для отправки статуса
            chunk_size: Размер чанка (если None, используется адаптивный)
            
        Returns:
            str: Сообщение о результате загрузки
        """
        if self.loading_process and self.loading_process.is_alive():
            return i18n.format_value('loading_already_running')

        self.stop_loading.clear()
        
        # Запускаем процесс загрузки
        self.loading_process = Process(
            target=self._load_documents_process_run,
            args=(directory_path, chat_id, chunk_size),
            daemon=True
        )
        self.loading_process.start()
        
        return i18n.format_value('loading_started')
    
    def _load_documents_process_run(self, directory_path: str, chat_id: str, chunk_size: int):
        """Запуск процесса загрузки документов"""
        asyncio.run(self._load_documents_process(directory_path, chat_id, chunk_size))
    
    async def _send_progress_ping(self, bot, chat_id: str, message: str, interval: int = 30):
        """Отправляет периодические сообщения о прогрессе"""
        while not self.stop_loading.is_set():
            await asyncio.sleep(interval)
            if not self.stop_loading.is_set():
                await bot.send_message(chat_id=chat_id, text=f"⏳ {message} (все еще работаем...)")
    
    async def _load_documents_process(self, directory_path: str, chat_id: str, chunk_size: int):
        """Процесс загрузки документов"""
        session = AiohttpSession()
        bot = Bot(
            token=env_config.get('TOKEN'),
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
            session=session
        )
        
        try:
            await bot.send_message(
                chat_id=chat_id, 
                text="🔄 Начинаем загрузку документов с улучшенной обработкой..."
            )
            
            # Загружаем документы
            await bot.send_message(
                chat_id=chat_id, 
                text="📖 Сканируем директорию и загружаем файлы..."
            )
            
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
            
            await bot.send_message(
                chat_id=chat_id, 
                text="✅ Файлы загружены, подготавливаем метаданные..."
            )
            
            if not documents:
                await bot.send_message(chat_id=chat_id, text=i18n.format_value('loading_no_files'))
                return
            
            await bot.send_message(
                chat_id=chat_id, 
                text=f"📁 Найдено {len(documents)} документов. Обрабатываем метаданные..."
            )
            
            # Обновляем метаданные с индикацией прогресса
            total_docs = len(documents)
            for i, doc in enumerate(documents, 1):
                file_path = Path(doc.metadata.get('file_path', ''))
                if file_path:
                    doc.doc_id = file_path.stem
                    doc.metadata.update({
                        "file_type": file_path.suffix.lower().lstrip('.'),
                        "file_name": file_path.name,
                        "source": file_path.stem,
                        "type": "vector",
                        "processed_at": datetime.now().isoformat(),
                        "processing_method": "enhanced"
                    })
                
                # Отправляем обновление прогресса каждые 10 документов
                if i % 10 == 0 or i == total_docs:
                    progress_percent = (i * 100) // total_docs
                    await bot.send_message(
                        chat_id=chat_id, 
                        text=f"📝 Обработано метаданных: {i}/{total_docs} ({progress_percent}%)"
                    )
            
            # Создаем улучшенный пайплайн
            await bot.send_message(
                chat_id=chat_id, 
                text="🔧 Создаем улучшенный пайплайн обработки..."
            )
            
            pipeline = self.create_enhanced_pipeline(chunk_size)
            
            processing_method = "семантическим чанкингом" if (self.use_advanced_chunking and chunk_size is None) else "стандартным чанкингом"
            await bot.send_message(
                chat_id=chat_id, 
                text=f"⚙️ Обрабатываем документы с {processing_method}...\n⏳ Это может занять несколько минут для больших документов."
            )
            
            # Создаем векторный индекс напрямую из документов (минуя ноды)
            await bot.send_message(
                chat_id=chat_id, 
                text=f"🔍 Создаем векторный индекс из {len(documents)} документов..."
            )
            
            # Проверяем подключение к Couchbase
            try:
                # Простая проверка подключения
                bucket = self.cluster.bucket("vector_store")
                collection = bucket.default_collection()
                await bot.send_message(
                    chat_id=chat_id, 
                    text="✅ Подключение к базе данных проверено"
                )
            except Exception as e:
                await bot.send_message(
                    chat_id=chat_id, 
                    text=f"⚠️ Проблема с подключением к БД: {str(e)}"
                )
            
            # Создаем индекс напрямую из документов с пайплайном
            index_start_time = time.time()
            try:
                # Используем from_documents с нашим пайплайном
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    transformations=pipeline.transformations,  # Используем наш улучшенный пайплайн
                    show_progress=True
                )
                index_time = time.time() - index_start_time
                
                await bot.send_message(
                    chat_id=chat_id, 
                    text=f"✅ Векторный индекс создан за {index_time:.1f} сек."
                )
            except Exception as e:
                await bot.send_message(
                    chat_id=chat_id, 
                    text=f"❌ Ошибка при создании индекса: {str(e)}"
                )
                raise
            
            file_count = len(documents)
            
            result_message = (
                "🎉 " + i18n.format_value('loading_complete') + '\n' +
                i18n.format_value('loading_files_count', {'count': file_count}) + '\n' +
                f"🔧 Метод: {'семантический' if (self.use_advanced_chunking and chunk_size is None) else 'стандартный'} чанкинг\n" +
                f"⏱️ Время создания индекса: {index_time:.1f} сек."
            )
            
            await bot.send_message(chat_id=chat_id, text=result_message)
            
        except Exception as e:
            error_msg = "❌ " + i18n.format_value('loading_error', {'error': str(e)})
            logger.exception(error_msg)
            await bot.send_message(chat_id=chat_id, text=error_msg)
        finally:
            self.stop_loading.set()
            self.loading_process = None
            await session.close()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Получает информацию о системе поиска
        
        Returns:
            Dict[str, Any]: Информация о системе
        """
        info = {
            "embedding_model": self.EMBEDDING_MODEL,
            "gpt_model": self.GPT_MODEL,
            "advanced_chunking": self.use_advanced_chunking,
            "query_enhancement": self.use_query_enhancement,
            "history_enabled": self.use_history,
            "vector_store": {
                "bucket_name": getattr(self.vector_store, '_bucket_name', 'vector_store'),
                "collection_name": getattr(self.vector_store, '_collection_name', '_default'),
                "scope_name": getattr(self.vector_store, '_scope_name', '_default'),
                "index_name": getattr(self.vector_store, '_index_name', 'vector-index')
            }
        }
        
        # Добавляем количество документов
        try:
            bucket_name = getattr(self.vector_store, '_bucket_name', 'vector_store')
            scope_name = getattr(self.vector_store, '_scope_name', '_default')
            collection_name = getattr(self.vector_store, '_collection_name', '_default')
            count_query = f"SELECT COUNT(*) as count FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            result = self.cluster.query(count_query).rows()
            doc_count = next(result)['count']
            info["document_count"] = doc_count
        except Exception as e:
            logger.error(f"Ошибка при получении количества документов: {str(e)}")
            info["document_count"] = "unknown"
        
        return info
    
    def clear_database(self) -> str:
        """
        Очистка базы данных (из оригинального кода)
        
        Returns:
            str: Результат операции
        """
        try:
            # Получаем доступ к коллекции
            bucket = self.cluster.bucket(self.vector_store._bucket_name)
            scope = bucket.scope(self.vector_store._scope_name)
            collection = scope.collection(self.vector_store._collection_name)
            
            # Проверяем количество документов
            count_query = f"SELECT COUNT(*) as count FROM `{self.vector_store._bucket_name}`.`{self.vector_store._scope_name}`.`{self.vector_store._collection_name}`"
            result = self.cluster.query(count_query).rows()
            initial_count = next(result)['count']
            logger.info(f"Documents before deletion: {initial_count}")
            
            # Получаем все ID документов
            id_query = f"SELECT META().id FROM `{self.vector_store._bucket_name}`.`{self.vector_store._scope_name}`.`{self.vector_store._collection_name}`"
            result = self.cluster.query(id_query).rows()
            
            # Удаляем документы
            deleted_count = 0
            for row in result:
                try:
                    collection.remove(row['id'])
                    deleted_count += 1
                except Exception as e:
                    logger.exception(f"Error deleting document {row['id']}: {str(e)}")
            
            logger.info(f"Deleted {deleted_count} documents")
            time.sleep(3)
            
            # Проверяем результат
            result = self.cluster.query(count_query).rows()
            final_count = next(result)['count']
            logger.info(f"Documents after deletion: {final_count}")
            
            if final_count == 0:
                return i18n.format_value('db_cleared')
            else:
                return i18n.format_value('db_clear_partial', {'count': final_count})
                
        except Exception as e:
            error_msg = i18n.format_value('db_clear_error', {'error': str(e)})
            logger.exception(error_msg)
            return error_msg
