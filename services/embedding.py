import ast
import logging
from pathlib import Path
from openai import OpenAI
import docx
import PyPDF2
import pandas as pd
import tiktoken
from scipy import spatial

logger = logging.getLogger(__name__)

class EmbeddingsSearch:
    def __init__(self, api_key, model):
        """Инициализация с API ключом OpenAI"""
        self.client = OpenAI(api_key=api_key)
        self.EMBEDDING_MODEL = "text-embedding-3-small"
        self.GPT_MODEL = model
        self.df = None

    def prepare_text_data(self, texts, file_paths=None):
        """Подготовка данных и создание эмбеддингов"""
        if file_paths is None:
            file_paths = [f"text_{i}" for i in range(len(texts))]
            
        embeddings = []
        try:
            for text in texts:
                response = self.client.embeddings.create(
                    model=self.EMBEDDING_MODEL,
                    input=text
                )
                embeddings.append(response.data[0].embedding)
        except Exception as e:
            logger.exception(str(e))

        self.df = pd.DataFrame({
            'text': texts,
            'embedding': embeddings,
            'file_path': file_paths
        })
        
    def load_embeddings(self, path):
        """Загрузка предварительно сохраненных эмбеддингов"""
        self.df = pd.read_csv(path)
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

    def save_embeddings(self, path):
        """Сохранение эмбеддингов в CSV"""
        self.df.to_csv(path, index=False)

    def strings_ranked_by_relatedness(self, query, top_n=5):
        """Поиск наиболее релевантных текстов"""
        query_embedding_response = self.client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding
        
        strings_and_relatednesses = [
            (row["text"], 1 - spatial.distance.cosine(query_embedding, row["embedding"]))
            for i, row in self.df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

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

    def ask(self, query, print_message=False):
        """Ответ на вопрос с использованием GPT и релевантных текстов"""
        message = self.query_message(query)
        if print_message:
            print(message)
            
        messages = [
            {"role": "system", "content": "Ты помощник, который отвечает на вопросы на основе предоставленных текстов."},
            {"role": "user", "content": message},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.exception(str(e))
            return "OpenAI вернул ошибку: " + str(e)

    def load_documents_from_directory(self, directory_path):
        """Загрузка документов из директории и создание эмбеддингов
        
        Поддерживаемые форматы файлов: .txt, .pdf, .doc, .docx
        
        Args:
            directory_path (str): Путь к директории с документами
        """
        texts = []
        file_paths = []
        
        for file_path in Path(directory_path).rglob('*'):
            if not file_path.is_file():
                continue
                
            try:
                if file_path.suffix.lower() == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        file_paths.append(str(file_path))
                        
                elif file_path.suffix.lower() == '.pdf':
                    text = ""
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    if text.strip():
                        texts.append(text)
                        file_paths.append(str(file_path))
                        
                elif file_path.suffix.lower() in ['.doc', '.docx']:
                    doc = docx.Document(file_path)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    if text.strip():
                        texts.append(text)
                        file_paths.append(str(file_path))
                        
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
                continue
        
        if texts:
            self.prepare_text_data(texts, file_paths)
        else:
            logger.warning(f"В директории {directory_path} не найдено поддерживаемых документов")
