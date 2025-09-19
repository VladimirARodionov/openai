"""
Продвинутый семантический поиск для DeepSeek
Включает гибридный поиск, перерангирование и кластеризацию
"""

import logging
import re
import math
from typing import List, Dict, Any
from collections import defaultdict, Counter

from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Продвинутый семантический поиск с гибридным подходом"""
    
    def __init__(self, vector_index, llm, embed_model):
        self.vector_index = vector_index
        self.llm = llm
        self.embed_model = embed_model
        
        # Настройки поиска
        self.vector_top_k = 20  # Больше кандидатов для перерангирования
        self.final_top_k = 10   # Финальное количество результатов
        self.similarity_threshold = 0.7
        
        # Инициализируем компоненты
        self.retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self.vector_top_k,
        )
        
        # Постпроцессор для фильтрации по схожести
        self.similarity_postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_threshold
        )
        
    def expand_query(self, query: str) -> List[str]:
        """Интеллектуальное расширение запроса с помощью DeepSeek"""
        try:
            expansion_prompt = f"""
            Для данного запроса создай список связанных терминов, синонимов и ключевых слов на русском языке.
            Запрос: "{query}"
            
            Верни только список слов и фраз через запятую, без объяснений:
            """
            
            response = self.llm.complete(expansion_prompt)
            expanded_terms = [term.strip() for term in str(response).split(',')]
            
            # Добавляем оригинальный запрос
            all_terms = [query] + expanded_terms
            
            logger.info(f"Расширенный запрос: {all_terms}")
            return all_terms
            
        except Exception as e:
            logger.warning(f"Ошибка расширения запроса: {e}")
            return [query]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        # Простое извлечение ключевых слов
        # Удаляем стоп-слова и оставляем значимые термины
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'о', 'об',
            'что', 'как', 'где', 'когда', 'почему', 'который', 'которая', 'которое',
            'это', 'тот', 'та', 'то', 'все', 'всё', 'весь', 'вся', 'всю',
            'он', 'она', 'оно', 'они', 'его', 'её', 'их', 'ему', 'ей', 'им'
        }
        
        # Извлекаем слова длиннее 2 символов
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Возвращаем наиболее частые ключевые слова
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def calculate_bm25_score(self, query_terms: List[str], document_text: str, 
                           all_documents: List[str]) -> float:
        """Вычисление BM25 score для текстового поиска"""
        k1, b = 1.5, 0.75  # Параметры BM25
        
        doc_terms = self.extract_keywords(document_text)
        doc_length = len(doc_terms)
        
        if doc_length == 0:
            return 0.0
        
        # Средняя длина документа
        avg_doc_length = sum(len(self.extract_keywords(doc)) for doc in all_documents) / len(all_documents)
        
        score = 0.0
        for term in query_terms:
            term = term.lower().strip()
            if not term:
                continue
                
            # Частота термина в документе
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            
            # Количество документов содержащих термин
            df = sum(1 for doc in all_documents if term in self.extract_keywords(doc))
            if df == 0:
                continue
            
            # IDF
            idf = math.log((len(all_documents) - df + 0.5) / (df + 0.5))
            
            # BM25 формула
            term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
            score += term_score
        
        return score
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """Гибридный поиск: векторный + текстовый BM25"""
        try:
            # 1. Расширяем запрос
            expanded_queries = self.expand_query(query)
            
            # 2. Векторный поиск
            vector_nodes = self.retriever.retrieve(query)
            
            # 3. Получаем все тексты документов для BM25
            all_doc_texts = [node.node.text for node in vector_nodes]
            
            # 4. Вычисляем BM25 scores
            query_terms = []
            for q in expanded_queries:
                query_terms.extend(self.extract_keywords(q))
            
            hybrid_scores = []
            for node in vector_nodes:
                # Векторная схожесть (уже есть в node.score)
                vector_score = node.score if node.score else 0.0
                
                # BM25 score
                bm25_score = self.calculate_bm25_score(query_terms, node.node.text, all_doc_texts)
                
                # Комбинированный score (0.7 векторный + 0.3 BM25)
                hybrid_score = 0.7 * vector_score + 0.3 * bm25_score
                
                hybrid_scores.append((node, hybrid_score))
            
            # 5. Сортируем по комбинированному score
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 6. Создаем новые NodeWithScore с обновленными scores
            reranked_nodes = []
            for node, score in hybrid_scores[:top_k]:
                new_node = NodeWithScore(node=node.node, score=score)
                reranked_nodes.append(new_node)
            
            logger.info(f"Гибридный поиск вернул {len(reranked_nodes)} результатов")
            return reranked_nodes
            
        except Exception as e:
            logger.error(f"Ошибка гибридного поиска: {e}")
            # Fallback к обычному векторному поиску
            return self.retriever.retrieve(query)[:top_k]
    
    def semantic_clustering(self, nodes: List[NodeWithScore]) -> Dict[str, List[NodeWithScore]]:
        """Кластеризация результатов по семантическим темам"""
        if len(nodes) < 2:
            return {"main": nodes}
        
        try:
            # Простая кластеризация по ключевым словам
            clusters = defaultdict(list)
            
            for node in nodes:
                keywords = self.extract_keywords(node.node.text)
                
                # Определяем главную тему по наиболее частым ключевым словам
                if keywords:
                    main_topic = keywords[0]  # Берем самое частое ключевое слово
                    clusters[main_topic].append(node)
                else:
                    clusters["общее"].append(node)
            
            # Если слишком много мелких кластеров, объединяем их
            if len(clusters) > 5:
                main_clusters = {}
                other_nodes = []
                
                for topic, cluster_nodes in clusters.items():
                    if len(cluster_nodes) >= 2:
                        main_clusters[topic] = cluster_nodes
                    else:
                        other_nodes.extend(cluster_nodes)
                
                if other_nodes:
                    main_clusters["прочее"] = other_nodes
                
                return main_clusters
            
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}")
            return {"main": nodes}
    
    def search_with_context(self, query: str, context: str = "", top_k: int = 10) -> Dict[str, Any]:
        """Семантический поиск с учетом контекста"""
        try:
            # Если есть контекст, объединяем с запросом
            if context:
                enhanced_query = f"Контекст: {context}\n\nВопрос: {query}"
            else:
                enhanced_query = query
            
            # Выполняем гибридный поиск
            nodes = self.hybrid_search(enhanced_query, top_k * 2)  # Берем больше для кластеризации
            
            # Кластеризуем результаты
            clusters = self.semantic_clustering(nodes)
            
            # Готовим итоговые результаты
            results = {
                "query": query,
                "total_results": len(nodes),
                "clusters": {},
                "top_results": nodes[:top_k]
            }
            
            for cluster_name, cluster_nodes in clusters.items():
                results["clusters"][cluster_name] = {
                    "count": len(cluster_nodes),
                    "nodes": cluster_nodes[:5],  # Топ-5 из каждого кластера
                    "avg_score": sum(n.score for n in cluster_nodes) / len(cluster_nodes)
                }
            
            logger.info(f"Поиск завершен: {len(nodes)} результатов в {len(clusters)} кластерах")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            # Fallback к простому поиску
            simple_nodes = self.retriever.retrieve(query)[:top_k]
            return {
                "query": query,
                "total_results": len(simple_nodes),
                "clusters": {"main": {"count": len(simple_nodes), "nodes": simple_nodes, "avg_score": 0.0}},
                "top_results": simple_nodes
            }
    
    def explain_relevance(self, query: str, node: NodeWithScore) -> str:
        """Объяснение релевантности результата"""
        try:
            explanation_prompt = f"""
            Объясни кратко (1-2 предложения), почему этот фрагмент документа релевантен запросу.
            
            Запрос: "{query}"
            
            Фрагмент: "{node.node.text[:300]}..."
            
            Объяснение:
            """
            
            response = self.llm.complete(explanation_prompt)
            # Очищаем ответ от тегов <think> для объяснений релевантности
            clean_response = str(response).strip()
            
            # Удаляем теги <think> из объяснений релевантности
            import re
            clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL)
            clean_response = clean_response.strip()
            
            return clean_response
            
        except Exception as e:
            logger.warning(f"Ошибка объяснения релевантности: {e}")
            return f"Релевантность: {node.score:.2f}"
