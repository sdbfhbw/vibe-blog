"""
Researcher Agent - 素材收集
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional

from urllib.parse import urlparse

from ..prompts import get_prompt_manager
from ..services.smart_search_service import get_smart_search_service, init_smart_search_service
from ..utils.cache_utils import get_cache_manager

logger = logging.getLogger(__name__)


def _extract_domain(url: str) -> str:
    """从 URL 提取域名"""
    try:
        return urlparse(url).hostname or ''
    except Exception:
        return ''


class ResearcherAgent:
    """
    主题素材收集师 - 负责联网搜索收集背景资料
    支持文档知识融合（一期）
    """
    
    def __init__(self, llm_client, search_service=None, knowledge_service=None):
        """
        初始化 Researcher Agent

        Args:
            llm_client: LLM 客户端
            search_service: 搜索服务 (可选，如果不提供则跳过搜索)
            knowledge_service: 知识服务 (可选，用于文档知识融合)
        """
        self.llm = llm_client
        self.search_service = search_service
        self.knowledge_service = knowledge_service
        self.task_manager = None
        self.task_id = None

        # 初始化缓存管理器
        self.cache_enabled = os.environ.get('RESEARCHER_CACHE_ENABLED', 'true').lower() == 'true'
        if self.cache_enabled:
            self.cache = get_cache_manager()
            logger.info("💾 Researcher 缓存已启用")
        else:
            self.cache = None

        # 检查是否启用智能搜索
        self.smart_search_enabled = os.environ.get('SMART_SEARCH_ENABLED', 'false').lower() == 'true'
        if self.smart_search_enabled:
            # 初始化智能搜索服务
            smart_service = get_smart_search_service()
            if not smart_service:
                init_smart_search_service(llm_client)
            logger.info("🧠 智能知识源搜索已启用")

        # 75.03 深度抓取开关
        self.deep_scrape_enabled = os.environ.get('DEEP_SCRAPE_ENABLED', 'false').lower() == 'true'
        self._deep_scraper = None
        if self.deep_scrape_enabled:
            try:
                from ..services.deep_scraper import DeepScraper
                self._deep_scraper = DeepScraper(
                    jina_api_key=os.environ.get('JINA_API_KEY'),
                    llm_service=llm_client,
                    top_n=int(os.environ.get('DEEP_SCRAPE_TOP_N', '3')),
                )
                logger.info("🔗 深度抓取已启用 (Jina + httpx)")
            except Exception as e:
                logger.warning(f"深度抓取初始化失败: {e}")

        # 75.06 本地素材库开关
        self.local_material_enabled = os.environ.get('LOCAL_MATERIAL_ENABLED', 'false').lower() == 'true'
        self._material_store = None
        if self.local_material_enabled:
            try:
                from ..services.local_material_store import LocalMaterialStore
                material_dir = os.environ.get(
                    'LOCAL_MATERIAL_DIR',
                    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'materials')
                )
                self._material_store = LocalMaterialStore(base_dir=material_dir)
                logger.info(f"📦 本地素材库已启用: {material_dir}")
            except Exception as e:
                logger.warning(f"本地素材库初始化失败: {e}")

        # 102.08 配置驱动工具注册表（可选，默认 false）
        self._tool_registry = None
        if os.environ.get('TOOL_REGISTRY_ENABLED', 'false').lower() == 'true':
            try:
                from ..tools.registry import get_tool_registry
                self._tool_registry = get_tool_registry()
                available = self._tool_registry.list_tools()
                logger.info(f"102.08 ToolRegistry 已启用，已加载工具: {available}")
            except Exception as e:
                logger.warning(f"ToolRegistry 初始化失败，回退到硬编码路径: {e}")

        # 41.04 子查询并行研究引擎
        self.sub_query_enabled = os.environ.get('SUB_QUERY_ENABLED', 'false').lower() == 'true'
        self._sub_query_engine = None
        if self.sub_query_enabled:
            try:
                from ..services.sub_query_engine import SubQueryEngine
                self._sub_query_engine = SubQueryEngine(
                    llm_client=llm_client,
                    search_service=search_service,
                )
                logger.info("41.04 子查询并行研究引擎已启用")
            except Exception as e:
                logger.warning(f"子查询引擎初始化失败: {e}")

        # 41.03 语义压缩器
        self._semantic_compressor = None
        if os.environ.get('SEMANTIC_COMPRESS_ENABLED', 'false').lower() == 'true':
            try:
                from ..services.semantic_compressor import SemanticCompressor
                self._semantic_compressor = SemanticCompressor()
                logger.info("41.03 语义压缩器已启用")
            except Exception as e:
                logger.warning(f"语义压缩器初始化失败: {e}")

        # 41.01 深度研究引擎
        self._deep_research_engine = None
        if os.environ.get('DEEP_RESEARCH_ENABLED', 'false').lower() == 'true':
            try:
                from ..services.deep_research_engine import DeepResearchEngine
                self._deep_research_engine = DeepResearchEngine(
                    llm_client=llm_client,
                    search_service=search_service,
                )
                logger.info("41.01 深度研究引擎已启用")
            except Exception as e:
                logger.warning(f"深度研究引擎初始化失败: {e}")
    
    def generate_search_queries(self, topic: str, target_audience: str) -> List[str]:
        """
        生成搜索查询
        
        Args:
            topic: 技术主题
            target_audience: 目标受众
            
        Returns:
            搜索查询列表
        """
        # 默认搜索策略
        default_queries = [
            f"{topic} 教程 tutorial",
            f"{topic} 最佳实践 best practices",
            f"{topic} 常见问题 FAQ",
        ]
        
        if not self.llm:
            return default_queries
        
        try:
            pm = get_prompt_manager()
            prompt = pm.render_search_query(
                topic=topic,
                target_audience=target_audience
            )
            
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            queries = json.loads(response)
            if isinstance(queries, list):
                # 确保原始 topic 作为第一个 query（防止 LLM 改写主题）
                if queries and topic.lower() not in queries[0].lower():
                    queries.insert(0, topic)
                return queries
            return default_queries
            
        except Exception as e:
            logger.warning(f"生成搜索查询失败: {e}，使用默认查询")
            return default_queries
    
    def search(self, topic: str, target_audience: str, max_results: int = 10) -> List[Dict]:
        """
        执行搜索

        Args:
            topic: 技术主题
            target_audience: 目标受众
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        # 尝试从缓存获取
        if self.cache:
            cached_result = self.cache.get(
                'search',
                topic=topic,
                target_audience=target_audience,
                max_results=max_results
            )
            if cached_result is not None:
                return cached_result

        if not self.search_service:
            logger.warning("搜索服务未配置，跳过搜索")
            return []

        queries = self.generate_search_queries(topic, target_audience)
        all_results = []

        for query in queries:
            try:
                # 推送 search_started 事件
                if self.task_manager and self.task_id:
                    self.task_manager.send_event(self.task_id, 'result', {
                        'type': 'search_started',
                        'data': {'query': query, 'engine': 'zhipu'}
                    })
                result = self.search_service.search(query, max_results=max_results // len(queries))
                if result.get('success') and result.get('results'):
                    all_results.extend(result['results'])
                    # 推送 search_results 事件
                    if self.task_manager and self.task_id:
                        card_results = []
                        for r in result['results'][:10]:
                            url = r.get('url', '')
                            card_results.append({
                                'url': url,
                                'title': r.get('title', ''),
                                'snippet': (r.get('content', '') or r.get('snippet', ''))[:120],
                                'domain': _extract_domain(url),
                            })
                        self.task_manager.send_event(self.task_id, 'result', {
                            'type': 'search_results',
                            'data': {'query': query, 'results': card_results}
                        })
            except Exception as e:
                logger.error(f"搜索失败 [{query}]: {e}")

        # 去重
        seen_urls = set()
        unique_results = []
        for item in all_results:
            url = item.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(item)

        final_results = unique_results[:max_results]

        # 保存到缓存
        if self.cache:
            self.cache.set(
                'search',
                final_results,
                topic=topic,
                target_audience=target_audience,
                max_results=max_results
            )

        return final_results

    @staticmethod
    def _clean_search_results(results: List[Dict]) -> List[Dict]:
        """统一清洗搜索结果：去除 HTML 标签、修正 source/url 字段"""
        for item in results:
            # 修正 source 字段：优先使用 url
            if not item.get('source') or item.get('source') == '通用搜索':
                item['source'] = item.get('url', '通用搜索')
            # 确保 url 字段存在且优先
            if not item.get('url') and item.get('source', '') != '通用搜索':
                item['url'] = item.get('source', '')
            # 去除 HTML 标签
            for field in ('title', 'content', 'snippet'):
                if item.get(field):
                    item[field] = re.sub(r'<[^>]+>', '', item[field])
        return results

    def _smart_search(self, topic: str, target_audience: str, max_results: int = 15) -> List[Dict]:
        """
        使用智能搜索服务（LLM 路由 + 多源并行）

        Args:
            topic: 技术主题
            target_audience: 目标受众
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        # 尝试从缓存获取
        if self.cache:
            cached_result = self.cache.get(
                'smart_search',
                topic=topic,
                target_audience=target_audience,
                max_results=max_results
            )
            if cached_result is not None:
                return cached_result

        smart_service = get_smart_search_service()
        if not smart_service:
            logger.warning("智能搜索服务未初始化，回退到普通搜索")
            return self.search(topic, target_audience, max_results)

        try:
            result = smart_service.search(
                topic=topic,
                article_type=target_audience,
                max_results_per_source=5
            )

            if result.get('success'):
                sources_used = result.get('sources_used', [])
                logger.info(f"智能搜索完成，使用搜索源: {sources_used}")
                # 将搜索路由结果发送到前端
                if self.task_manager and self.task_id:
                    source_names = ', '.join(sources_used) if sources_used else '无'
                    self.task_manager.send_event(self.task_id, 'log', {
                        'logger': 'search_router',
                        'message': f'搜索路由决策: [{source_names}]，共 {len(result.get("results", []))} 条结果',
                    })
                search_results = result.get('results', [])[:max_results]

                # 保存到缓存
                if self.cache:
                    self.cache.set(
                        'smart_search',
                        search_results,
                        topic=topic,
                        target_audience=target_audience,
                        max_results=max_results
                    )

                return search_results
            else:
                logger.warning(f"智能搜索失败: {result.get('error')}，回退到普通搜索")
                return self.search(topic, target_audience, max_results)

        except Exception as e:
            logger.error(f"智能搜索异常: {e}，回退到普通搜索")
            return self.search(topic, target_audience, max_results)
    
    def summarize(
        self,
        topic: str,
        search_results: List[Dict],
        target_audience: str,
        search_depth: str = "medium"
    ) -> Dict[str, Any]:
        """
        整理搜索结果，生成背景知识摘要

        Args:
            topic: 技术主题
            search_results: 搜索结果
            target_audience: 目标受众
            search_depth: 搜索深度

        Returns:
            整理后的结果
        """
        if not search_results:
            return {
                "background_knowledge": f"关于 {topic} 的背景知识将在后续章节中详细介绍。",
                "key_concepts": [],
                "top_references": []
            }

        # 尝试从缓存获取（基于 topic 和搜索结果的 URL 列表）
        if self.cache:
            result_urls = [r.get('url', '') for r in search_results[:10]]
            cached_result = self.cache.get(
                'summarize',
                topic=topic,
                target_audience=target_audience,
                search_depth=search_depth,
                result_urls=result_urls
            )
            if cached_result is not None:
                return cached_result

        pm = get_prompt_manager()
        prompt = pm.render_researcher(
            topic=topic,
            search_depth=search_depth,
            target_audience=target_audience,
            search_results=search_results[:10]
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # 提取 JSON（处理 markdown 代码块）
            json_str = response.strip()
            if '```json' in json_str:
                start = json_str.find('```json') + 7
                end = json_str.find('```', start)
                json_str = json_str[start:end].strip() if end != -1 else json_str[start:].strip()
            elif '```' in json_str:
                start = json_str.find('```') + 3
                end = json_str.find('```', start)
                json_str = json_str[start:end].strip() if end != -1 else json_str[start:].strip()

            # 尝试解析 JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                result = json.loads(json_str, strict=False)
            key_concepts = result.get("key_concepts", [])

            # 调试：打印实际返回内容
            logger.info(f"LLM 返回 key_concepts 类型: {type(key_concepts)}, 值: {key_concepts}")

            # 如果 key_concepts 为空但有其他可能的字段名
            if not key_concepts:
                # 尝试其他可能的字段名
                for alt_key in ['keyConcepts', 'concepts', 'core_concepts', 'keywords']:
                    if result.get(alt_key):
                        key_concepts = result.get(alt_key)
                        logger.info(f"使用备选字段 {alt_key}: {key_concepts}")
                        break

            if key_concepts:
                logger.info(f"核心概念: {[c.get('name', c) if isinstance(c, dict) else c for c in key_concepts[:5]]}")

            # 解析 Instructional Design 分析（新增）
            instructional_analysis = result.get("instructional_analysis", {})
            if instructional_analysis:
                learning_objectives = instructional_analysis.get("learning_objectives", [])
                verbatim_data = instructional_analysis.get("verbatim_data", [])
                content_type = instructional_analysis.get("content_type", "tutorial")
                logger.info(f"📚 教学设计分析: 学习目标 {len(learning_objectives)} 个, "
                           f"Verbatim 数据 {len(verbatim_data)} 项, 内容类型: {content_type}")

            summary_result = {
                "background_knowledge": result.get("background_knowledge", ""),
                "key_concepts": key_concepts,
                "top_references": result.get("top_references", []),
                "instructional_analysis": instructional_analysis  # 新增
            }

            # 保存到缓存
            if self.cache:
                result_urls = [r.get('url', '') for r in search_results[:10]]
                self.cache.set(
                    'summarize',
                    summary_result,
                    topic=topic,
                    target_audience=target_audience,
                    search_depth=search_depth,
                    result_urls=result_urls
                )

            return summary_result

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}, 响应内容: {response[:500] if response else 'None'}")
        except Exception as e:
            logger.error(f"整理搜索结果失败: {e}")

        # 返回简单摘要
        return {
            "background_knowledge": '\n'.join([
                item.get('content', '')[:200] for item in search_results[:3]
            ]),
            "key_concepts": [],
            "top_references": [
                {"title": item.get('title', ''), "url": item.get('url', '')}
                    for item in search_results[:5]
                ]
            }
    
    def distill(self, topic: str, search_results: List[Dict]) -> Dict[str, Any]:
        """
        深度提炼搜索结果（类 OpenDraft Scribe）

        Args:
            topic: 技术主题
            search_results: 原始搜索结果

        Returns:
            提炼后的结构化素材
        """
        empty_result = {
            "sources": [],
            "common_themes": [],
            "contradictions": [],
            "material_by_type": {"concepts": [], "cases": [], "data": [], "comparisons": []}
        }
        if not search_results:
            return empty_result

        # 尝试从缓存获取
        if self.cache:
            result_urls = [r.get('url', '') for r in search_results[:15]]
            cached_result = self.cache.get(
                'distill',
                topic=topic,
                result_urls=result_urls
            )
            if cached_result is not None:
                return cached_result

        pm = get_prompt_manager()
        prompt = pm.render_distill_sources(
            topic=topic,
            search_results=search_results[:15]
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # 提取 JSON
            json_str = response.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()

            result = json.loads(json_str)

            # 确保必要字段存在
            result.setdefault('sources', [])
            result.setdefault('common_themes', [])
            result.setdefault('contradictions', [])
            result.setdefault('material_by_type',
                              {"concepts": [], "cases": [], "data": [], "comparisons": []})

            logger.info(f"🔬 深度提炼完成: {len(result['sources'])} 条素材, "
                        f"{len(result['common_themes'])} 个共同主题, "
                        f"{len(result['contradictions'])} 个矛盾点")

            # 保存到缓存
            if self.cache:
                result_urls = [r.get('url', '') for r in search_results[:15]]
                self.cache.set(
                    'distill',
                    result,
                    topic=topic,
                    result_urls=result_urls
                )

            return result

        except Exception as e:
            logger.error(f"深度提炼失败: {e}")
            return empty_result

    def analyze_gaps(self, topic: str, article_type: str, distilled: Dict[str, Any]) -> Dict[str, Any]:
        """
        缺口分析（类 OpenDraft Signal）

        Args:
            topic: 技术主题
            article_type: 文章类型
            distilled: distill() 的输出

        Returns:
            缺口分析结果
        """
        empty_result = {
            "content_gaps": [],
            "unique_angles": [],
            "writing_recommendations": {}
        }
        if not distilled or not distilled.get('sources'):
            return empty_result

        # 尝试从缓存获取
        if self.cache:
            cached_result = self.cache.get(
                'analyze_gaps',
                topic=topic,
                article_type=article_type,
                themes_count=len(distilled.get('common_themes', []))
            )
            if cached_result is not None:
                return cached_result

        pm = get_prompt_manager()
        prompt = pm.render_analyze_gaps(
            topic=topic,
            article_type=article_type,
            common_themes=distilled.get('common_themes', []),
            material_by_type=distilled.get('material_by_type', {}),
            contradictions=distilled.get('contradictions', [])
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # 提取 JSON
            json_str = response.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()

            result = json.loads(json_str)

            # 确保必要字段存在
            result.setdefault('content_gaps', [])
            result.setdefault('unique_angles', [])
            result.setdefault('writing_recommendations', {})

            logger.info(f"🔍 缺口分析完成: {len(result['content_gaps'])} 个缺口, "
                        f"{len(result['unique_angles'])} 个独特角度")

            # 保存到缓存
            if self.cache:
                self.cache.set(
                    'analyze_gaps',
                    result,
                    topic=topic,
                    article_type=article_type,
                    themes_count=len(distilled.get('common_themes', []))
                )

            return result

        except Exception as e:
            logger.error(f"缺口分析失败: {e}")
            return empty_result

    @staticmethod
    def _expand_neighbor_chunks(
        selected_chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        window: int = 1,
    ) -> List[Dict[str, Any]]:
        """Expand retrieved chunks with same-parent left/right neighbors."""
        if not selected_chunks or window <= 0:
            return selected_chunks

        by_position = {}
        for chunk in all_chunks:
            try:
                key = (
                    chunk.get('document_id'),
                    chunk.get('parent_id') or '',
                    int(chunk.get('chunk_index', -1)),
                )
            except (TypeError, ValueError):
                continue
            by_position[key] = chunk

        selected_ids = {chunk.get('id') for chunk in selected_chunks if chunk.get('id')}
        expanded = {}

        for chunk in selected_chunks:
            chunk_id = chunk.get('id')
            if chunk_id:
                expanded[chunk_id] = chunk

            try:
                base_index = int(chunk.get('chunk_index', -1))
            except (TypeError, ValueError):
                continue

            document_id = chunk.get('document_id')
            parent_id = chunk.get('parent_id') or ''
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor = by_position.get((document_id, parent_id, base_index + offset))
                if not neighbor or not neighbor.get('id'):
                    continue
                if neighbor['id'] not in expanded:
                    copied = dict(neighbor)
                    copied['_neighbor_expanded'] = True
                    copied['_expanded_from'] = chunk_id
                    expanded[neighbor['id']] = copied

        def sort_key(item: Dict[str, Any]):
            try:
                idx = int(item.get('chunk_index', 0))
            except (TypeError, ValueError):
                idx = 0
            return (str(item.get('document_id', '')), str(item.get('parent_id', '')), idx)

        ordered = sorted(expanded.values(), key=sort_key)
        # Keep a small signal on direct hits after document-order sorting.
        for item in ordered:
            item['_direct_retrieval_hit'] = item.get('id') in selected_ids
        return ordered

    @staticmethod
    def _expand_parent_chunks(
        selected_chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        fallback_neighbor_window: int = 1,
    ) -> List[Dict[str, Any]]:
        """Return full parent sections for matched child chunks when available."""
        if not selected_chunks:
            return []

        parent_by_key = {
            (chunk.get('document_id'), chunk.get('parent_id')): chunk
            for chunk in all_chunks
            if chunk.get('chunk_type') == 'parent' and chunk.get('parent_id')
        }
        expanded = []
        expanded_by_key = {}
        missing_parent_chunks = []

        for child in selected_chunks:
            key = (child.get('document_id'), child.get('parent_id'))
            parent = parent_by_key.get(key)
            child_id = child.get('id')
            if not parent:
                missing_parent_chunks.append(child)
                continue

            if key not in expanded_by_key:
                copied = dict(parent)
                copied['_parent_expanded'] = True
                copied['_matched_child_ids'] = []
                copied['_retrieval_sources'] = []
                copied['_direct_retrieval_hit'] = False
                expanded_by_key[key] = copied
                expanded.append(copied)

            item = expanded_by_key[key]
            if child_id and child_id not in item['_matched_child_ids']:
                item['_matched_child_ids'].append(child_id)

            for source in child.get('_retrieval_sources', []):
                if source not in item['_retrieval_sources']:
                    item['_retrieval_sources'].append(source)

            for score_key in (
                'relevance_score',
                '_bm25_score',
                '_rrf_score',
                '_rerank_score',
                '_cross_encoder_score',
            ):
                value = child.get(score_key)
                if value is None:
                    continue
                current = item.get(score_key)
                if current is None or value > current:
                    item[score_key] = value

        if missing_parent_chunks:
            child_chunks = [
                chunk for chunk in all_chunks
                if chunk.get('chunk_type') != 'parent'
            ]
            fallback_chunks = ResearcherAgent._expand_neighbor_chunks(
                missing_parent_chunks,
                child_chunks,
                window=fallback_neighbor_window,
            )
            seen_ids = {chunk.get('id') for chunk in expanded if chunk.get('id')}
            for chunk in fallback_chunks:
                if chunk.get('id') in seen_ids:
                    continue
                copied = dict(chunk)
                copied['_parent_expanded'] = False
                expanded.append(copied)

        return expanded

    @staticmethod
    def _expand_context_units(
        selected_chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        fallback_neighbor_window: int = 1,
    ) -> List[Dict[str, Any]]:
        """Expand text chunks to parents while keeping structured chunks direct."""
        direct_types = {'code', 'table', 'image'}
        direct_chunks = []
        text_chunks = []

        for chunk in selected_chunks:
            if chunk.get('chunk_type') in direct_types:
                copied = dict(chunk)
                copied['_context_unit'] = 'direct'
                copied['_direct_retrieval_hit'] = True
                direct_chunks.append(copied)
            else:
                text_chunks.append(chunk)

        expanded = ResearcherAgent._expand_parent_chunks(
            text_chunks,
            all_chunks,
            fallback_neighbor_window=fallback_neighbor_window,
        )
        for chunk in expanded:
            chunk.setdefault('_context_unit', 'parent')

        merged = []
        seen_ids = set()
        for chunk in [*direct_chunks, *expanded]:
            chunk_id = chunk.get('id')
            if chunk_id and chunk_id in seen_ids:
                continue
            if chunk_id:
                seen_ids.add(chunk_id)
            merged.append(chunk)
        return merged

    @staticmethod
    def _hydrate_image_vector_chunks(
        image_hits: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Map image-vector hits back to retrievable image chunks."""
        chunk_by_key = {
            (chunk.get('document_id'), chunk.get('image_index')): chunk
            for chunk in chunks
            if chunk.get('chunk_type') == 'image'
        }
        hydrated = []
        for image in image_hits:
            key = (image.get('document_id'), image.get('image_index'))
            chunk = chunk_by_key.get(key)
            if not chunk:
                continue
            item = dict(chunk)
            item['relevance_score'] = max(
                float(item.get('relevance_score') or 0.0),
                float(image.get('image_relevance_score') or 0.0),
            )
            item['_image_vector_score'] = image.get('image_relevance_score')
            item['_vector_store'] = image.get('_vector_store', 'multimodal_image')
            sources = list(item.get('_retrieval_sources', []))
            if 'image_vector' not in sources:
                sources.append('image_vector')
            item['_retrieval_sources'] = sources
            hydrated.append(item)
        return hydrated

    @classmethod
    def _rerank_document_chunks(
        cls,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Lightweight deterministic rerank after embedding recall.

        Score combines embedding similarity, query overlap with content/title,
        and heading path overlap. This avoids an extra LLM call while improving
        cases where local hash embeddings under-rank exact technical terms.
        """
        if not candidate_chunks:
            return []

        query_terms = cls._tokenize_for_rerank(query)
        scored = []
        for chunk in candidate_chunks:
            content = chunk.get('content', '') or ''
            title = chunk.get('title', '') or ''
            heading_path = chunk.get('heading_path', [])
            if isinstance(heading_path, str):
                try:
                    heading_path = json.loads(heading_path)
                except Exception:
                    heading_path = [heading_path]
            heading_text = ' '.join(str(part) for part in heading_path)

            content_terms = cls._tokenize_for_rerank(content)
            title_terms = cls._tokenize_for_rerank(title)
            heading_terms = cls._tokenize_for_rerank(heading_text)

            embedding_score = float(chunk.get('relevance_score') or 0.0)
            content_overlap = cls._term_overlap(query_terms, content_terms)
            title_overlap = cls._term_overlap(query_terms, title_terms)
            heading_overlap = cls._term_overlap(query_terms, heading_terms)

            rerank_score = (
                embedding_score * 0.55
                + content_overlap * 0.25
                + title_overlap * 0.12
                + heading_overlap * 0.08
            )

            item = dict(chunk)
            item['_rerank_score'] = round(rerank_score, 6)
            item['_term_overlap'] = round(content_overlap, 6)
            item['_title_overlap'] = round(title_overlap, 6)
            item['_heading_overlap'] = round(heading_overlap, 6)
            scored.append(item)

        scored.sort(key=lambda item: item.get('_rerank_score', 0.0), reverse=True)
        return scored[:top_k]

    @staticmethod
    def _tokenize_for_rerank(text: str) -> set:
        if not text:
            return set()
        text = text.lower()
        terms = set(re.findall(r'[a-z0-9_+\-./#]{2,}', text))
        terms.update(re.findall(r'[\u4e00-\u9fff]', text))
        return terms

    @staticmethod
    def _term_overlap(query_terms: set, doc_terms: set) -> float:
        if not query_terms or not doc_terms:
            return 0.0
        return len(query_terms & doc_terms) / max(len(query_terms), 1)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行素材收集

        支持两种模式：
        1. 无文档上传 → 原有流程（仅网络搜索）
        2. 有文档上传 → 知识融合流程（文档 + 网络搜索）

        Args:
            state: 共享状态

        Returns:
            更新后的状态
        """
        topic = state.get('topic', '')
        target_audience = state.get('target_audience', 'intermediate')
        
        # 获取文档知识（如果有上传文档）
        document_knowledge = state.get('document_knowledge', [])
        has_document = bool(document_knowledge)
        
        logger.info(f"🔍 开始收集素材: {topic}")
        
        # 展示文档知识（标题 + 预览内容分开）
        for doc in document_knowledge[:3]:
            file_name = doc.get('file_name', '未知文档')
            content = doc.get('content', '')
            # 标题行
            logger.info(f"📄 文档: {file_name} ({len(content)} 字)")
            # 预览内容（前1000字，作为单独的日志）
            preview = content[:1000] + '...' if len(content) > 1000 else content
            logger.info(f"__DOC_PREVIEW__{preview}__END_PREVIEW__")
        
        # 1. 执行网络搜索
        if self._sub_query_engine:
            # 41.04 子查询并行研究模式
            logger.info(f"🔬 启动子查询并行研究...")
            sq_result = self._sub_query_engine.run(
                topic=topic, target_audience=target_audience, max_results=15,
            )
            search_results = sq_result['results']
            state['sub_queries'] = sq_result['sub_queries']
            state['sub_query_stats'] = sq_result['stats']
            logger.info(
                f"🔬 子查询并行研究完成: {sq_result['stats']['sub_query_count']} 个子查询, "
                f"{sq_result['stats']['final_results']} 条结果"
            )
        elif self.smart_search_enabled:
            # 使用智能搜索（LLM 路由 + 多源并行）
            logger.info(f"🧠 启动智能知识源搜索...")
            search_results = self._smart_search(topic, target_audience)
        else:
            # 使用普通搜索
            logger.info(f"🌐 启动网络搜索...")
            search_results = self.search(topic, target_audience)

        # 统一清洗搜索结果（无论来自缓存还是实时搜索）
        search_results = self._clean_search_results(search_results)

        # 2. 知识融合分支
        if self.knowledge_service and has_document:
            # ✅ 有文档 → 走知识融合逻辑
            logger.info("使用知识融合模式")
            
            # 将搜索结果转换为 KnowledgeItem
            web_items = self.knowledge_service.convert_search_results(search_results)

            document_chunks = []
            document_images = []
            documents_for_v2 = []
            for doc in document_knowledge:
                doc_id = doc.get('id') or doc.get('document_id') or doc.get('file_name', '')
                documents_for_v2.append({
                    'id': doc_id,
                    'filename': doc.get('file_name', ''),
                    'summary': '',
                })
                document_chunks.extend(doc.get('chunks') or [])
                document_images.extend(doc.get('images') or [])

            if document_chunks:
                all_document_chunks = document_chunks
                retrievable_document_chunks = [
                    chunk for chunk in all_document_chunks
                    if chunk.get('chunk_type') != 'parent'
                ]
                document_chunks = retrievable_document_chunks
                retrieval_query_parts = [
                    topic,
                    state.get('article_type', ''),
                    target_audience,
                    " ".join(state.get('sub_queries', [])[:5]) if state.get('sub_queries') else "",
                ]
                retrieval_query = " ".join(part for part in retrieval_query_parts if part)
                top_k = int(os.environ.get('DOCUMENT_RETRIEVAL_TOP_K', '12'))
                candidate_k = int(os.environ.get('DOCUMENT_RETRIEVAL_CANDIDATE_K', str(max(top_k * 2, top_k))))
                neighbor_window = int(os.environ.get('DOCUMENT_NEIGHBOR_WINDOW', '1'))
                vector_backend = 'sqlite'
                retrieval_backend = 'hybrid'
                rerank_backend = 'rule'
                try:
                    from services.document_embedding_service import get_document_embedding_service
                    from services.document_hybrid_retrieval_service import (
                        get_document_hybrid_retrieval_service,
                    )
                    embedding_service = get_document_embedding_service()
                    hybrid_service = get_document_hybrid_retrieval_service()
                    vector_chunks = []
                    image_vector_chunks = []
                    try:
                        from services.document_vector_store_service import get_document_vector_store_service
                        vector_store = get_document_vector_store_service()
                        vector_chunks = vector_store.query_chunks(
                            retrieval_query,
                            retrievable_document_chunks,
                            embedding_service=embedding_service,
                            top_k=candidate_k
                        )
                        if vector_chunks:
                            vector_backend = 'chroma'
                    except Exception as e:
                        logger.warning(f"Chroma 文档检索失败，回退 SQLite 全量扫描: {e}")

                    if not vector_chunks:
                        vector_chunks = embedding_service.rank_chunks(
                            retrieval_query,
                            retrievable_document_chunks,
                            top_k=candidate_k
                        )

                    try:
                        from services.document_multimodal_embedding_service import (
                            get_document_multimodal_embedding_service,
                        )
                        multimodal_service = get_document_multimodal_embedding_service()
                        image_hits = []
                        if document_images and multimodal_service.is_available():
                            try:
                                image_hits = vector_store.query_images(
                                    retrieval_query,
                                    document_images,
                                    multimodal_service=multimodal_service,
                                    top_k=candidate_k,
                                )
                            except Exception as e:
                                logger.warning(f"Chroma 图片向量检索失败，回退 SQLite 图像扫描: {e}")
                            if not image_hits:
                                image_hits = multimodal_service.rank_images(
                                    retrieval_query,
                                    document_images,
                                    top_k=candidate_k,
                                )
                            image_vector_chunks = self._hydrate_image_vector_chunks(
                                image_hits,
                                retrievable_document_chunks,
                            )
                            if image_vector_chunks:
                                try:
                                    from services.multimodal_rerank_service import (
                                        get_multimodal_rerank_service,
                                    )
                                    multimodal_reranker = get_multimodal_rerank_service()
                                    reranked_image_chunks = multimodal_reranker.rerank_images(
                                        retrieval_query,
                                        image_vector_chunks,
                                        document_images,
                                        top_k=candidate_k,
                                    )
                                    if reranked_image_chunks:
                                        image_vector_chunks = reranked_image_chunks
                                except Exception as e:
                                    logger.warning(
                                        f"Multimodal image rerank failed, keep image vector order: {e}"
                                    )
                    except Exception as e:
                        logger.warning(f"图片多模态检索不可用，继续使用文本召回: {e}")

                    bm25_chunks = hybrid_service.rank_bm25(
                        retrieval_query,
                        retrievable_document_chunks,
                        top_k=candidate_k,
                    )
                    candidate_chunks = hybrid_service.reciprocal_rank_fusion(
                        {
                            'vector': vector_chunks,
                            'bm25': bm25_chunks,
                            'image_vector': image_vector_chunks,
                        },
                        top_k=candidate_k,
                    )

                    selected_chunks = []
                    try:
                        from services.cross_encoder_rerank_service import get_cross_encoder_rerank_service
                        cross_encoder = get_cross_encoder_rerank_service()
                        selected_chunks = cross_encoder.rerank(
                            retrieval_query,
                            candidate_chunks,
                            top_k=top_k
                        )
                        if selected_chunks:
                            rerank_backend = 'cross_encoder'
                    except Exception as e:
                        logger.warning(f"Cross-encoder rerank 失败，回退规则 rerank: {e}")

                    if not selected_chunks:
                        selected_chunks = self._rerank_document_chunks(
                            retrieval_query,
                            candidate_chunks,
                            top_k=top_k
                        )
                    if selected_chunks:
                        if os.environ.get('DOCUMENT_PARENT_CHILD_ENABLED', 'true').lower() == 'true':
                            document_chunks = self._expand_context_units(
                                selected_chunks,
                                all_document_chunks,
                                fallback_neighbor_window=neighbor_window,
                            )
                        else:
                            document_chunks = self._expand_neighbor_chunks(
                                selected_chunks,
                                retrievable_document_chunks,
                                window=neighbor_window
                            )
                    logger.info(
                        f"文档向量检索完成: candidates={len(candidate_chunks) if candidate_chunks else 0}, "
                        f"vector_hits={len(vector_chunks)}, bm25_hits={len(bm25_chunks)}, "
                        f"image_vector_hits={len(image_vector_chunks)}, "
                        f"rerank_hits={len(selected_chunks) if selected_chunks else 0}, "
                        f"expanded_chunks={len(document_chunks)}, top_k={top_k}, "
                        f"candidate_k={candidate_k}, "
                        f"retrieval_backend={retrieval_backend}, vector_backend={vector_backend}, "
                        f"rerank_backend={rerank_backend}, "
                        f"neighbor_window={neighbor_window}, query={retrieval_query[:120]}"
                    )
                except Exception as e:
                    logger.warning(f"文档向量检索失败，回退使用原始分块: {e}")
                    document_chunks = retrievable_document_chunks

                state['retrieved_document_chunks'] = [
                    {
                        'id': chunk.get('id'),
                        'document_id': chunk.get('document_id'),
                        'title': chunk.get('title', ''),
                        'chunk_type': chunk.get('chunk_type'),
                        'parent_id': chunk.get('parent_id'),
                        'heading_path': chunk.get('heading_path'),
                        'chunk_index': chunk.get('chunk_index'),
                        'image_index': chunk.get('image_index'),
                        'file_name': next(
                            (doc.get('file_name', '') for doc in document_knowledge
                             if (doc.get('id') or doc.get('document_id')) == chunk.get('document_id')),
                            ''
                        ),
                        'relevance_score': chunk.get('relevance_score', 0.0),
                        'bm25_score': chunk.get('_bm25_score'),
                        'rrf_score': chunk.get('_rrf_score'),
                        'retrieval_sources': chunk.get('_retrieval_sources', []),
                        'matched_child_ids': chunk.get('_matched_child_ids', []),
                        'parent_expanded': chunk.get('_parent_expanded', False),
                        'vector_rank': chunk.get('_vector_rank'),
                        'bm25_rank': chunk.get('_bm25_rank'),
                        'image_vector_rank': chunk.get('_image_vector_rank'),
                        'image_vector_score': chunk.get('_image_vector_score'),
                        'image_rerank_score': chunk.get('_image_rerank_score'),
                        'image_rerank_model': chunk.get('_image_rerank_model'),
                        'rerank_score': chunk.get('_rerank_score'),
                        'term_overlap': chunk.get('_term_overlap'),
                        'title_overlap': chunk.get('_title_overlap'),
                        'heading_overlap': chunk.get('_heading_overlap'),
                        'cross_encoder_score': chunk.get('_cross_encoder_score'),
                        'vector_store': chunk.get('_vector_store'),
                        'direct_retrieval_hit': chunk.get('_direct_retrieval_hit', False),
                        'neighbor_expanded': chunk.get('_neighbor_expanded', False),
                        'context_unit': chunk.get('_context_unit'),
                        'start_pos': chunk.get('start_pos'),
                        'end_pos': chunk.get('end_pos'),
                    }
                    for chunk in document_chunks
                ]
                logger.info(
                    f"使用知识融合 v2: docs={len(documents_for_v2)}, "
                    f"chunks={len(document_chunks)}, images={len(document_images)}"
                )
                merged_knowledge = self.knowledge_service.get_merged_knowledge_v2(
                    documents=documents_for_v2,
                    chunks=document_chunks,
                    images=document_images,
                    web_knowledge=web_items
                )
                summary = self.knowledge_service.summarize_for_prompt_v2(merged_knowledge)
            else:
                logger.info("文档分块为空，回退知识融合 v1")
                doc_items = self.knowledge_service.prepare_document_knowledge(
                    [{'filename': d.get('file_name', ''), 'markdown_content': d.get('content', '')}
                     for d in document_knowledge]
                )
                merged_knowledge = self.knowledge_service.get_merged_knowledge(
                    document_knowledge=doc_items,
                    web_knowledge=web_items
                )
                summary = self.knowledge_service.summarize_for_prompt(merged_knowledge)
            
            # 记录知识来源统计
            state['knowledge_source_stats'] = {
                'document_count': len([k for k in merged_knowledge if k.source_type == 'document']),
                'web_count': len([k for k in merged_knowledge if k.source_type == 'web_search']),
                'total_items': len(merged_knowledge)
            }
            state['document_references'] = summary.get('document_references', [])
            
        else:
            # ✅ 无文档 → 完全走原有逻辑，零改动
            logger.info("📋 使用原有搜索模式（无文档上传）")
            logger.info(f"📋 将使用网络搜索结果生成博客内容")

            # 41.01 深度研究：在初始搜索后迭代补充
            if self._deep_research_engine and search_results:
                logger.info("🔬 启动深度研究迭代...")
                dr_result = self._deep_research_engine.run(
                    topic=topic,
                    target_audience=target_audience,
                    initial_results=search_results,
                )
                search_results = dr_result['results']
                state['deep_research_stats'] = {
                    'rounds': dr_result['rounds'],
                    'total_queries': dr_result['total_queries'],
                    'coverage_score': dr_result['coverage_score'],
                }
                logger.info(
                    f"🔬 深度研究完成: {dr_result['rounds']} 轮, "
                    f"{len(search_results)} 条结果, 覆盖度 {dr_result['coverage_score']}%"
                )

            # 41.03 语义压缩：在 summarize 前压缩搜索结果
            compressed_results = search_results
            if self._semantic_compressor and search_results:
                compressed_results = self._semantic_compressor.compress(
                    query=topic, search_results=search_results,
                )

            summary = self.summarize(
                topic=topic,
                search_results=compressed_results,
                target_audience=target_audience
            )
            state['knowledge_source_stats'] = {
                'document_count': 0,
                'web_count': len(search_results),
                'total_items': len(search_results)
            }
            state['document_references'] = []
        
        # 3. 更新状态
        state['search_results'] = search_results
        # 句子级去重：消除 LLM summarize 输出的自我重复
        bg_raw = summary.get('background_knowledge', '')
        if bg_raw:
            sentences = [s.strip() for s in bg_raw.split('。') if s.strip()]
            seen = []
            for s in sentences:
                if s not in seen:
                    seen.append(s)
            bg_raw = '。'.join(seen) + ('。' if sentences else '')
        state['background_knowledge'] = bg_raw
        state['key_concepts'] = [
            c.get('name', c) if isinstance(c, dict) else c
            for c in summary.get('key_concepts', [])
        ]
        # 保留完整的引用信息（包含 title 和 url）
        state['reference_links'] = [
            r if isinstance(r, dict) else {'title': '', 'url': r}
            for r in summary.get('top_references', summary.get('web_references', []))
        ]
        
        # 4. 更新 Instructional Design 相关状态（新增）
        instructional_analysis = summary.get('instructional_analysis', {})
        state['instructional_analysis'] = instructional_analysis
        state['learning_objectives'] = instructional_analysis.get('learning_objectives', [])
        state['verbatim_data'] = instructional_analysis.get('verbatim_data', [])

        # 5. 本地素材库查询（75.06）
        if self._material_store and search_results:
            try:
                local_hits = self._material_store.search(topic, limit=5)
                if local_hits:
                    logger.info(f"📦 本地素材库命中 {len(local_hits)} 条")
                    for hit in local_hits:
                        search_results.append({
                            'title': hit.get('title', ''),
                            'url': hit.get('url', ''),
                            'content': hit.get('summary', ''),
                            'source': 'local_material',
                        })
            except Exception as e:
                logger.warning(f"本地素材库查询失败: {e}")

        # 6. 深度抓取 Top N 搜索结果（75.03）
        deep_scraped = []
        if self._deep_scraper and search_results:
            try:
                logger.info("🔗 开始深度抓取 Top N 搜索结果...")
                deep_scraped = self._deep_scraper.scrape_top_n(search_results, topic)
                if deep_scraped:
                    logger.info(f"🔗 深度抓取完成: {len(deep_scraped)} 篇高质量素材")
                    # 推送 crawl_completed 事件
                    if self.task_manager and self.task_id:
                        for item in deep_scraped:
                            url = item.get('url', '')
                            self.task_manager.send_event(self.task_id, 'result', {
                                'type': 'crawl_completed',
                                'data': {
                                    'url': url,
                                    'title': item.get('title', ''),
                                    'content_length': len(item.get('content', '') or item.get('summary', '')),
                                    'domain': _extract_domain(url),
                                }
                            })
            except Exception as e:
                logger.warning(f"深度抓取失败: {e}")

        # 7. 深度提炼 + 缺口分析（52号方案）
        distilled = {}
        gap_analysis = {}
        if search_results:
            logger.info("🔬 开始深度提炼搜索结果...")
            distilled = self.distill(topic, search_results)

            logger.info("🔍 开始缺口分析...")
            article_type = state.get('article_type', 'tutorial')
            gap_analysis = self.analyze_gaps(topic, article_type, distilled)

        state['distilled_sources'] = distilled.get('sources', [])
        # 清洗 distilled_sources 中的 HTML 标签（缓存可能包含旧数据）
        for src in state['distilled_sources']:
            for field in ('core_insight', 'title', 'key_facts'):
                val = src.get(field)
                if isinstance(val, str):
                    src[field] = re.sub(r'<[^>]+>', '', val)
                elif isinstance(val, list):
                    src[field] = [re.sub(r'<[^>]+>', '', v) if isinstance(v, str) else v for v in val]
        state['material_by_type'] = distilled.get('material_by_type', {})
        state['common_themes'] = distilled.get('common_themes', [])
        state['contradictions'] = distilled.get('contradictions', [])
        state['content_gaps'] = gap_analysis.get('content_gaps', [])
        state['unique_angles'] = gap_analysis.get('unique_angles', [])
        state['writing_recommendations'] = gap_analysis.get('writing_recommendations', {})
        state['deep_scraped_materials'] = deep_scraped  # 75.03 深度抓取素材

        stats = state['knowledge_source_stats']
        logger.info(f"✅ 素材收集完成: 文档知识 {stats['document_count']} 条, "
                    f"网络搜索 {stats['web_count']} 条, 核心概念 {len(state['key_concepts'])} 个")
        
        # 打印 Instructional Design 统计
        if instructional_analysis:
            logger.info(f"📚 教学设计: 学习目标 {len(state['learning_objectives'])} 个, "
                       f"Verbatim 数据 {len(state['verbatim_data'])} 项")
        
        # 输出 researcher 阶段结果（用于测试 mock）
        import json
        researcher_output = {
            'background_knowledge': state.get('background_knowledge', ''),
            'key_concepts': state.get('key_concepts', []),
            'reference_links': state.get('reference_links', []),
            'learning_objectives': state.get('learning_objectives', []),
            'verbatim_data': state.get('verbatim_data', []),
            'knowledge_source_stats': state.get('knowledge_source_stats', {}),
            'distilled_sources': state.get('distilled_sources', []),
            'content_gaps': state.get('content_gaps', []),
            'writing_recommendations': state.get('writing_recommendations', {}),
        }
        logger.info(f"__RESEARCHER_OUTPUT_JSON__{json.dumps(researcher_output, ensure_ascii=False)}__END_JSON__")
        
        return state
