"""
文件解析服务 - 使用 MinerU 解析 PDF/文档

二期新增：
- 知识分块功能
- 图片摘要生成（多模态模型）
"""
import ast
import os
import re
import time
import uuid
import base64
import logging
import zipfile
import io
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Dict, Any

import requests
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# 初始化 Jinja2 模板环境
_templates_dir = Path(__file__).parent.parent / 'infrastructure' / 'prompts' / 'shared'
_jinja_env = Environment(loader=FileSystemLoader(str(_templates_dir)))


class FileParserService:
    """文件解析服务，支持 MinerU OCR 解析 PDF"""
    
    def __init__(
        self,
        mineru_token: str,
        mineru_api_base: str = "https://mineru.net",
        upload_folder: str = "",
        pdf_max_pages: int = 15
    ):
        """
        初始化文件解析服务
        
        Args:
            mineru_token: MinerU API Token
            mineru_api_base: MinerU API 基础 URL
            upload_folder: 上传文件存储目录
            pdf_max_pages: PDF 最大页数限制
        """
        self.mineru_token = mineru_token
        self.mineru_api_base = mineru_api_base
        self.upload_url_api = f"{mineru_api_base}/api/v4/file-urls/batch"
        self.result_api_template = f"{mineru_api_base}/api/v4/extract-results/batch/{{}}"
        
        self.upload_folder = upload_folder or str(Path(__file__).parent.parent / 'uploads')
        self.pdf_max_pages = pdf_max_pages
        
        logger.info(f"FileParserService 初始化完成, upload_folder={self.upload_folder}, pdf_max_pages={self.pdf_max_pages}")
    
    def parse_file(
        self, 
        file_path: str, 
        filename: str, 
        on_progress: Callable[[int, int, str, str], None] = None
    ) -> dict:
        """
        解析文件
        
        Args:
            file_path: 文件路径
            filename: 原始文件名
            on_progress: 进度回调函数 (step: int, total: int, message: str, detail: str)
            
        Returns:
            dict: {
                'success': bool,
                'batch_id': str | None,
                'markdown': str | None,
                'images': list[dict] | None,  # [{path, url, page_num}]
                'mineru_folder': str | None,  # MinerU 解析结果目录
                'error': str | None
            }
        """
        try:
            file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            
            # 纯文本文件直接读取
            if file_ext in ['txt', 'md', 'markdown']:
                logger.info(f"直接读取文本文件: {filename}")
                if on_progress:
                    on_progress(1, 1, "读取文本文件", filename)
                return self._parse_text_file(file_path)
            
            # PDF 文件检查页数限制
            if file_ext == 'pdf':
                page_count = self._get_pdf_page_count(file_path)
                if page_count > self.pdf_max_pages:
                    logger.warning(f"PDF 页数超限: {page_count} 页 (最大 {self.pdf_max_pages} 页)")
                    return {
                        'success': False,
                        'batch_id': None,
                        'markdown': None,
                        'images': None,
                        'mineru_folder': None,
                        'error': f'PDF 页数超过限制：{page_count} 页（最大支持 {self.pdf_max_pages} 页）'
                    }
                logger.info(f"PDF 页数检查通过: {page_count} 页")
            
            # 其他文件使用 MinerU 解析
            logger.info(f"使用 MinerU 解析文件: {filename}")
            return self._parse_with_mineru(file_path, filename, on_progress)
            
        except Exception as e:
            logger.error(f"文件解析异常: {e}", exc_info=True)
            return {
                'success': False,
                'batch_id': None,
                'markdown': None,
                'images': None,
                'mineru_folder': None,
                'error': str(e)
            }
    
    def _get_pdf_page_count(self, file_path: str) -> int:
        """获取 PDF 页数"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # 简单方法：统计 /Type /Page 出现次数（不包括 /Pages）
                # 更准确的方法需要用 PyPDF2，但这里用简单方法避免额外依赖
                # 匹配 /Type /Page 但不匹配 /Type /Pages
                pages = re.findall(rb'/Type\s*/Page[^s]', content)
                count = len(pages)
                if count == 0:
                    # 备用方法：查找 /Count 字段
                    count_match = re.search(rb'/Count\s+(\d+)', content)
                    if count_match:
                        count = int(count_match.group(1))
                logger.info(f"PDF 页数检测: {count} 页")
                return count if count > 0 else 1
        except Exception as e:
            logger.warning(f"无法获取 PDF 页数: {e}")
            return 0
    
    def _parse_text_file(self, file_path: str) -> dict:
        """解析纯文本文件"""
        try:
            # 尝试 UTF-8 编码
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 尝试 GBK 编码
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            
            logger.info(f"文本文件读取成功: {len(content)} 字符")
            
            return {
                'success': True,
                'batch_id': None,
                'markdown': content,
                'images': [],
                'mineru_folder': None,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'batch_id': None,
                'markdown': None,
                'images': None,
                'mineru_folder': None,
                'error': f"读取文本文件失败: {e}"
            }
    
    def _parse_with_mineru(
        self, 
        file_path: str, 
        filename: str, 
        on_progress: Callable = None
    ) -> dict:
        """使用 MinerU 解析文件"""
        # Step 1: 获取上传 URL
        logger.info("Step 1/3: 获取上传 URL...")
        if on_progress:
            on_progress(1, 3, "准备上传", f"正在获取上传地址...")
        batch_id, upload_url, error = self._get_upload_url(filename)
        if error:
            return {
                'success': False,
                'batch_id': None,
                'markdown': None,
                'images': None,
                'mineru_folder': None,
                'error': error
            }
        
        # Step 2: 上传文件
        logger.info(f"Step 2/3: 上传文件... batch_id={batch_id}")
        if on_progress:
            on_progress(2, 3, "上传文件", f"正在上传 {filename}...")
        error = self._upload_file(file_path, upload_url)
        if error:
            return {
                'success': False,
                'batch_id': batch_id,
                'markdown': None,
                'images': None,
                'mineru_folder': None,
                'error': error
            }
        
        # Step 3: 轮询解析结果
        logger.info("Step 3/3: 等待解析完成...")
        if on_progress:
            on_progress(3, 3, "解析文档", "MinerU 正在解析文档内容...")
        extract_id = str(uuid.uuid4())[:8]
        markdown, images, mineru_folder, error = self._poll_and_download(
            batch_id, extract_id, on_progress
        )
        if error:
            return {
                'success': False,
                'batch_id': batch_id,
                'markdown': None,
                'images': None,
                'mineru_folder': None,
                'error': error
            }
        
        return {
            'success': True,
            'batch_id': batch_id,
            'markdown': markdown,
            'images': images,
            'mineru_folder': mineru_folder,
            'error': None
        }
    
    def _get_upload_url(self, filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """从 MinerU 获取上传 URL"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mineru_token}"
        }
        
        payload = {
            "files": [{"name": filename}],
            "model_version": "vlm"
        }
        
        try:
            logger.info(f"请求 MinerU 上传 URL: {self.upload_url_api}")
            response = requests.post(
                self.upload_url_api,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"MinerU 响应: code={result.get('code')}, msg={result.get('msg')}")
            
            if result.get("code") != 0:
                error_msg = f"获取上传 URL 失败: {result.get('msg')}"
                logger.error(error_msg)
                return None, None, error_msg
            
            batch_id = result["data"]["batch_id"]
            upload_url = result["data"]["file_urls"][0]
            logger.info(f"成功获取上传 URL: batch_id={batch_id}")
            return batch_id, upload_url, None
            
        except requests.RequestException as e:
            error_msg = f"网络请求失败: {e}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg
        except Exception as e:
            error_msg = f"解析响应失败: {e}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg
    
    def _upload_file(self, file_path: str, upload_url: str) -> Optional[str]:
        """上传文件到 MinerU"""
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(
                    upload_url,
                    data=f,
                    timeout=300
                )
                response.raise_for_status()
            return None
        except requests.RequestException as e:
            return f"文件上传失败: {e}"
        except IOError as e:
            return f"文件读取失败: {e}"
    
    def _poll_and_download(
        self, 
        batch_id: str, 
        extract_id: str,
        on_progress: Callable = None,
        max_wait: int = 600
    ) -> Tuple[Optional[str], Optional[List[dict]], Optional[str], Optional[str]]:
        """轮询解析结果并下载"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mineru_token}"
        }
        
        result_url = self.result_api_template.format(batch_id)
        start_time = time.time()
        poll_count = 0
        
        while True:
            if time.time() - start_time > max_wait:
                return None, None, None, f"解析超时 ({max_wait}s)"
            
            try:
                response = requests.get(result_url, headers=headers, timeout=30)
                response.raise_for_status()
                task_info = response.json()
                
                if task_info.get("code") != 0:
                    return None, None, None, f"查询状态失败: {task_info.get('msg')}"
                
                state = task_info["data"]["extract_result"][0]["state"]
                
                if state == "done":
                    logger.info("解析完成，开始下载结果...")
                    if on_progress:
                        on_progress(3, 3, "下载结果", "解析完成，正在下载结果...")
                    zip_url = task_info["data"]["extract_result"][0]["full_zip_url"]
                    return self._download_and_extract(zip_url, extract_id)
                elif state == "failed":
                    err_msg = task_info["data"]["extract_result"][0].get("err_msg", "未知错误")
                    return None, None, None, f"解析失败: {err_msg}"
                else:
                    poll_count += 1
                    elapsed = int(time.time() - start_time)
                    logger.info(f"当前状态: {state}, 继续等待...")
                    if on_progress and poll_count % 3 == 0:  # 每 6 秒更新一次
                        on_progress(3, 3, "解析文档", f"MinerU 正在解析... 已等待 {elapsed} 秒")
                    time.sleep(2)
                    
            except requests.RequestException as e:
                logger.warning(f"轮询请求失败: {e}, 重试中...")
                time.sleep(2)
    
    def _download_and_extract(
        self, 
        zip_url: str, 
        extract_id: str
    ) -> Tuple[Optional[str], Optional[List[dict]], Optional[str], Optional[str]]:
        """下载并解压结果"""
        try:
            response = requests.get(zip_url, timeout=120)
            response.raise_for_status()
            
            # 创建存储目录
            storage_dir = Path(self.upload_folder) / 'mineru_files' / extract_id
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            markdown_content = None
            images = []
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(storage_dir)
                logger.info(f"解压 {len(z.namelist())} 个文件到 {storage_dir}")
                
                # 查找 Markdown 文件
                for name in z.namelist():
                    if name.lower().endswith('.md'):
                        md_path = storage_dir / name
                        with open(md_path, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()
                        logger.info(f"找到 Markdown 文件: {name}")
                        break
                
                # 收集图片文件
                for name in z.namelist():
                    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        img_path = storage_dir / name
                        # 生成访问 URL
                        url = f"/files/mineru/{extract_id}/{name}"
                        
                        # 尝试从文件名中提取页码
                        page_num = self._extract_page_num_from_filename(name)
                        
                        images.append({
                            'path': str(img_path),
                            'url': url,
                            'filename': os.path.basename(name),
                            'page_num': page_num
                        })
            
            if markdown_content is None:
                return None, None, None, "未找到 Markdown 文件"
            
            if not markdown_content.strip():
                return None, None, None, "PDF 解析结果为空，可能是扫描版 PDF 或内容无法识别"
            
            # 替换 Markdown 中的图片路径
            markdown_content = self._replace_image_paths(markdown_content, extract_id)
            
            return markdown_content, images, str(storage_dir), None
            
        except requests.RequestException as e:
            return None, None, None, f"下载结果失败: {e}"
        except zipfile.BadZipFile:
            return None, None, None, "下载的文件不是有效的 ZIP 文件"
        except Exception as e:
            return None, None, None, f"处理结果失败: {e}"
    
    def _extract_page_num_from_filename(self, filename: str) -> int:
        """
        从文件名中提取页码
        
        支持的格式:
        - page_1_xxx.png -> 1
        - 1_xxx.png -> 1
        - xxx_p1.png -> 1
        - xxx_page1.png -> 1
        - images/1/xxx.png -> 1 (从路径中提取)
        
        Returns:
            页码 (从 1 开始)，如果无法提取则返回 0
        """
        # 获取文件名（不含路径）
        basename = os.path.basename(filename)
        
        # 尝试多种模式
        patterns = [
            r'page[_-]?(\d+)',      # page_1, page-1, page1
            r'^(\d+)[_-]',          # 1_xxx, 1-xxx
            r'[_-]p(\d+)\.',        # xxx_p1.png
            r'[_-](\d+)\.',         # xxx_1.png
        ]
        
        for pattern in patterns:
            match = re.search(pattern, basename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # 尝试从路径中提取 (如 images/1/xxx.png)
        path_parts = filename.replace('\\', '/').split('/')
        for part in path_parts:
            if part.isdigit():
                return int(part)
        
        return 0  # 无法提取页码
    
    def _replace_image_paths(self, markdown: str, extract_id: str) -> str:
        """替换 Markdown 中的图片路径为本地服务 URL"""
        def replace_match(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # 跳过已经是 HTTP URL 的图片
            if img_path.startswith(('http://', 'https://')):
                return match.group(0)
            
            # 处理相对路径
            if img_path.startswith('/'):
                rel_path = img_path.lstrip('/')
            else:
                rel_path = img_path
            
            # 移除可能的 file/ 或 files/ 前缀
            for prefix in ['file/', 'files/']:
                if rel_path.startswith(prefix):
                    rel_path = rel_path[len(prefix):]
                    break
            
            new_url = f"/files/mineru/{extract_id}/{rel_path}"
            return f"![{alt_text}]({new_url})"
        
        pattern = r'!\[(.*?)\]\(([^\)]+)\)'
        return re.sub(pattern, replace_match, markdown)
    
    # ========== 二期新增：知识分块 ==========
    
    def chunk_markdown(
        self,
        markdown: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 Markdown 内容分块
        
        策略：
        1. 优先按标题构建 parent section，并保留 heading_path
        2. 在 parent 内按近似 token 数切 child chunk
        3. 如果单段过长，再按句子/窗口切分
        4. 保留 parent_id、heading_path、位置和 token_count
        
        Args:
            markdown: Markdown 内容
            chunk_size: 兼容旧配置；实际默认会换算为近似 token 上限
            chunk_overlap: 兼容旧配置；实际默认会换算为近似 token overlap
        
        Returns:
            分块列表，每个分块包含 {chunk_type, title, content, parent_id, heading_path, token_count, start_pos, end_pos}
        """
        chunks = []
        target_tokens = int(os.getenv('KNOWLEDGE_CHUNK_TOKEN_SIZE', str(max(300, chunk_size // 3))))
        overlap_tokens = int(os.getenv('KNOWLEDGE_CHUNK_TOKEN_OVERLAP', str(max(40, chunk_overlap // 3))))
        
        # 按标题构建 parent sections
        sections = self._split_by_headers(markdown)
        
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', '')
            start_pos = section.get('start_pos', 0)
            parent_id = section.get('parent_id', '')
            heading_path = section.get('heading_path', [])
            token_count = self._estimate_tokens(content)

            chunks.append({
                'chunk_type': 'parent',
                'title': title,
                'content': content,
                'parent_id': parent_id,
                'heading_path': heading_path,
                'token_count': token_count,
                'start_pos': start_pos,
                'end_pos': start_pos + len(content)
            })
             
            chunks.extend(
                self._split_section_content(
                    content=content,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    base_pos=start_pos,
                    parent_title=title,
                    parent_id=parent_id,
                    heading_path=heading_path,
                )
            )

        chunks.extend(self._build_image_chunks(markdown, images or [], sections))

        logger.info(f"Markdown 分块完成: {len(chunks)} 块")
        return chunks
    
    def _split_by_headers(self, markdown: str) -> List[Dict[str, Any]]:
        """按标题分割 Markdown"""
        sections = []
        
        # 匹配 # 到 ###### 标题，形成 heading path
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = markdown.split('\n')
        
        heading_stack = []
        section_index = 0
        current_section = {
            'title': '',
            'content': '',
            'start_pos': 0,
            'parent_id': 'parent_0',
            'heading_path': []
        }
        current_pos = 0
        
        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # 保存之前的 section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(title)
                section_index += 1

                # 开始新 parent section
                current_section = {
                    'title': title,
                    'content': line + '\n',
                    'start_pos': current_pos,
                    'parent_id': f'parent_{section_index}',
                    'heading_path': list(heading_stack)
                }
            else:
                current_section['content'] += line + '\n'
            
            current_pos += len(line) + 1  # +1 for newline
        
        # 保存最后一个 section
        if current_section['content'].strip():
            sections.append(current_section)
        
        # 如果没有找到任何标题，整个文档作为一个 section
        if not sections:
            sections.append({
                'title': '',
                'content': markdown,
                'start_pos': 0,
                'parent_id': 'parent_0',
                'heading_path': []
            })
        
        return sections

    def _split_section_content(
        self,
        content: str,
        target_tokens: int,
        overlap_tokens: int,
        base_pos: int,
        parent_title: str,
        parent_id: str,
        heading_path: List[str],
    ) -> List[Dict[str, Any]]:
        """Split one section into text/code/table child chunks."""
        chunks: List[Dict[str, Any]] = []
        text_buffer = ''
        text_start = base_pos

        def flush_text_buffer():
            nonlocal text_buffer, text_start
            if not text_buffer.strip():
                text_buffer = ''
                return
            token_count = self._estimate_tokens(text_buffer)
            if token_count <= target_tokens:
                chunks.append({
                    'chunk_type': 'section',
                    'title': parent_title,
                    'content': text_buffer.strip(),
                    'parent_id': parent_id,
                    'heading_path': heading_path,
                    'token_count': token_count,
                    'start_pos': text_start,
                    'end_pos': text_start + len(text_buffer),
                })
            else:
                chunks.extend(
                    self._split_by_paragraphs(
                        text_buffer,
                        target_tokens,
                        overlap_tokens,
                        text_start,
                        parent_title,
                        parent_id=parent_id,
                        heading_path=heading_path,
                    )
                )
            text_buffer = ''

        for block in self._iter_markdown_blocks(content, base_pos):
            block_type = block['type']
            if block_type == 'text':
                if not text_buffer:
                    text_start = block['start_pos']
                text_buffer += block['content']
                continue

            flush_text_buffer()
            if block_type == 'code':
                chunks.extend(
                    self._split_code_block(
                        block,
                        parent_title=parent_title,
                        parent_id=parent_id,
                        heading_path=heading_path,
                    )
                )
            elif block_type == 'table':
                chunks.extend(
                    self._split_table_block(
                        block,
                        target_tokens=target_tokens,
                        parent_title=parent_title,
                        parent_id=parent_id,
                        heading_path=heading_path,
                    )
                )

        flush_text_buffer()
        return chunks

    def _iter_markdown_blocks(self, content: str, base_pos: int):
        """Yield text, fenced-code, and markdown-table blocks in source order."""
        lines = content.splitlines(keepends=True)
        i = 0
        cursor = 0
        text_start = base_pos
        text_parts: List[str] = []

        def flush_text():
            nonlocal text_parts
            if text_parts:
                joined = ''.join(text_parts)
                yield {
                    'type': 'text',
                    'content': joined,
                    'start_pos': text_start,
                    'end_pos': text_start + len(joined),
                }
                text_parts = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith('```'):
                for item in flush_text():
                    yield item
                start = cursor
                language = stripped[3:].strip().split()[0] if stripped[3:].strip() else ''
                block_lines = [line]
                cursor += len(line)
                i += 1
                while i < len(lines):
                    block_lines.append(lines[i])
                    cursor += len(lines[i])
                    if lines[i].strip().startswith('```'):
                        i += 1
                        break
                    i += 1
                raw = ''.join(block_lines)
                yield {
                    'type': 'code',
                    'content': raw,
                    'language': language.lower(),
                    'start_pos': base_pos + start,
                    'end_pos': base_pos + start + len(raw),
                }
                text_start = base_pos + cursor
                continue

            if self._is_table_start(lines, i):
                for item in flush_text():
                    yield item
                start = cursor
                table_lines = []
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    cursor += len(lines[i])
                    i += 1
                raw = ''.join(table_lines)
                yield {
                    'type': 'table',
                    'content': raw,
                    'start_pos': base_pos + start,
                    'end_pos': base_pos + start + len(raw),
                }
                text_start = base_pos + cursor
                continue

            if not text_parts:
                text_start = base_pos + cursor
            text_parts.append(line)
            cursor += len(line)
            i += 1

        for item in flush_text():
            yield item

    @staticmethod
    def _is_table_start(lines: List[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        return (
            '|' in header
            and '|' in separator
            and re.match(r'^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$', separator) is not None
        )

    def _split_code_block(
        self,
        block: Dict[str, Any],
        parent_title: str,
        parent_id: str,
        heading_path: List[str],
    ) -> List[Dict[str, Any]]:
        raw = block['content']
        language = block.get('language', '')
        body = self._strip_code_fence(raw)
        units = self._split_code_by_syntax(body, language)
        chunks = []
        for index, unit in enumerate(units, 1):
            code_text = unit['content'].strip()
            if not code_text:
                continue
            content = self._wrap_code_fence(code_text, language)
            title_suffix = unit.get('name') or f"Code {index}"
            chunks.append({
                'chunk_type': 'code',
                'title': f"{parent_title} - {title_suffix}" if parent_title else title_suffix,
                'content': content,
                'parent_id': parent_id,
                'heading_path': heading_path,
                'token_count': self._estimate_tokens(content),
                'start_pos': block['start_pos'],
                'end_pos': block['end_pos'],
            })
        return chunks

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        return '\n'.join(lines)

    @staticmethod
    def _wrap_code_fence(code: str, language: str) -> str:
        return f"```{language or ''}\n{code.rstrip()}\n```"

    def _split_code_by_syntax(self, code: str, language: str) -> List[Dict[str, str]]:
        language = (language or '').lower()
        if language in {'py', 'python'}:
            units = self._split_python_code_with_ast(code)
            if units:
                return units

        units = self._split_code_with_tree_sitter(code, language)
        if units:
            return units

        return [{'name': '', 'content': code}]

    @staticmethod
    def _split_python_code_with_ast(code: str) -> List[Dict[str, str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        lines = code.splitlines()
        structural_nodes = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not structural_nodes:
            return []

        prefix = lines[:max(structural_nodes[0].lineno - 1, 0)]
        units = []
        for node in structural_nodes:
            end_lineno = getattr(node, 'end_lineno', None)
            if end_lineno is None:
                continue
            body_lines = lines[node.lineno - 1:end_lineno]
            unit_lines = [*prefix, *body_lines] if prefix else body_lines
            units.append({
                'name': getattr(node, 'name', ''),
                'content': '\n'.join(unit_lines).strip(),
            })
        return units

    @staticmethod
    def _split_code_with_tree_sitter(code: str, language: str) -> List[Dict[str, str]]:
        if not language:
            return []
        try:
            from tree_sitter_languages import get_parser
        except Exception:
            return []

        node_types = {
            'javascript': {'function_declaration', 'class_declaration', 'method_definition'},
            'typescript': {'function_declaration', 'class_declaration', 'method_definition'},
            'java': {'method_declaration', 'class_declaration'},
            'go': {'function_declaration', 'method_declaration', 'type_declaration'},
        }.get(language, set())
        if not node_types:
            return []

        try:
            parser = get_parser(language)
            tree = parser.parse(code.encode('utf-8'))
        except Exception:
            return []

        units = []
        for node in tree.root_node.children:
            if node.type not in node_types:
                continue
            text = code[node.start_byte:node.end_byte].strip()
            if text:
                units.append({'name': node.type, 'content': text})
        return units

    def _split_table_block(
        self,
        block: Dict[str, Any],
        target_tokens: int,
        parent_title: str,
        parent_id: str,
        heading_path: List[str],
    ) -> List[Dict[str, Any]]:
        rows = [line.rstrip('\n') for line in block['content'].splitlines() if line.strip()]
        if len(rows) < 2:
            return []

        header_rows = rows[:2]
        data_rows = rows[2:]
        groups = []
        current_rows = []
        for row in data_rows:
            candidate = '\n'.join([*header_rows, *current_rows, row])
            if current_rows and self._estimate_tokens(candidate) > target_tokens:
                groups.append(current_rows)
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows or not data_rows:
            groups.append(current_rows)

        chunks = []
        for index, group in enumerate(groups, 1):
            content = '\n'.join([*header_rows, *group])
            chunks.append({
                'chunk_type': 'table',
                'title': f"{parent_title} - Table {index}" if parent_title else f"Table {index}",
                'content': content,
                'parent_id': parent_id,
                'heading_path': heading_path,
                'token_count': self._estimate_tokens(content),
                'start_pos': block['start_pos'],
                'end_pos': block['end_pos'],
            })
        return chunks

    def _build_image_chunks(
        self,
        markdown: str,
        images: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        chunks = []
        for image_index, image in enumerate(images):
            display_index = image_index + 1
            caption = (image.get('caption') or '').strip()
            ocr_text = (image.get('ocr_text') or '').strip()
            alt_text = self._find_image_alt_text(markdown, image)
            if not any((caption, ocr_text, alt_text)):
                continue

            position = self._find_image_position(markdown, image)
            section = self._find_section_for_position(sections, position)
            parent_id = section.get('parent_id', 'parent_0') if section else 'parent_0'
            heading_path = section.get('heading_path', []) if section else []
            section_title = section.get('title', '') if section else ''
            content_parts = []
            if alt_text:
                content_parts.append(f"Alt text: {alt_text}")
            if caption:
                content_parts.append(f"Caption: {caption}")
            if ocr_text:
                content_parts.append(f"OCR: {ocr_text}")
            content = '\n'.join(content_parts)
            chunks.append({
                'chunk_type': 'image',
                'title': f"{section_title} - Image {display_index}" if section_title else f"Image {display_index}",
                'content': content,
                'image_index': image_index,
                'parent_id': parent_id,
                'heading_path': heading_path,
                'token_count': self._estimate_tokens(content),
                'start_pos': position if position >= 0 else 0,
                'end_pos': (position + len(content)) if position >= 0 else len(content),
            })
        return chunks

    @staticmethod
    def _find_image_position(markdown: str, image: Dict[str, Any]) -> int:
        for candidate in (image.get('url'), image.get('filename')):
            if candidate:
                position = markdown.find(candidate)
                if position >= 0:
                    return position
        return -1

    @staticmethod
    def _find_image_alt_text(markdown: str, image: Dict[str, Any]) -> str:
        candidates = [re.escape(v) for v in (image.get('url'), image.get('filename')) if v]
        if not candidates:
            return ''
        pattern = r'!\[(.*?)\]\((?:' + '|'.join(candidates) + r')\)'
        match = re.search(pattern, markdown)
        return match.group(1).strip() if match else ''

    @staticmethod
    def _find_section_for_position(
        sections: List[Dict[str, Any]],
        position: int,
    ) -> Optional[Dict[str, Any]]:
        if position < 0:
            return sections[0] if sections else None
        candidates = [section for section in sections if section.get('start_pos', 0) <= position]
        return candidates[-1] if candidates else (sections[0] if sections else None)
    
    def _split_by_paragraphs(
        self, 
        content: str, 
        chunk_size: int,
        chunk_overlap: int,
        base_pos: int,
        parent_title: str,
        parent_id: str = '',
        heading_path: List[str] = None
    ) -> List[Dict[str, Any]]:
        """按近似 token 数切分 parent 内长内容，单段过长时继续切分。"""
        chunks = []
        heading_path = heading_path or []
        
        # 按空行分割段落
        paragraphs = []
        cursor = 0
        for part in re.split(r'(\n\s*\n)', content):
            if not part:
                continue
            if part.strip():
                paragraphs.append((part.strip(), base_pos + cursor))
            cursor += len(part)
        
        current_chunk = ''
        current_start = base_pos
        chunk_index = 0
        
        for para, para_start in paragraphs:
            if not para:
                continue

            para_parts = (
                self._split_long_text_by_tokens(para, chunk_size, chunk_overlap)
                if self._estimate_tokens(para) > chunk_size
                else [para]
            )
            
            for part in para_parts:
                if self._estimate_tokens(current_chunk + part) <= chunk_size:
                    if not current_chunk:
                        current_start = para_start
                    current_chunk += part + '\n\n'
                    continue

                if current_chunk.strip():
                    token_count = self._estimate_tokens(current_chunk)
                    chunks.append({
                        'chunk_type': 'paragraph',
                        'title': f"{parent_title} (Part {chunk_index + 1})" if parent_title else f"Part {chunk_index + 1}",
                        'content': current_chunk.strip(),
                        'parent_id': parent_id,
                        'heading_path': heading_path,
                        'token_count': token_count,
                        'start_pos': current_start,
                        'end_pos': current_start + len(current_chunk)
                    })
                    chunk_index += 1

                overlap_text = self._tail_by_tokens(current_chunk, chunk_overlap)
                current_start = max(base_pos, current_start + len(current_chunk) - len(overlap_text))
                current_chunk = (overlap_text + '\n\n' if overlap_text else '') + part + '\n\n'
        
        # 保存最后一个分块
        if current_chunk.strip():
            token_count = self._estimate_tokens(current_chunk)
            chunks.append({
                'chunk_type': 'paragraph',
                'title': f"{parent_title} (Part {chunk_index + 1})" if parent_title else f"Part {chunk_index + 1}",
                'content': current_chunk.strip(),
                'parent_id': parent_id,
                'heading_path': heading_path,
                'token_count': token_count,
                'start_pos': current_start,
                'end_pos': current_start + len(current_chunk)
            })
        
        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count without adding tokenizer dependencies."""
        if not text:
            return 0
        cjk = len(re.findall(r'[\u4e00-\u9fff]', text))
        words = len(re.findall(r'[A-Za-z0-9_+\-./#]+', text))
        other = max(0, len(text) - cjk - sum(len(w) for w in re.findall(r'[A-Za-z0-9_+\-./#]+', text)))
        return cjk + words + other // 4

    def _split_long_text_by_tokens(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split one very long paragraph by sentence/window using approximate tokens."""
        sentences = [s for s in re.split(r'(?<=[。！？.!?])\s*', text) if s]
        parts = []
        current = ''
        for sentence in sentences:
            if self._estimate_tokens(sentence) > chunk_size:
                if current.strip():
                    parts.append(current.strip())
                    current = ''
                parts.extend(self._sliding_window_text(sentence, chunk_size, chunk_overlap))
                continue
            if self._estimate_tokens(current + sentence) <= chunk_size:
                current += sentence
            else:
                if current.strip():
                    parts.append(current.strip())
                overlap_text = self._tail_by_tokens(current, chunk_overlap)
                current = (overlap_text if overlap_text else '') + sentence
        if current.strip():
            parts.append(current.strip())
        return parts

    def _sliding_window_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fallback splitter for text with no usable sentence boundaries."""
        approx_chars = max(200, chunk_size * 2)
        overlap_chars = max(0, chunk_overlap * 2)
        step = max(1, approx_chars - overlap_chars)
        return [text[i:i + approx_chars] for i in range(0, len(text), step) if text[i:i + approx_chars].strip()]

    def _tail_by_tokens(self, text: str, token_count: int) -> str:
        if not text or token_count <= 0:
            return ''
        rough_chars = max(0, token_count * 2)
        return text[-rough_chars:]
    
    # ========== 二期新增：图片摘要 ==========
    
    def generate_image_captions(
        self, 
        images: List[Dict[str, Any]], 
        llm_service=None,
        max_images: int = 10
    ) -> List[Dict[str, Any]]:
        """
        为图片生成摘要描述
        
        Args:
            images: 图片列表，每个包含 {path, url, filename, page_num}
            llm_service: LLM 服务实例（需支持 vision 模型）
            max_images: 最多处理的图片数量
        
        Returns:
            带有 caption 的图片列表
        """
        if not llm_service:
            logger.warning("未提供 LLM 服务，跳过图片摘要生成")
            return images
        
        result = []
        processed = 0
        
        for img in images:
            if processed >= max_images:
                # 超过限制的图片不生成摘要
                result.append(img)
                continue
            
            img_path = img.get('path', '')
            if not img_path or not os.path.exists(img_path):
                result.append(img)
                continue
            
            try:
                # 读取图片并转为 base64
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # 确定 MIME 类型
                ext = os.path.splitext(img_path)[1].lower()
                mime_map = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                mime_type = mime_map.get(ext, 'image/jpeg')
                
                # 调用多模态模型生成描述
                template = _jinja_env.get_template('image_caption.j2')
                prompt = template.render(max_length=200)
                caption = llm_service.chat_with_image(prompt, img_base64, mime_type)
                ocr_text = llm_service.chat_with_image(
                    "Extract all visible text from this image. Return only the visible text. "
                    "If there is no readable text, return an empty string.",
                    img_base64,
                    mime_type,
                )

                if caption:
                    img['caption'] = caption
                if ocr_text:
                    img['ocr_text'] = ocr_text.strip()
                if caption or ocr_text:
                    logger.info(f"图片摘要生成成功: {img.get('filename', '')}")
                    processed += 1
                
            except Exception as e:
                logger.warning(f"图片摘要生成失败: {img_path}, 错误: {e}")
            
            result.append(img)
        
        logger.info(f"图片摘要生成完成: {processed}/{len(images)} 张")
        return result
    
    def generate_document_summary(
        self, 
        markdown: str, 
        llm_service=None,
        max_length: int = 500
    ) -> Optional[str]:
        """
        生成文档摘要（二期新增）
        
        Args:
            markdown: 文档 Markdown 内容
            llm_service: LLM 服务实例
            max_length: 摘要最大长度
        
        Returns:
            文档摘要
        """
        if not llm_service:
            logger.warning("未提供 LLM 服务，跳过文档摘要生成")
            return None
        
        try:
            # 截取前 4000 字符用于生成摘要
            content_preview = markdown[:4000] if len(markdown) > 4000 else markdown
            
            template = _jinja_env.get_template('document_summary.j2')
            prompt = template.render(max_length=max_length, content_preview=content_preview)
            
            summary = llm_service.chat([{"role": "user", "content": prompt}])
            
            if summary:
                # 确保不超过最大长度
                if len(summary) > max_length:
                    summary = summary[:max_length-3] + "..."
                logger.info(f"文档摘要生成成功: {len(summary)} 字")
                return summary
            
        except Exception as e:
            logger.error(f"文档摘要生成失败: {e}")
        
        return None


# 全局单例
_file_parser: Optional[FileParserService] = None


def get_file_parser() -> Optional[FileParserService]:
    """获取文件解析服务单例"""
    return _file_parser


def init_file_parser(
    mineru_token: str,
    mineru_api_base: str = "https://mineru.net",
    upload_folder: str = "",
    pdf_max_pages: int = 15
) -> FileParserService:
    """初始化文件解析服务"""
    global _file_parser
    _file_parser = FileParserService(
        mineru_token=mineru_token,
        mineru_api_base=mineru_api_base,
        upload_folder=upload_folder,
        pdf_max_pages=pdf_max_pages
    )
    return _file_parser


def create_file_parser_from_config(config) -> FileParserService:
    """从 Flask config 创建 FileParserService 实例"""
    return FileParserService(
        mineru_token=getattr(config, 'MINERU_TOKEN', ''),
        mineru_api_base=getattr(config, 'MINERU_API_BASE', 'https://mineru.net'),
        upload_folder=getattr(config, 'UPLOAD_FOLDER', '')
    )
