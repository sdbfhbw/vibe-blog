<div align="center">

# 博客生成平台

_Turn complex tech into stories everyone can understand._



[![Version](https://img.shields.io/badge/version-v0.1.0-4CAF50.svg)](https://github.com/sdbfhbw/vibe-blog)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)

</div>

vibe-blog 是一个基于多 Agent 工作流的长文内容生成项目，目标是把资料检索、内容规划、章节写作、代码生成、配图生成、质量审核和导出流程串成一个可运行的系统。

## 核心能力

- 多 Agent 协作：研究、规划、写作、追问、审校、组装分工明确
- 检索增强：支持联网检索、知识补全、来源筛选和引用整理
- 内容生成：支持长文、代码块、Mermaid 图表和 AI 配图
- 前后端联动：提供实时进度、Markdown 预览和多格式导出
- 扩展模块：包含知识库检索、教程评估、图文内容和发布相关能力

## 技术栈

### 后端

- Python 3.10+
- Flask
- LangGraph
- Jinja2
- Server-Sent Events

### 前端

- Vue 3
- Vite
- TypeScript
- Mermaid
- Vitest

### 相关服务

- OpenAI 兼容模型接口
- 智谱搜索
- 图像生成服务
- Langfuse 链路追踪

## 快速开始

### 方式一：Docker

```bash
cp backend/.env.example backend/.env
docker compose -f docker/docker-compose.yml up -d
```

启动后访问：

- 前端：`http://localhost:3000`
- API：`http://localhost:5000`

### 方式二：本地开发

1. 克隆仓库

```bash
git clone https://github.com/sdbfhbw/vibe-blog
cd vibe-blog
```

2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
pip install -r requirements-dev.txt
npm run install:frontend
```

3. 配置环境变量

```bash
cp backend/.env.example backend/.env
```

4. 启动后端

```bash
cd backend
python app.py
```

5. 启动前端

```bash
cd frontend
npm run dev
```

本地开发默认访问：

- 前端：`http://localhost:5173`
- API：`http://localhost:5001/api`

## 环境变量

完整配置见 [backend/.env.example](./backend/.env.example)。常用变量如下：

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI 兼容接口密钥 |
| `OPENAI_API_BASE` | OpenAI 兼容接口地址 |
| `TEXT_MODEL` | 文本生成模型 |
| `ZAI_SEARCH_API_KEY` | 智谱搜索密钥 |
| `NANO_BANANA_API_KEY` | 图像生成服务密钥 |
| `TRACE_ENABLED` | 是否启用 Langfuse 链路追踪 |

## 项目结构

```text
vibe-blog/
├── backend/
│   ├── app.py                         # Flask 应用入口
│   ├── routes/                        # API 路由
│   ├── services/
│   │   ├── blog_generator/            # 多 Agent 博客生成核心
│   │   │   ├── agents/                # Writer、Planner、Reviewer 等 Agent
│   │   │   ├── orchestrator/          # 工作流编排
│   │   │   ├── services/              # 搜索、抓取、来源筛选等能力
│   │   │   ├── tools/                 # Agent 工具封装
│   │   │   └── workflow_configs/      # 工作流配置
│   │   ├── chat/                      # 聊天与写作会话
│   │   ├── publishers/                # 发布平台适配
│   │   └── task_queue/                # 定时任务与队列
│   ├── eval/                          # 检索评估脚本与数据
│   ├── tests/                         # 后端测试
│   └── vibe_reviewer/                 # 教程质量评估模块
├── frontend/
│   ├── src/
│   │   ├── components/                # 页面组件
│   │   ├── composables/               # 组合式逻辑
│   │   ├── services/                  # 前端 API 封装
│   │   ├── stores/                    # 状态管理
│   │   └── views/                     # 页面视图
│   └── __tests__/                     # 前端测试
├── docker/                            # Docker 部署配置
├── docs/                              # 设计和部署文档
├── tests/                             # 端到端与跨模块测试
├── requirements.txt                   # 后端运行依赖入口
├── requirements-dev.txt               # 后端开发与测试依赖入口
└── package.json                       # 前端命令入口
```

## 测试

```bash
# 后端
pytest

# 前端
npm run test:frontend
```

## 相关文档

- [CHANGELOG.md](./CHANGELOG.md)
- [TESTING.md](./TESTING.md)
- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [SECURITY.md](./SECURITY.md)

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
