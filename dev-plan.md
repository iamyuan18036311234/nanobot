# nanobot 从0到1学习计划

> **项目**: nanobot - Ultra-Lightweight Personal AI Assistant
> **GitHub**: https://github.com/HKUDS/nanobot
> **核心特点**: 仅约4000行代码实现核心功能，比Clawdbot的430k+行小99%

---

## 一、项目概览

### 1.1 什么是 nanobot？

nanobot 是一个**超轻量级个人AI助手框架**，受 Clawdbot 启发但大幅简化：

- **核心代码量**: ~3,448行（可通过 `bash core_agent_lines.sh` 验证）
- **语言**: Python 3.11+
- **架构**: 异步事件驱动，模块化设计
- **能力**:
  - 24/7实时市场分析
  - 全栈软件工程师助手
  - 智能日程管理
  - 个人知识助手

### 1.2 项目架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         nanobot 架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐      ┌─────────────────┐                 │
│  │   聊天平台        │      │   CLI命令行      │                 │
│  │ Telegram/Discord │      │   直接交互       │                 │
│  │ WhatsApp/Feishu  │      │                 │                 │
│  └────────┬─────────┘      └────────┬────────┘                 │
│           │                          │                           │
│           ▼                          ▼                           │
│  ┌───────────────────────────────────────────────────┐          │
│  │              Channel Manager                       │          │
│  │  (channels/manager.py)                             │          │
│  └─────────────────────┬─────────────────────────────┘          │
│                        │ InboundMessage                        │
│                        ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │              MessageBus (消息总线)                  │          │
│  │  (bus/queue.py)                                    │          │
│  └─────────────────────┬─────────────────────────────┘          │
│                        │                                       │
│                        ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │              AgentLoop (核心引擎)                   │          │
│  │  (agent/loop.py)                                   │          │
│  │  ┌─────────────────────────────────────────────┐  │          │
│  │  │ 1. 构建上下文 (context.py)                   │  │          │
│  │  │    - Bootstrap文件 (AGENTS.md, SOUL.md...)   │  │          │
│  │  │    - 记忆系统 (memory.py)                    │  │          │
│  │  │    - 技能加载 (skills.py)                    │  │          │
│  │  └─────────────────────────────────────────────┘  │          │
│  │  ┌─────────────────────────────────────────────┐  │          │
│  │  │ 2. 调用LLM (providers/)                      │  │          │
│  │  │    - LiteLLM多提供商支持                     │  │          │
│  │  │    - 模型选择与参数配置                      │  │          │
│  │  └─────────────────────────────────────────────┘  │          │
│  │  ┌─────────────────────────────────────────────┐  │          │
│  │  │ 3. 执行工具 (agent/tools/)                   │  │          │
│  │  │    - Shell执行                               │  │          │
│  │  │    - 文件操作                                │  │          │
│  │  │    - 网络搜索/抓取                           │  │          │
│  │  │    - 消息发送                                │  │          │
│  │  │    - 子代理生成                              │  │          │
│  │  │    - 定时任务                                │  │          │
│  │  └─────────────────────────────────────────────┘  │          │
│  └─────────────────────┬─────────────────────────────┘          │
│                        │ OutboundMessage                       │
│                        ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │     Session Manager (会话管理)                     │          │
│  │  (session/manager.py)                              │          │
│  └───────────────────────────────────────────────────┘          │
│                        │                                       │
│                        ▼                                       │
│           返回到聊天平台/CLI                                   │
│                                                                 │
│  ┌───────────────────────────────────────────────────┐          │
│  │  辅助服务                                           │          │
│  │  - Cron服务 (cron/service.py) - 定时任务          │          │
│  │  - Heartbeat服务 (heartbeat/service.py) - 心跳唤醒 │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 核心模块结构

```
nanobot/
├── __main__.py              # 入口点
├── cli/commands.py          # CLI命令定义
├── agent/                   # 核心智能体
│   ├── loop.py              # 主处理循环 ⭐
│   ├── context.py           # 上下文构建器 ⭐
│   ├── memory.py            # 记忆系统
│   ├── skills.py            # 技能加载器
│   ├── subagent.py          # 子代理管理
│   └── tools/               # 工具系统
│       ├── base.py          # 工具基类 ⭐
│       ├── registry.py      # 工具注册表
│       ├── filesystem.py    # 文件操作
│       ├── shell.py         # Shell执行
│       ├── web.py           # 网络工具
│       ├── message.py       # 消息发送
│       ├── spawn.py         # 子代理生成
│       └── cron.py          # 定时任务
├── channels/                # 通信渠道
│   ├── base.py              # 渠道基类 ⭐
│   ├── manager.py           # 渠道管理器
│   ├── telegram.py          # Telegram集成
│   ├── whatsapp.py          # WhatsApp集成
│   ├── discord.py           # Discord集成
│   ├── feishu.py            # 飞书集成
│   └── dingtalk.py          # 钉钉集成
├── providers/               # LLM提供商
│   ├── base.py              # 提供商基类 ⭐
│   ├── litellm_provider.py  # LiteLLM实现
│   ├── registry.py          # 提供商注册表 ⭐
│   └── transcription.py     # 语音转文字
├── bus/                     # 事件系统
│   ├── events.py            # 事件定义 ⭐
│   └── queue.py             # 消息总线
├── config/                  # 配置管理
│   ├── loader.py            # 配置加载器
│   └── schema.py            # 配置模式
├── cron/                    # 定时任务
│   ├── service.py           # Cron服务
│   └── types.py             # Cron类型
├── session/                 # 会话管理
│   └── manager.py           # 会话管理器
├── heartbeat/               # 心跳服务
│   └── service.py
├── skills/                  # 内置技能
│   ├── github/SKILL.md
│   ├── weather/SKILL.md
│   └── ...
└── utils/                   # 工具函数
    └── helpers.py
```

---

## 二、学习路径（从零到一）

### 阶段一：基础准备（1-2天）

#### 2.1 环境搭建

```bash
# 1. 克隆项目
git clone https://github.com/HKUDS/nanobot.git
cd nanobot

# 2. 安装依赖
pip install -e .

# 3. 初始化配置
nanobot onboard

# 4. 编辑配置文件，添加API密钥
# ~/.nanobot/config.json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}

# 5. 测试运行
nanobot agent -m "Hello!"
```

#### 2.2 理解关键概念

**学习要点**：

1. **消息流**: 用户消息 → Channel → Bus → Agent → LLM → Tools → Response → Bus → Channel → 用户
2. **工具系统**: LLM通过Function Calling调用工具执行实际操作
3. **异步架构**: 所有I/O操作都是异步的（asyncio）
4. **事件驱动**: Channel和Agent通过消息总线解耦

### 阶段二：核心模块深度学习（3-5天）

#### 2.3 深入 AgentLoop（agent/loop.py）⭐

**核心文件**: [agent/loop.py](nanobot/agent/loop.py)

**学习目标**: 理解智能体如何处理消息

**关键代码分析**:

```python
class AgentLoop:
    """智能体主循环

    处理流程：
    1. 从消息总线接收消息
    2. 构建上下文（历史、记忆、技能）
    3. 调用LLM
    4. 执行工具调用
    5. 发送响应
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: ExecConfig = None,
        cron_service: CronService = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager = None,
    ):
        # 初始化组件
        self.bus = bus
        self.provider = provider
        self.context_builder = ContextBuilder(workspace)
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        self.tool_registry = ToolRegistry()
        self.session_manager = session_manager or SessionManager(workspace)

        # 注册工具
        self._register_tools(brave_api_key, exec_config, cron_service)

    async def run(self):
        """主循环：持续从消息总线消费并处理消息"""
        while self._running:
            msg = await self.bus.consume_inbound()
            response = await self._process_message(msg)
            if response:
                await self.bus.publish_outbound(response)

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """处理单条消息"""
        # 1. 加载会话历史
        session = self.session_manager.get_or_create(msg.session_key)
        history = session.get_messages()

        # 2. 构建上下文
        context = await self.context_builder.build(
            history=history,
            memory=self.memory.get_recent_memories(),
            skills=self.skills,
            media=msg.media,
        )

        # 3. 主处理循环（处理工具调用）
        for _ in range(self.max_iterations):
            # 调用LLM
            llm_response = await self.provider.chat(
                messages=context + history,
                tools=self.tool_registry.get_definitions(),
                model=self.model,
            )

            # 处理内容响应
            if llm_response.content:
                history.append({"role": "assistant", "content": llm_response.content})

            # 处理工具调用
            if not llm_response.tool_calls:
                break

            for tool_call in llm_response.tool_calls:
                result = await self.tool_registry.execute(
                    tool_call.name,
                    tool_call.arguments,
                    context={
                        "workspace": self.workspace,
                        "restrict": self.restrict_to_workspace,
                        "bus": self.bus,
                    }
                )
                history.append({
                    "role": "tool",
                    "name": tool_call.name,
                    "content": result
                })

        # 4. 保存会话
        session.set_messages(history)
        self.session_manager.save(session)

        # 5. 返回响应
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=llm_response.content,
        )
```

**实践练习**：
1. 阅读 [agent/loop.py](nanobot/agent/loop.py) 完整代码
2. 添加调试日志，追踪消息处理流程
3. 修改 `max_iterations` 参数，观察对复杂任务的影响

#### 2.4 深入 ContextBuilder（agent/context.py）⭐

**核心文件**: [agent/context.py](nanobot/agent/context.py)

**学习目标**: 理解如何为LLM构建完整的上下文

**关键代码分析**:

```python
class ContextBuilder:
    """上下文构建器

    将以下内容组装成系统提示：
    1. Bootstrap文件（AGENTS.md, SOUL.md, USER.md等）
    2. 长期记忆
    3. 技能文档
    4. 工具定义
    """

    BOOTSTRAP_FILES = [
        "AGENTS.md",   # 智能体行为指南
        "SOUL.md",     # 个性特征
        "USER.md",     # 用户信息
        "TOOLS.md",    # 工具使用说明
        "IDENTITY.md", # 身份定义
    ]

    async def build(
        self,
        history: list[dict],
        memory: str,
        skills: SkillsLoader,
        media: list[Media] | None = None,
    ) -> list[dict]:
        """构建完整上下文"""

        # 1. 系统提示
        system_parts = []

        # 加载Bootstrap文件
        for filename in self.BOOTSTRAP_FILES:
            content = self._load_bootstrap_file(filename)
            if content:
                system_parts.append(content)

        # 添加记忆
        if memory:
            system_parts.append(f"## Memory\n{memory}")

        # 添加技能（渐进式加载）
        always_loaded = skills.get_always_loaded_skills()
        available_skills = skills.get_available_skills_summary()

        if always_loaded:
            system_parts.append("## Available Skills\n" + "\n\n".join(always_loaded))
        if available_skills:
            system_parts.append(f"## Additional Skills\n{available_skills}")

        system_prompt = "\n\n".join(system_parts)

        # 2. 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史消息
        for msg in history:
            messages.append(msg)

        # 处理媒体（图片等）
        if media:
            for item in media:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "[Image attached]"},
                        {"type": "image_url", "image_url": {"url": item.base64_data}}
                    ]
                })

        return messages
```

**实践练习**：
1. 修改 Bootstrap 文件内容，观察智能体行为变化
2. 添加自定义 Bootstrap 文件（如 `PROMPTS.md`）
3. 理解"渐进式技能加载"的设计思路

#### 2.5 深入工具系统（agent/tools/）⭐

**核心文件**:
- [agent/tools/base.py](nanobot/agent/tools/base.py) - 工具基类
- [agent/tools/registry.py](nanobot/agent/tools/registry.py) - 工具注册表

**学习目标**: 理解工具系统如何工作

**工具基类分析**:

```python
class Tool(ABC):
    """所有工具的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（用于函数调用）"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（告诉LLM这个工具做什么）"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """参数的JSON Schema定义"""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具"""
        pass

    def to_openai_schema(self) -> dict:
        """转换为OpenAI Function Calling格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

**示例：Shell工具**

```python
class ExecTool(Tool):
    """执行Shell命令的工具"""

    name = "exec"
    description = "Execute a shell command"

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute"
            },
            "cwd": {
                "type": "string",
                "description": "Working directory"
            }
        },
        "required": ["command"]
    }

    def __init__(
        self,
        deny_patterns: list[str] = None,
        restrict_to: Path | None = None,
        timeout: int = 30,
    ):
        self.deny_patterns = deny_patterns or []
        self.restrict_to = restrict_to
        self.timeout = timeout

    async def execute(self, command: str, cwd: str | None = None) -> str:
        # 1. 安全检查
        for pattern in self.deny_patterns:
            if pattern in command:
                return f"Error: Command blocked by deny pattern '{pattern}'"

        # 2. 路径限制检查
        if self.restrict_to and cwd:
            cwd_path = Path(cwd).resolve()
            if not str(cwd_path).startswith(str(self.restrict_to)):
                return "Error: Path outside workspace"

        # 3. 执行命令
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd or os.getcwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            return stdout.decode() + stderr.decode()
        except asyncio.TimeoutError:
            process.kill()
            return f"Error: Command timed out after {self.timeout}s"
```

**实践练习**：
1. 阅读所有内置工具的实现
2. 创建一个自定义工具（如天气查询、数据库操作等）
3. 理解工具的安全限制机制

### 阶段三：通信渠道学习（2-3天）

#### 2.6 深入 Channel 基类（channels/base.py）⭐

**核心文件**: [channels/base.py](nanobot/channels/base.py)

**学习目标**: 理解如何添加新的通信渠道

**关键代码分析**:

```python
class BaseChannel(ABC):
    """所有通信渠道的抽象基类"""

    def __init__(
        self,
        bus: MessageBus,
        allowed_senders: list[str] | None = None,
        session_manager: SessionManager = None,
    ):
        self.bus = bus
        self.allowed_senders = allowed_senders or []
        self.session_manager = session_manager
        self._running = False

    @abstractmethod
    async def start(self):
        """启动渠道（连接、监听）"""
        pass

    @abstractmethod
    async def stop(self):
        """停止渠道（清理资源）"""
        pass

    @abstractmethod
    async def send(self, chat_id: str, content: str, **kwargs):
        """发送消息"""
        pass

    def is_allowed(self, sender_id: str) -> bool:
        """检查发送者是否在白名单中"""
        if not self.allowed_senders:
            return True  # 空白名单 = 允许所有人
        return sender_id in self.allowed_senders

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[Media] | None = None,
    ):
        """处理接收到的消息"""
        if not self.is_allowed(sender_id):
            return

        msg = InboundMessage(
            channel=self.__class__.__name__.lower().replace("channel", ""),
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={},
        )

        await self.bus.publish_inbound(msg)
```

**Telegram 示例**:

```python
import telegram
from telegram.ext import Application

class TelegramChannel(BaseChannel):
    """Telegram 集成"""

    def __init__(self, token: str, **kwargs):
        super().__init__(**kwargs)
        self.token = token
        self.app = None

    async def start(self):
        """启动Telegram bot"""
        self.app = Application.builder().token(self.token).build()

        # 注册消息处理器
        self.app.add_handler(telegram.MessageHandler(
            telegram.filters.TEXT & ~telegram.filters.COMMAND,
            self._on_message
        ))

        await self.app.initialize()
        await self.app.start()
        self._running = True

    async def _on_message(self, update: telegram.Update, context):
        """处理接收到的消息"""
        sender_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)
        content = update.effective_message.text

        await self._handle_message(sender_id, chat_id, content)

    async def send(self, chat_id: str, content: str, **kwargs):
        """发送消息到Telegram"""
        await self.app.bot.sendMessage(chat_id=chat_id, text=content)

    async def stop(self):
        """停止Telegram bot"""
        self._running = False
        if self.app:
            await self.app.stop()
            await self.app.shutdown()
```

**实践练习**：
1. 阅读所有渠道实现
2. 添加一个新的渠道（如 Slack、微信等）
3. 理解渠道的权限控制机制

### 阶段四：LLM 提供商系统（1-2天）

#### 2.7 深入提供商系统（providers/）⭐

**核心文件**:
- [providers/base.py](nanobot/providers/base.py)
- [providers/registry.py](nanobot/providers/registry.py)

**学习目标**: 理解如何添加新的LLM提供商

**提供商基类**:

```python
class LLMProvider(ABC):
    """LLM提供商的抽象接口"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """发送聊天完成请求"""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """获取默认模型"""
        pass

@dataclass
class LLMResponse:
    """LLM响应"""
    content: str | None
    tool_calls: list[ToolCall] | None
    usage: dict | None
    reasoning_content: str | None = None  # 用于推理模型
```

**提供商注册表**:

```python
@dataclass(frozen=True)
class ProviderSpec:
    """单个LLM提供商的元数据"""

    # 身份
    name: str                    # 配置字段名
    keywords: tuple[str, ...]    # 模型名称关键字（用于自动匹配）
    env_key: str                 # LiteLLM的环境变量名
    display_name: str = ""       # 显示名称

    # 模型前缀处理
    litellm_prefix: str = ""           # 自动添加前缀
    skip_prefixes: tuple[str, ...] = ()  # 跳过这些前缀

    # 额外环境变量
    env_extras: tuple[tuple[str, str], ...] = ()

    # 网关/本地检测
    is_gateway: bool = False
    is_local: bool = False
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""
    default_api_base: str = ""

# 提供商注册表
PROVIDERS = (
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        env_key="OPENROUTER_API_KEY",
        display_name="OpenRouter",
        litellm_prefix="openrouter/",
        skip_prefixes=("openrouter/",),
        is_gateway=True,
        detect_by_key_prefix="sk-or-",
    ),
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        litellm_prefix="anthropic/",
        skip_prefixes=("anthropic/", "claude-"),
    ),
    # ... 更多提供商
)
```

**添加新提供商只需2步**：

**步骤1**: 在 [providers/registry.py](nanobot/providers/registry.py) 添加 `ProviderSpec`：

```python
PROVIDERS = (
    # ... 现有提供商
    ProviderSpec(
        name="myprovider",
        keywords=("myprovider", "mymodel"),
        env_key="MYPROVIDER_API_KEY",
        display_name="My Provider",
        litellm_prefix="myprovider/",
        skip_prefixes=("myprovider/",),
    ),
)
```

**步骤2**: 在 [config/schema.py](nanobot/config/schema.py) 添加配置字段：

```python
class ProvidersConfig(BaseModel):
    # ... 现有字段
    myprovider: ProviderConfig = ProviderConfig()
```

**实践练习**：
1. 添加一个新的LLM提供商（如 Zhipu、Moonshot 等）
2. 理解网关提供商的特殊处理
3. 测试不同模型的兼容性

### 阶段五：实战项目（3-5天）

#### 2.8 从零构建自己的 Agent

**项目目标**: 基于对 nanobot 的理解，构建一个简化版的 AI Agent

**实现内容**：

```python
# my_agent/ 目录结构
my_agent/
├── __main__.py
├── agent/
│   ├── __init__.py
│   ├── loop.py          # 主循环
│   ├── context.py       # 上下文构建
│   └── tools/
│       ├── __init__.py
│       ├── base.py      # 工具基类
│       ├── registry.py  # 工具注册
│       ├── calculator.py  # 计算器工具
│       └── weather.py     # 天气工具
├── providers/
│   ├── __init__.py
│   └── openai.py       # OpenAI提供商
└── config/
    ├── __init__.py
    └── settings.py     # 配置管理
```

**简化版实现步骤**：

1. **实现工具基类** (tools/base.py)
2. **实现OpenAI提供商** (providers/openai.py)
3. **实现上下文构建器** (agent/context.py)
4. **实现主循环** (agent/loop.py)
5. **添加CLI命令** (__main__.py)
6. **测试运行**

---

## 三、二次开发指南

### 3.1 添加新技能（Skill）

**技能目录结构**:

```
workspace/skills/my-skill/
├── SKILL.md              # 技能文档（必需）
├── requirements.txt      # Python依赖（可选）
└── scripts/              # 辅助脚本（可选）
    └── helper.sh
```

**SKILL.md 模板**:

```markdown
---
name: "my-skill"
description: "技能描述"
alwaysLoad: false
bins:
  - /usr/bin/my-tool
env:
  - MY_API_KEY
---

# 我的技能

## 功能说明

这个技能可以做什么...

## 使用方法

### 工具1: tool_name

\`\`\`python
await tool_name(param1="value1")
\`\`\`

参数说明：
- param1: 参数1说明
- param2: 参数2说明

### 工具2: another_tool

...

## 示例

用户: "请执行某个操作"
Agent: "我来帮你..."

## 注意事项

- 注意事项1
- 注意事项2
```

**内置技能参考**：
- [skills/github/SKILL.md](nanobot/skills/github/SKILL.md)
- [skills/weather/SKILL.md](nanobot/skills/weather/SKILL.md)

### 3.2 添加新工具（Tool）

**步骤1**: 创建工具文件

```python
# agent/tools/mytool.py

from .base import Tool

class MyCustomTool(Tool):
    """自定义工具"""

    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return "A brief description of what this tool does"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of param1"
                },
                "param2": {
                    "type": "integer",
                    "description": "Description of param2"
                }
            },
            "required": ["param1"]
        }

    async def execute(self, **kwargs) -> str:
        param1 = kwargs.get("param1")
        param2 = kwargs.get("param2", 0)

        # 实现你的逻辑
        result = f"Processed {param1} with value {param2}"

        return result
```

**步骤2**: 在 AgentLoop 中注册

```python
# agent/loop.py

from nanobot.agent.tools.mytool import MyCustomTool

class AgentLoop:
    def _register_tools(self, ...):
        # 现有工具
        self.tool_registry.register(ExecTool(...))
        self.tool_registry.register(ReadFileTool(...))

        # 添加自定义工具
        self.tool_registry.register(MyCustomTool())
```

### 3.3 添加新渠道（Channel）

**步骤1**: 创建渠道类

```python
# channels/mychannel.py

from .base import BaseChannel
from nanobot.bus.events import InboundMessage, Media

class MyChannel(BaseChannel):
    """自定义通信渠道"""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.client = None

    async def start(self):
        """启动渠道"""
        # 初始化客户端
        self.client = MyPlatformClient(self.api_key)

        # 注册消息处理器
        self.client.on_message(self._on_message)

        await self.client.connect()
        self._running = True

    async def _on_message(self, message):
        """处理接收到的消息"""
        await self._handle_message(
            sender_id=message.sender_id,
            chat_id=message.chat_id,
            content=message.text,
            media=None,  # 或处理附件
        )

    async def send(self, chat_id: str, content: str, **kwargs):
        """发送消息"""
        await self.client.send_message(chat_id, content)

    async def stop(self):
        """停止渠道"""
        self._running = False
        if self.client:
            await self.client.disconnect()
```

**步骤2**: 在 [config/schema.py](nanobot/config/schema.py) 添加配置

```python
class MyChannelConfig(BaseModel):
    enabled: bool = False
    apiKey: str = ""
    allowFrom: list[str] = []

class ChannelsConfig(BaseModel):
    # ... 现有渠道
    mychannel: MyChannelConfig = MyChannelConfig()
```

**步骤3**: 在 [channels/manager.py](nanobot/channels/manager.py) 添加初始化

```python
class ChannelManager:
    async def _initialize_channels(self):
        # ... 现有渠道

        if self.config.channels.mychannel.enabled:
            from nanobot.channels.mychannel import MyChannel
            channel = MyChannel(
                api_key=self.config.channels.mychannel.apiKey,
                allowed_senders=self.config.channels.mychannel.allowFrom,
                session_manager=self.session_manager,
            )
            self._channels["mychannel"] = channel
```

### 3.4 添加新LLM提供商

参考前面的 [2.7 深入提供商系统](#27-深入提供商系统providers-)

只需2步即可添加新提供商。

### 3.5 修改智能体行为

**方法1**: 修改 Bootstrap 文件

```bash
# 编辑 workspace/AGENTS.md
vim workspace/AGENTS.md
```

**方法2**: 添加自定义 Bootstrap 文件

修改 [agent/context.py](nanobot/agent/context.py):

```python
class ContextBuilder:
    BOOTSTRAP_FILES = [
        "AGENTS.md",
        "SOUL.md",
        "USER.md",
        "TOOLS.md",
        "IDENTITY.md",
        "CUSTOM.md",  # 添加自定义文件
    ]
```

**方法3**: 修改系统提示模板

修改 [agent/context.py](nanobot/agent/context.py) 的 `build()` 方法。

---

## 四、关键设计模式总结

### 4.1 使用的模式

| 模式 | 位置 | 说明 |
|-----|------|------|
| **抽象基类 (ABC)** | Tool, LLMProvider, BaseChannel | 定义接口契约 |
| **注册表模式** | ToolRegistry, ProviderRegistry | 动态注册和查找 |
| **事件驱动** | MessageBus | 解耦组件 |
| **构建器模式** | ContextBuilder | 复杂对象构建 |
| **策略模式** | SkillsLoader | 可插拔的技能加载策略 |
| **单例模式** | SessionManager | 会话状态管理 |

### 4.2 异步编程模式

```python
# 异步生成器
async def process_stream():
    async for chunk in provider.chat_stream(messages):
        yield chunk

# 异步上下文管理器
async with async_session() as session:
    await session.execute(query)

# 任务组
async with asyncio.TaskGroup() as tg:
    tg.create_task(channel1.start())
    tg.create_task(channel2.start())
```

---

## 五、学习检查清单

### 基础知识
- [ ] Python 3.11+ 异步编程 (asyncio)
- [ ] 类型提示 (typing)
- [ ] Pydantic 数据验证
- [ ] Typer CLI 框架

### nanobot 核心
- [ ] 理解消息流（用户→Channel→Bus→Agent→LLM→Tools→Response）
- [ ] 理解 AgentLoop 处理流程
- [ ] 理解 ContextBuilder 如何构建上下文
- [ ] 理解工具系统的设计和注册机制
- [ ] 理解提供商系统的自动检测逻辑
- [ ] 理解消息总线的事件驱动架构

### 二次开发能力
- [ ] 能够创建自定义工具
- [ ] 能够创建自定义技能
- [ ] 能够添加新的通信渠道
- [ ] 能够添加新的LLM提供商
- [ ] 能够修改智能体行为

### 实战项目
- [ ] 从零构建一个简化版Agent
- [ ] 为 nanobot 贡献一个新功能

---

## 六、推荐学习顺序

```
第1周: 基础准备
├── Day 1-2: 环境搭建 + 运行示例
└── Day 3-7: 理解核心概念 + 阅读代码

第2周: 深度学习
├── Day 1-2: AgentLoop + ContextBuilder
├── Day 3-4: 工具系统
└── Day 5-7: 通信渠道 + 提供商系统

第3周: 实战
├── Day 1-3: 添加自定义功能
└── Day 4-7: 从零构建简化版Agent

第4周: 进阶
├── Day 1-3: 二次开发项目
└── Day 4-7: 贡献到上游
```

---

## 七、常见问题

### Q1: 如何调试 Agent 的执行流程？

在 [agent/loop.py](nanobot/agent/loop.py) 添加日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

class AgentLoop:
    async def _process_message(self, msg: InboundMessage):
        logging.info(f"Processing message: {msg.content}")
        # ...
```

### Q2: 如何添加状态持久化？

使用 SessionManager 的 `save()` 方法：

```python
self.session_manager.save(session)
```

### Q3: 如何限制工具的执行权限？

在工具的 `execute()` 方法中添加权限检查：

```python
async def execute(self, **kwargs):
    if not self._check_permission():
        return "Permission denied"
    # ...
```

### Q4: 如何支持流式响应？

修改 LLMProvider 接口，添加 `chat_stream()` 方法。

---

## 八、参考资源

- **项目**: https://github.com/HKUDS/nanobot
- **LiteLLM 文档**: https://docs.litellm.ai/
- **Python asyncio**: https://docs.python.org/3/library/asyncio.html
- **Function Calling**: https://platform.openai.com/docs/guides/function-calling

---

**学习愉快！如果有任何问题，欢迎提出 PR 或 Issue！**
