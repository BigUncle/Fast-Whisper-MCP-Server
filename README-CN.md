# Whisper 语音识别 MCP 服务器

基于 Faster Whisper 的语音识别 MCP 服务器，提供高性能的音频转录功能。

## 功能特点

- 集成 Faster Whisper 进行高效语音识别
- 支持批处理加速，提高转录速度
- 自动使用 CUDA 加速（如果可用）
- 支持多种模型大小（tiny 到 large-v3）
- 输出格式支持 VTT 字幕和 JSON
- 支持批量转录文件夹中的音频文件
- 模型实例缓存，避免重复加载

## 安装

### 依赖项

- Python 3.10+
- faster-whisper>=0.9.0
- torch==2.6.0+cu126
- torchaudio==2.6.0+cu126
- mcp[cli]>=1.2.0

### 安装步骤

1. 克隆或下载此仓库
2. 创建并激活虚拟环境（推荐）
3. 安装依赖项：

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动服务器

在 Windows 上，直接运行 `start_server.bat`。

在其他平台上，运行：

```bash
python whisper_server.py
```

### 配置 Claude Desktop

1. 打开 Claude Desktop 配置文件：
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. 添加 Whisper 服务器配置：

```json
{
  "mcpServers": {
    "whisper": {
      "command": "python",
      "args": ["D:/path/to/whisper_server.py"],
      "env": {}
    }
  }
}
```

3. 重启 Claude Desktop

### 可用工具

服务器提供以下工具：

1. **get_model_info** - 获取可用的 Whisper 模型信息
2. **transcribe** - 转录单个音频文件
3. **batch_transcribe** - 批量转录文件夹中的音频文件

## 性能优化建议

- 使用 CUDA 加速可显著提高转录速度
- 对于大量短音频，批处理模式效率更高
- 根据 GPU 显存大小自动调整批处理大小
- 对于长音频，使用 VAD 过滤可提高准确性
- 指定正确的语言可提高转录质量

## 本地测试方案

1. 使用 MCP Inspector 进行快速测试：

```bash
mcp dev whisper_server.py
```

2. 使用 Claude Desktop 进行集成测试

3. 使用命令行直接调用（需要安装 mcp[cli]）：

```bash
mcp run whisper_server.py
```

## 错误处理

服务器实现了以下错误处理机制：

- 音频文件不存在检查
- 模型加载失败处理
- 转录过程异常捕获
- GPU 内存管理
- 批处理参数自适应调整

## 许可证

MIT


# Whisper 语音识别 MCP 服务器（cline claude sonnet 3.7 完成所有任务后的说明）

我已经成功创建了基于Python的语音识别MCP服务器，该服务器集成了Faster Whisper并支持批处理加速。服务器具有以下特点：

## 核心功能

- **集成Faster Whisper**：使用最新版本(>=0.9.0)进行高效语音识别
- **CUDA加速**：自动检测并使用CUDA加速，默认使用torch==2.6.0+cu126
- **批处理优化**：使用BatchedInferencePipeline实现批处理，根据GPU显存自动调整批大小
- **模型缓存**：实现模型实例缓存，避免重复加载模型
- **多格式输出**：支持VTT字幕和JSON格式的转录结果
- **批量处理**：支持批量转录整个文件夹的音频文件

## 主要工具

服务器提供了三个主要工具：

1. **get_model_info**：获取可用的Whisper模型信息和系统配置
2. **transcribe**：转录单个音频文件，支持多种参数配置
3. **batch_transcribe**：批量转录文件夹中的音频文件

## 错误处理机制

- 音频文件存在性验证
- 模型加载异常捕获和日志记录
- 转录过程异常处理
- GPU内存管理和清理
- 批处理参数自适应调整

## 性能优化

- 根据GPU显存大小动态调整批处理大小(4-32)
- 使用VAD(语音活动检测)过滤提高准确性
- 模型实例缓存避免重复加载
- 自动选择最佳设备和计算类型

## 本地测试方案

提供了多种测试方法：

- 使用MCP Inspector进行快速测试：`mcp dev whisper_server.py`
- 使用Claude Desktop进行集成测试
- 使用命令行直接调用：`mcp run whisper_server.py`

所有文件已准备就绪，包括：

- whisper_server.py：主服务器代码
- requirements.txt：依赖项列表
- start_server.bat：Windows启动脚本
- README.md：详细文档

您可以通过运行start_server.bat或直接执行`python whisper_server.py`来启动服务器。