# Whisper Speech Recognition MCP Server
---
[中文文档](README-CN.md)
---
A high-performance speech recognition MCP server based on Faster Whisper, providing efficient audio transcription capabilities.

## Features

- Integrated with Faster Whisper for efficient speech recognition
- Batch processing acceleration for improved transcription speed
- Automatic CUDA acceleration (if available)
- Support for multiple model sizes (tiny to large-v3)
- Output formats include VTT subtitles, SRT, and JSON
- Support for batch transcription of audio files in a folder
- Model instance caching to avoid repeated loading
- Dynamic batch size adjustment based on GPU memory

## Installation

### Dependencies

- Python 3.10+
- faster-whisper>=0.9.0
- torch==2.6.0+cu126
- torchaudio==2.6.0+cu126
- mcp[cli]>=1.2.0

### Installation Steps

1. Clone or download this repository
2. Create and activate a virtual environment (recommended)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### PyTorch Installation Guide

Install the appropriate version of PyTorch based on your CUDA version:

- CUDA 12.6:
  ```bash
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
  ```

- CUDA 12.1:
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  ```

- CPU version:
  ```bash
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  ```

You can check your CUDA version with `nvcc --version` or `nvidia-smi`.

## Usage

### Starting the Server

On Windows, simply run `start_server.bat`.

On other platforms, run:

```bash
python whisper_server.py
```

### Configuring Claude Desktop

1. Open the Claude Desktop configuration file:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add the Whisper server configuration:

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

3. Restart Claude Desktop

### Available Tools

The server provides the following tools:

1. **get_model_info** - Get information about available Whisper models
2. **transcribe** - Transcribe a single audio file
3. **batch_transcribe** - Batch transcribe audio files in a folder

## Performance Optimization Tips

- Using CUDA acceleration significantly improves transcription speed
- Batch processing mode is more efficient for large numbers of short audio files
- Batch size is automatically adjusted based on GPU memory size
- Using VAD (Voice Activity Detection) filtering improves accuracy for long audio
- Specifying the correct language can improve transcription quality

## Local Testing Methods

1. Use MCP Inspector for quick testing:

```bash
mcp dev whisper_server.py
```

2. Use Claude Desktop for integration testing

3. Use command line direct invocation (requires mcp[cli]):

```bash
mcp run whisper_server.py
```

## Error Handling

The server implements the following error handling mechanisms:

- Audio file existence check
- Model loading failure handling
- Transcription process exception catching
- GPU memory management
- Batch processing parameter adaptive adjustment

## Project Structure

- `whisper_server.py`: Main server code
- `model_manager.py`: Whisper model loading and caching
- `audio_processor.py`: Audio file validation and preprocessing
- `formatters.py`: Output formatting (VTT, SRT, JSON)
- `transcriber.py`: Core transcription logic
- `start_server.bat`: Windows startup script

## License

MIT

## Acknowledgements

This project was developed with the assistance of these amazing AI tools and models:

- [GitHub Copilot](https://github.com/features/copilot) - AI pair programmer
- [Trae](https://trae.ai/) - Agentic AI coding assistant
- [Cline](https://cline.ai/) - AI-powered terminal
- [DeepSeek](https://www.deepseek.com/) - Advanced AI model
- [Claude-3.7-Sonnet](https://www.anthropic.com/claude) - Anthropic's powerful AI assistant
- [Gemini-2.0-Flash](https://ai.google/gemini/) - Google's multimodal AI model
- [VS Code](https://code.visualstudio.com/) - Powerful code editor
- [Whisper](https://github.com/openai/whisper) - OpenAI's speech recognition model
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation

Special thanks to these incredible tools and the teams behind them.

