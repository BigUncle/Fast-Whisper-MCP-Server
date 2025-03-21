#!/usr/bin/env python3
"""
基于Faster Whisper的语音识别MCP服务
提供高性能的音频转录功能，支持批处理加速和多种输出格式
"""

import os
import logging
from mcp.server.fastmcp import FastMCP

from model_manager import get_model_info
from transcriber import transcribe_audio, batch_transcribe

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastMCP服务器实例
mcp = FastMCP(
    name="fast-whisper-mcp-server",
    version="0.1.1",
    dependencies=["faster-whisper>=0.9.0", "torch==2.6.0+cu126", "torchaudio==2.6.0+cu126", "numpy>=1.20.0"]
)

@mcp.tool()
def get_model_info_api() -> str:
    """
    获取可用的Whisper模型信息
    """
    return get_model_info()

@mcp.tool()
def transcribe(audio_path: str, model_name: str = "large-v3", device: str = "auto",
              compute_type: str = "auto", language: str = None, output_format: str = "vtt",
              beam_size: int = 5, temperature: float = 0.0, initial_prompt: str = None,
              output_directory: str = None) -> str:
    """
    使用Faster Whisper转录音频文件

    Args:
        audio_path: 音频文件路径
        model_name: 模型名称 (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: 运行设备 (cpu, cuda, auto)
        compute_type: 计算类型 (float16, int8, auto)
        language: 语言代码 (如zh, en, ja等，默认自动检测)
        output_format: 输出格式 (vtt, srt或json)
        beam_size: 波束搜索大小，较大的值可能提高准确性但会降低速度
        temperature: 采样温度，贪婪解码
        initial_prompt: 初始提示文本，可以帮助模型更好地理解上下文
        output_directory: 输出目录路径，默认为音频文件所在目录

    Returns:
        str: 转录结果，格式为VTT字幕或JSON
    """
    return transcribe_audio(
        audio_path=audio_path,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        output_format=output_format,
        beam_size=beam_size,
        temperature=temperature,
        initial_prompt=initial_prompt,
        output_directory=output_directory
    )

@mcp.tool()
def batch_transcribe_audio(audio_folder: str, output_folder: str = None, model_name: str = "large-v3",
                    device: str = "auto", compute_type: str = "auto", language: str = None,
                    output_format: str = "vtt", beam_size: int = 5, temperature: float = 0.0,
                    initial_prompt: str = None, parallel_files: int = 1) -> str:
    """
    批量转录文件夹中的音频文件

    Args:
        audio_folder: 包含音频文件的文件夹路径
        output_folder: 输出文件夹路径，默认为audio_folder下的transcript子文件夹
        model_name: 模型名称 (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: 运行设备 (cpu, cuda, auto)
        compute_type: 计算类型 (float16, int8, auto)
        language: 语言代码 (如zh, en, ja等，默认自动检测)
        output_format: 输出格式 (vtt, srt或json)
        beam_size: 波束搜索大小，较大的值可能提高准确性但会降低速度
        temperature: 采样温度，0表示贪婪解码
        initial_prompt: 初始提示文本，可以帮助模型更好地理解上下文
        parallel_files: 并行处理的文件数量（仅在CPU模式下有效）

    Returns:
        str: 批处理结果摘要，包含处理时间和成功率
    """
    return batch_transcribe(
        audio_folder=audio_folder,
        output_folder=output_folder,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        output_format=output_format,
        beam_size=beam_size,
        temperature=temperature,
        initial_prompt=initial_prompt,
        parallel_files=parallel_files
    )

if __name__ == "__main__":
    # 运行服务器
    mcp.run()