#!/usr/bin/env python3
"""
音频处理模块
负责音频文件的验证和预处理
"""

import os
import logging
from typing import Union, Any
from faster_whisper import decode_audio

# 日志配置
logger = logging.getLogger(__name__)

def validate_audio_file(audio_path: str) -> str:
    """
    验证音频文件是否有效

    Args:
        audio_path: 音频文件路径

    Returns:
        str: 验证结果，"ok"表示验证通过，否则返回错误信息
    """
    # 验证参数
    if not os.path.exists(audio_path):
        return f"错误: 音频文件不存在: {audio_path}"

    # 验证文件格式
    supported_formats = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in supported_formats:
        return f"错误: 不支持的音频格式: {file_ext}。支持的格式: {', '.join(supported_formats)}"

    # 验证文件大小
    try:
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return f"错误: 音频文件为空: {audio_path}"

        # 大文件警告（超过1GB）
        if file_size > 1024 * 1024 * 1024:
            logger.warning(f"警告: 文件大小超过1GB，可能需要较长处理时间: {audio_path}")
    except Exception as e:
        logger.error(f"检查文件大小失败: {str(e)}")
        return f"错误: 检查文件大小失败: {str(e)}"

    return "ok"

def process_audio(audio_path: str) -> Union[str, Any]:
    """
    处理音频文件，进行解码和预处理

    Args:
        audio_path: 音频文件路径

    Returns:
        Union[str, Any]: 处理后的音频数据或原始文件路径
    """
    # 尝试使用decode_audio预处理音频，以处理更多格式
    try:
        audio_data = decode_audio(audio_path)
        logger.info(f"成功预处理音频: {os.path.basename(audio_path)}")
        return audio_data
    except Exception as audio_error:
        logger.warning(f"音频预处理失败，将直接使用文件路径: {str(audio_error)}")
        return audio_path