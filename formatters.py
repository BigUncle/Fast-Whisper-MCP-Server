#!/usr/bin/env python3
"""
格式化输出模块
负责将转录结果格式化为不同的输出格式（VTT、SRT、JSON）
"""

import json
from typing import List, Dict, Any

def format_vtt(segments: List) -> str:
    """
    将转录结果格式化为VTT

    Args:
        segments: 转录段落列表

    Returns:
        str: VTT格式的字幕内容
    """
    vtt_content = "WEBVTT\n\n"

    for segment in segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()

        if text:
            vtt_content += f"{start} --> {end}\n{text}\n\n"

    return vtt_content

def format_srt(segments: List) -> str:
    """
    将转录结果格式化为SRT

    Args:
        segments: 转录段落列表

    Returns:
        str: SRT格式的字幕内容
    """
    srt_content = ""
    index = 1

    for segment in segments:
        start = format_timestamp_srt(segment.start)
        end = format_timestamp_srt(segment.end)
        text = segment.text.strip()

        if text:
            srt_content += f"{index}\n{start} --> {end}\n{text}\n\n"
            index += 1

    return srt_content

def format_json(segments: List, info: Any) -> str:
    """
    将转录结果格式化为JSON

    Args:
        segments: 转录段落列表
        info: 转录信息对象

    Returns:
        str: JSON格式的转录结果
    """
    result = {
        "segments": [{
            "id": i,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [{
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "probability": word.probability
            } for word in segment.words] if hasattr(segment, 'words') and segment.words else []
        } for i, segment in enumerate(segments)],
        "language": info.language,
        "language_probability": info.language_probability if hasattr(info, 'language_probability') else None,
        "duration": info.duration,
        "all_language_probs": info.all_language_probs if hasattr(info, 'all_language_probs') else None
    }
    return json.dumps(result, indent=2, ensure_ascii=False)

def format_timestamp(seconds: float) -> str:
    """
    格式化时间戳为VTT格式

    Args:
        seconds: 秒数

    Returns:
        str: 格式化的时间戳 (HH:MM:SS.mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def format_timestamp_srt(seconds: float) -> str:
    """
    格式化时间戳为SRT格式

    Args:
        seconds: 秒数

    Returns:
        str: 格式化的时间戳 (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

def format_time(seconds: float) -> str:
    """
    格式化时间为可读格式

    Args:
        seconds: 秒数

    Returns:
        str: 格式化的时间 (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"