#!/usr/bin/env python3
"""
转录核心模块
包含音频转录的核心逻辑
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, List, Optional, Union

from model_manager import get_whisper_model
from audio_processor import validate_audio_file, process_audio
from formatters import format_vtt, format_srt, format_json, format_time

# 日志配置
logger = logging.getLogger(__name__)

def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = None,
    output_format: str = "vtt",
    beam_size: int = 5,
    temperature: float = 0.0,
    initial_prompt: str = None,
    output_directory: str = None
) -> str:
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
    # 验证音频文件
    validation_result = validate_audio_file(audio_path)
    if validation_result != "ok":
        return validation_result

    try:
        # 获取模型实例
        model_instance = get_whisper_model(model_name, device, compute_type)

        # 验证语言代码
        supported_languages = {
            "zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语", "de": "德语",
            "fr": "法语", "es": "西班牙语", "ru": "俄语", "it": "意大利语",
            "pt": "葡萄牙语", "nl": "荷兰语", "ar": "阿拉伯语", "hi": "印地语",
            "tr": "土耳其语", "vi": "越南语", "th": "泰语", "id": "印尼语"
        }

        if language is not None and language not in supported_languages:
            logger.warning(f"未知的语言代码: {language}，将使用自动检测")
            language = None

        # 设置转录参数
        options = {
            "language": language,
            "vad_filter": True,  # 使用语音活动检测
            "vad_parameters": {"min_silence_duration_ms": 500},  # VAD参数优化
            "beam_size": beam_size,
            "temperature": temperature,
            "initial_prompt": initial_prompt,
            "word_timestamps": True,  # 启用单词级时间戳
            "suppress_tokens": [-1],  # 抑制特殊标记
            "condition_on_previous_text": True,  # 基于前文进行条件生成
            "compression_ratio_threshold": 2.4  # 压缩比阈值，用于过滤重复内容
        }

        start_time = time.time()
        logger.info(f"开始转录文件: {os.path.basename(audio_path)}")

        # 处理音频
        audio_source = process_audio(audio_path)

        # 执行转录 - 优先使用批处理模型
        if model_instance['batched_model'] is not None and model_instance['device'] == 'cuda':
            logger.info("使用批处理加速进行转录...")
            # 批处理模型需要单独设置batch_size参数
            segments, info = model_instance['batched_model'].transcribe(
                audio_source,
                batch_size=model_instance['batch_size'],
                **options
            )
        else:
            logger.info("使用标准模型进行转录...")
            segments, info = model_instance['model'].transcribe(audio_source, **options)

        # 将生成器转换为列表
        segment_list = list(segments)

        if not segment_list:
            return "转录失败，未获得结果"

        # 记录转录信息
        elapsed_time = time.time() - start_time
        logger.info(f"转录完成，用时: {elapsed_time:.2f}秒，检测语言: {info.language}，音频长度: {info.duration:.2f}秒")

        # 格式化转录结果
        if output_format.lower() == "vtt":
            transcription_result = format_vtt(segment_list)
        elif output_format.lower() == "srt":
            transcription_result = format_srt(segment_list)
        else:
            transcription_result = format_json(segment_list, info)

        # 获取音频文件的目录和文件名
        audio_dir = os.path.dirname(audio_path)
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]

        # 设置输出目录
        if output_directory is None:
            output_dir = audio_dir
        else:
            output_dir = output_directory
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

        # 生成带有时间戳的文件名
        timestamp = time.strftime("%Y%m%d%H%M%S")
        output_filename = f"{audio_filename}_{timestamp}.{output_format.lower()}"
        output_path = os.path.join(output_dir, output_filename)

        # 将转录结果写入文件
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription_result)
            logger.info(f"转录结果已保存到: {output_path}")
            return f"转录成功，结果已保存到: {output_path}"
        except Exception as e:
            logger.error(f"保存转录结果失败: {str(e)}")
            return f"转录成功，但保存结果失败: {str(e)}"

    except Exception as e:
        logger.error(f"转录失败: {str(e)}")
        return f"转录过程中发生错误: {str(e)}"


def report_progress(current: int, total: int, elapsed_time: float) -> str:
    """
    生成进度报告

    Args:
        current: 当前处理的项目数
        total: 总项目数
        elapsed_time: 已用时间（秒）

    Returns:
        str: 格式化的进度报告
    """
    progress = current / total * 100
    eta = (elapsed_time / current) * (total - current) if current > 0 else 0
    return (f"进度: {current}/{total} ({progress:.1f}%)" +
            f" | 已用时间: {format_time(elapsed_time)}" +
            f" | 预计剩余: {format_time(eta)}")

def batch_transcribe(
    audio_folder: str,
    output_folder: str = None,
    model_name: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = None,
    output_format: str = "vtt",
    beam_size: int = 5,
    temperature: float = 0.0,
    initial_prompt: str = None,
    parallel_files: int = 1
) -> str:
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
    if not os.path.isdir(audio_folder):
        return f"错误: 文件夹不存在: {audio_folder}"

    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(audio_folder, "transcript")

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 验证输出格式
    valid_formats = ["vtt", "srt", "json"]
    if output_format.lower() not in valid_formats:
        return f"错误: 不支持的输出格式: {output_format}。支持的格式: {', '.join(valid_formats)}"

    # 获取所有音频文件
    audio_files = []
    supported_formats = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]

    for filename in os.listdir(audio_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in supported_formats:
            audio_files.append(os.path.join(audio_folder, filename))

    if not audio_files:
        return f"在 {audio_folder} 中未找到支持的音频文件。支持的格式: {', '.join(supported_formats)}"

    # 记录开始时间
    start_time = time.time()
    total_files = len(audio_files)
    logger.info(f"开始批量转录 {total_files} 个文件，输出格式: {output_format}")

    # 预加载模型以避免重复加载
    try:
        get_whisper_model(model_name, device, compute_type)
        logger.info(f"已预加载模型: {model_name}")
    except Exception as e:
        logger.error(f"预加载模型失败: {str(e)}")
        return f"批处理失败: 无法加载模型 {model_name}: {str(e)}"

    # 处理每个文件
    results = []
    success_count = 0
    error_count = 0
    total_audio_duration = 0

    # 处理每个文件
    for i, audio_path in enumerate(audio_files):
        file_name = os.path.basename(audio_path)
        elapsed = time.time() - start_time

        # 报告进度
        progress_msg = report_progress(i, total_files, elapsed)
        logger.info(f"{progress_msg} | 当前处理: {file_name}")

        # 执行转录
        try:
            result = transcribe_audio(
                audio_path=audio_path,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                language=language,
                output_format=output_format,
                beam_size=beam_size,
                temperature=temperature,
                initial_prompt=initial_prompt,
                output_directory=output_folder
            )

            # 检查结果是否包含错误信息
            if result.startswith("错误:") or result.startswith("转录过程中发生错误:"):
                logger.error(f"转录失败: {file_name} - {result}")
                results.append(f"❌ 失败: {file_name} - {result}")
                error_count += 1
                continue

            # 如果转录成功，提取输出路径信息
            if result.startswith("转录成功"):
                # 从返回消息中提取输出路径
                output_path = result.split(": ")[1] if ": " in result else "未知路径"
                success_count += 1
                results.append(f"✅ 成功: {file_name} -> {os.path.basename(output_path)}")

                # 提取音频时长
                audio_duration = 0
                if output_format.lower() == "json":
                    # 尝试从输出文件中解析音频时长
                    try:
                        import json
                        # 从输出文件中读取JSON内容
                        with open(output_path, "r", encoding="utf-8") as json_file:
                            json_content = json_file.read()
                            json_data = json.loads(json_content)
                            audio_duration = json_data.get("duration", 0)
                    except Exception as e:
                        logger.warning(f"无法从JSON文件中提取音频时长: {str(e)}")
                        audio_duration = 0
                else:
                    # 尝试从文件名中提取音频信息
                    try:
                        # 这里我们不能直接访问info对象，因为它在transcribe_audio函数的作用域内
                        # 使用一个保守的估计值或从结果字符串中提取信息
                        audio_duration = 0  # 默认为0
                    except Exception as e:
                        logger.warning(f"无法从文件名中提取音频时长: {str(e)}")
                        audio_duration = 0

                # 累加音频时长
                total_audio_duration += audio_duration
        except Exception as e:
            logger.error(f"转录过程中发生错误: {file_name} - {str(e)}")
            results.append(f"❌ 失败: {file_name} - {str(e)}")
            error_count += 1
    # 计算总转录时间
    total_transcription_time = time.time() - start_time
    # 生成批处理结果摘要
    summary = f"批处理完成，总转录时间: {format_time(total_transcription_time)}"
    summary += f" | 成功: {success_count}/{total_files}"
    summary += f" | 失败: {error_count}/{total_files}"
    # 输出结果
    for result in results:
        logger.info(result)
    return summary
