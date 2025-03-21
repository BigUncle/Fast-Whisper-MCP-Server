#!/usr/bin/env python3
"""
基于Faster Whisper的语音识别MCP服务
提供高性能的音频转录功能，支持批处理加速和多种输出格式
"""

import os
import json
import logging
import time
from typing import Dict
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline, decode_audio
from mcp.server.fastmcp import FastMCP

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastMCP服务器实例
mcp = FastMCP(
    name="fast-whisper-mcp-server",
    version="0.1.1",
    dependencies=["faster-whisper>=0.9.0", "torch==2.6.0+cu126", "torchaudio==2.6.0+cu126", "numpy>=1.20.0"]
)

# 全局模型实例缓存
model_instances = {}

@mcp.tool()
def get_model_info() -> str:
    """获取可用的Whisper模型信息"""
    models = [
        "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
    ]
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    compute_types = ["float16", "int8"] if torch.cuda.is_available() else ["int8"]

    # 支持的语言列表
    languages = {
        "zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语", "de": "德语",
        "fr": "法语", "es": "西班牙语", "ru": "俄语", "it": "意大利语",
        "pt": "葡萄牙语", "nl": "荷兰语", "ar": "阿拉伯语", "hi": "印地语",
        "tr": "土耳其语", "vi": "越南语", "th": "泰语", "id": "印尼语"
    }

    # 支持的音频格式
    audio_formats = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]

    info = {
        "available_models": models,
        "default_model": "large-v3",
        "available_devices": devices,
        "default_device": "cuda" if torch.cuda.is_available() else "cpu",
        "available_compute_types": compute_types,
        "default_compute_type": "float16" if torch.cuda.is_available() else "int8",
        "cuda_available": torch.cuda.is_available(),
        "supported_languages": languages,
        "supported_audio_formats": audio_formats,
        "version": "0.1.1"
    }

    if torch.cuda.is_available():
        info["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
            "memory_available": f"{torch.cuda.get_device_properties(0).total_memory / 1e9 - torch.cuda.memory_allocated() / 1e9:.2f} GB"
        }

    return json.dumps(info, indent=2)

def get_whisper_model(model_name: str, device: str, compute_type: str) -> Dict:
    """
    获取或创建Whisper模型实例

    Args:
        model_name: 模型名称 (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: 运行设备 (cpu, cuda, auto)
        compute_type: 计算类型 (float16, int8, auto)

    Returns:
        dict: 包含模型实例和配置的字典
    """
    global model_instances

    # 验证模型名称
    valid_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    if model_name not in valid_models:
        raise ValueError(f"无效的模型名称: {model_name}。有效的模型: {', '.join(valid_models)}")

    # 自动检测设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

    # 验证设备和计算类型
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"无效的设备: {device}。有效的设备: cpu, cuda")

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA不可用，自动切换到CPU")
        device = "cpu"
        compute_type = "int8"

    if compute_type not in ["float16", "int8"]:
        raise ValueError(f"无效的计算类型: {compute_type}。有效的计算类型: float16, int8")

    if device == "cpu" and compute_type == "float16":
        logger.warning("CPU设备不支持float16计算类型，自动切换到int8")
        compute_type = "int8"

    # 生成模型键
    model_key = f"{model_name}_{device}_{compute_type}"

    # 如果模型已实例化，直接返回
    if model_key in model_instances:
        logger.info(f"使用缓存的模型实例: {model_key}")
        return model_instances[model_key]

    # 清理GPU内存（如果使用CUDA）
    if device == "cuda":
        torch.cuda.empty_cache()

    # 实例化模型
    try:
        logger.info(f"加载Whisper模型: {model_name} 设备: {device} 计算类型: {compute_type}")

        # 基础模型
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=os.environ.get("WHISPER_MODEL_DIR", None)  # 支持自定义模型目录
        )

        # 批处理设置 - 默认启用批处理以提高速度
        batched_model = None
        batch_size = 0

        if device == "cuda":  # 只在CUDA设备上使用批处理
            # 根据显存大小确定合适的批大小
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                free_mem = gpu_mem - torch.cuda.memory_allocated()
                # 根据GPU显存动态调整批大小
                if free_mem > 16e9:  # >16GB
                    batch_size = 32
                elif free_mem > 12e9:  # >12GB
                    batch_size = 16
                elif free_mem > 8e9:   # >8GB
                    batch_size = 8
                elif free_mem > 4e9:   # >4GB
                    batch_size = 4
                else:                # 较小显存
                    batch_size = 2

                logger.info(f"可用GPU显存: {free_mem / 1e9:.2f} GB")
            else:
                batch_size = 8  # 默认值

            logger.info(f"启用批处理加速，批大小: {batch_size}")
            batched_model = BatchedInferencePipeline(model=model)

        # 创建结果对象
        result = {
            'model': model,
            'device': device,
            'compute_type': compute_type,
            'batched_model': batched_model,
            'batch_size': batch_size,
            'load_time': time.time()
        }

        # 缓存实例
        model_instances[model_key] = result
        return result

    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise


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

    try:
        # 获取模型实例
        model_instance = get_whisper_model(model_name, device, compute_type)

        # 验证语言代码
        if language is not None:
            # 支持的语言列表
            supported_languages = {
                "zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语", "de": "德语",
                "fr": "法语", "es": "西班牙语", "ru": "俄语", "it": "意大利语",
                "pt": "葡萄牙语", "nl": "荷兰语", "ar": "阿拉伯语", "hi": "印地语",
                "tr": "土耳其语", "vi": "越南语", "th": "泰语", "id": "印尼语"
            }

            if language not in supported_languages:
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

        # 尝试使用decode_audio预处理音频，以处理更多格式
        try:
            audio_data = decode_audio(audio_path)
            audio_source = audio_data
            logger.info(f"成功预处理音频: {os.path.basename(audio_path)}")
        except Exception as audio_error:
            logger.warning(f"音频预处理失败，将直接使用文件路径: {str(audio_error)}")
            audio_source = audio_path

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
def format_vtt(segments) -> str:
    """将转录结果格式化为VTT"""
    vtt_content = "WEBVTT\n\n"

    for segment in segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()

        if text:
            vtt_content += f"{start} --> {end}\n{text}\n\n"

    return vtt_content

def format_srt(segments) -> str:
    """将转录结果格式化为SRT"""
    srt_content = ""

    for segment in segments:
        start = format_timestamp_srt(segment.start)
        end = format_timestamp_srt(segment.end)
        text = segment.text.strip()

        if text:
            srt_content += f"{len(srt_content.splitlines()) + 1}\n{start} --> {end}\n{text}\n\n"

    return srt_content

def format_json(segments, info) -> str:
    """将转录结果格式化为JSON"""
    result = {
        "segments": [{
            "id": segments.index(segment),
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [{
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "probability": word.probability
            } for word in segment.words] if hasattr(segment, 'words') and segment.words else []
        } for segment in segments],
        "language": info.language,
        "language_probability": info.language_probability if hasattr(info, 'language_probability') else None,
        "duration": info.duration,
        "all_language_probs": info.all_language_probs if hasattr(info, 'all_language_probs') else None
    }
    return json.dumps(result, indent=2, ensure_ascii=False)

def format_timestamp(seconds: float) -> str:
    """格式化时间戳为VTT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def format_timestamp_srt(seconds: float) -> str:
    """格式化时间戳为SRT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

@mcp.tool()
def batch_transcribe(audio_folder: str, output_folder: str = None, model_name: str = "large-v3",
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

    # 处理每个文件
    results = []
    success_count = 0
    error_count = 0
    total_audio_duration = 0

        # 预加载模型以避免重复加载
    try:
        get_whisper_model(model_name, device, compute_type)
        logger.info(f"已预加载模型: {model_name}")
    except Exception as e:
        logger.error(f"预加载模型失败: {str(e)}")
        return f"批处理失败: 无法加载模型 {model_name}: {str(e)}"

    # 处理进度报告函数
    def report_progress(current, total, elapsed_time):
        progress = current / total * 100
        eta = (elapsed_time / current) * (total - current) if current > 0 else 0
        return (f"进度: {current}/{total} ({progress:.1f}%)" +
                f" | 已用时间: {format_time(elapsed_time)}" +
                f" | 预计剩余: {format_time(eta)}")

    # 格式化时间函数
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    # 处理每个文件
    for i, audio_path in enumerate(audio_files):
        file_name = os.path.basename(audio_path)
        elapsed = time.time() - start_time

        # 报告进度
        progress_msg = report_progress(i, total_files, elapsed)
        logger.info(f"{progress_msg} | 当前处理: {file_name}")

        # 设置输出文件路径
        base_name = os.path.splitext(file_name)[0]
        output_ext = "." + output_format.lower()
        if output_format.lower() == "json":
            output_ext = ".json"
        elif output_format.lower() == "vtt":
            output_ext = ".vtt"
        elif output_format.lower() == "srt":
            output_ext = ".srt"

        output_path = os.path.join(output_folder, f"{base_name}{output_ext}")

        # 执行转录
        try:
            result = transcribe(
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
                continue

            # 检查转录结果是否已成功保存
            if result.startswith("转录成功"):
                logger.info(f"转录结果已保存: {file_name}")
            else:
                logger.error(f"转录未成功保存: {file_name} - {result}")
                continue

            # 提取音频时长（如果是JSON格式）
            audio_duration = 0
            if output_format.lower() == "json":
                try:
                    json_result = json.loads(result)
                    audio_duration = json_result.get("duration", 0)
                    total_audio_duration += audio_duration
                except:
                    pass

            success_count += 1
            duration_info = f" (时长: {audio_duration:.1f}秒)" if audio_duration > 0 else ""
            results.append(f"✅ 成功: {file_name} -> {os.path.basename(output_path)}{duration_info}")

        except Exception as e:
            logger.error(f"转录失败: {file_name} - {str(e)}")
            results.append(f"❌ 失败: {file_name} - {str(e)}")
            error_count += 1

    # 计算总时间和处理速度
    total_time = time.time() - start_time
    processing_speed = total_audio_duration / total_time if total_audio_duration > 0 and total_time > 0 else 0

    # 生成摘要
    summary = f"批处理完成，用时: {format_time(total_time)}\n"
    summary += f"成功: {success_count}/{total_files} ({success_count/total_files*100:.1f}%)\n"
    if error_count > 0:
        summary += f"失败: {error_count}/{total_files} ({error_count/total_files*100:.1f}%)\n"
    if total_audio_duration > 0:
        summary += f"总音频时长: {total_audio_duration:.1f}秒\n"
        summary += f"处理速度: {processing_speed:.2f}x 实时速度\n"
    summary += f"输出目录: {output_folder}\n\n"

    # 添加详细结果
    summary += "详细结果:\n" + "\n".join(results)

    return summary

if __name__ == "__main__":
    # 运行服务器
    mcp.run()
