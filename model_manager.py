#!/usr/bin/env python3
"""
模型管理模块
负责Whisper模型的加载、缓存和管理
"""

import os
import time
import logging
from typing import Dict, Any
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline

# 日志配置
logger = logging.getLogger(__name__)

# 全局模型实例缓存
model_instances = {}

def get_whisper_model(model_name: str, device: str, compute_type: str) -> Dict[str, Any]:
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

def get_model_info() -> str:
    """
    获取可用的Whisper模型信息

    Returns:
        str: 模型信息的JSON字符串
    """
    import json

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