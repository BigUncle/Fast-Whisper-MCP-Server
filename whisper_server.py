#!/usr/bin/env python3
"""
基于Faster Whisper的语音识别MCP服务
"""

import os
import json
import logging
from typing import Optional, Dict, List
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from mcp.server.fastmcp import FastMCP, Context

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastMCP服务器实例
mcp = FastMCP(
    name="whisper-server",
    version="0.1.0",
    dependencies=["faster-whisper>=0.9.0", "torch==2.6.0+cu126", "torchaudio==2.6.0+cu126"]
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
    
    info = {
        "available_models": models,
        "default_model": "large-v3",
        "available_devices": devices,
        "default_device": "cuda" if torch.cuda.is_available() else "cpu",
        "available_compute_types": compute_types,
        "default_compute_type": "float16" if torch.cuda.is_available() else "int8",
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        }
    
    return json.dumps(info, indent=2)

def get_whisper_model(model_name: str, device: str, compute_type: str) -> Dict:
    """
    获取或创建Whisper模型实例
    
    Args:
        model_name: 模型名称 (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: 运行设备 (cpu, cuda)
        compute_type: 计算类型 (float16, int8)
    
    Returns:
        dict: 包含模型实例和配置的字典
    """
    global model_instances
    
    # 生成模型键
    model_key = f"{model_name}_{device}_{compute_type}"
    
    # 如果模型已实例化，直接返回
    if model_key in model_instances:
        return model_instances[model_key]
    
    # 自动检测设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
    
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
            compute_type=compute_type
        )
        
        # 批处理设置 - 默认启用批处理以提高速度
        batched_model = None
        batch_size = 0
        
        if device == "cuda":  # 只在CUDA设备上使用批处理
            # 根据显存大小确定合适的批大小
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                # 根据GPU显存动态调整批大小
                if gpu_mem > 16e9:  # >16GB
                    batch_size = 32
                elif gpu_mem > 12e9:  # >12GB
                    batch_size = 16
                elif gpu_mem > 8e9:   # >8GB
                    batch_size = 8
                else:                # 较小显存
                    batch_size = 4
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
            'batch_size': batch_size
        }
        
        # 缓存实例
        model_instances[model_key] = result
        return result
        
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise

@mcp.tool()
def transcribe(audio_path: str, model_name: str = "large-v3", device: str = "auto", 
              compute_type: str = "auto", language: str = None, output_format: str = "vtt") -> str:
    """
    使用Faster Whisper转录音频文件
    
    Args:
        audio_path: 音频文件路径
        model_name: 模型名称 (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: 运行设备 (cpu, cuda, auto)
        compute_type: 计算类型 (float16, int8, auto)
        language: 语言代码 (如zh, en, ja等，默认自动检测)
        output_format: 输出格式 (vtt或json)
    
    Returns:
        str: 转录结果，格式为VTT字幕或JSON
    """
    # 验证参数
    if not os.path.exists(audio_path):
        return f"错误: 音频文件不存在: {audio_path}"
    
    try:
        # 获取模型实例
        model_instance = get_whisper_model(model_name, device, compute_type)
        
        # 设置转录参数
        options = {
            "language": language,
            "vad_filter": True,  # 使用语音活动检测
            "vad_parameters": {"min_silence_duration_ms": 500},  # VAD参数优化
        }
        
        # 执行转录 - 优先使用批处理模型
        if model_instance['batched_model'] is not None and model_instance['device'] == 'cuda':
            logger.info("使用批处理加速进行转录...")
            # 批处理模型需要单独设置batch_size参数
            segments, info = model_instance['batched_model'].transcribe(
                audio_path, 
                batch_size=model_instance['batch_size'],
                **options
            )
        else:
            logger.info("使用标准模型进行转录...")
            segments, info = model_instance['model'].transcribe(audio_path, **options)
        
        # 将生成器转换为列表
        segment_list = list(segments)
        
        if not segment_list:
            return "转录失败，未获得结果"
        
        # 根据输出格式返回结果
        if output_format.lower() == "vtt":
            return format_vtt(segment_list)
        else:
            return format_json(segment_list, info)
            
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

def format_json(segments, info) -> str:
    """将转录结果格式化为JSON"""
    result = {
        "segments": [{
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        } for segment in segments],
        "language": info.language,
        "duration": info.duration
    }
    return json.dumps(result, indent=2, ensure_ascii=False)

def format_timestamp(seconds: float) -> str:
    """格式化时间戳为VTT格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

@mcp.tool()
def batch_transcribe(audio_folder: str, output_folder: str = None, model_name: str = "large-v3", 
                    device: str = "auto", compute_type: str = "auto") -> str:
    """
    批量转录文件夹中的音频文件
    
    Args:
        audio_folder: 包含音频文件的文件夹路径
        output_folder: 输出文件夹路径，默认为audio_folder下的transcript子文件夹
        model_name: 模型名称
        device: 运行设备
        compute_type: 计算类型
    
    Returns:
        str: 批处理结果摘要
    """
    if not os.path.isdir(audio_folder):
        return f"错误: 文件夹不存在: {audio_folder}"
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(audio_folder, "transcript")
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = []
    for filename in os.listdir(audio_folder):
        if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
            audio_files.append(os.path.join(audio_folder, filename))
    
    if not audio_files:
        return f"在 {audio_folder} 中未找到音频文件"
    
    # 处理每个文件
    results = []
    for i, audio_path in enumerate(audio_files):
        logger.info(f"处理第 {i+1}/{len(audio_files)} 个文件: {os.path.basename(audio_path)}")
        
        # 设置输出文件路径
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vtt_path = os.path.join(output_folder, f"{base_name}.vtt")
        
        # 执行转录
        result = transcribe(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            output_format="vtt"
        )
        
        # 保存结果到文件
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        results.append(f"已转录: {os.path.basename(audio_path)} -> {os.path.basename(vtt_path)}")
    
    summary = f"批处理完成，成功转录 {len(results)}/{len(audio_files)} 个文件\n\n"
    summary += "\n".join(results)
    return summary

if __name__ == "__main__":
    # 运行服务器
    mcp.run()
