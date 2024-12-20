import io
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

def log_to_buffer(buffer, message):
    """将日志写入缓冲区"""
    buffer.write(message + "\n")

def flush_buffer(buffer):
    """一次性输出缓冲区内容"""
    with open("output_log.txt", "w") as log_file:
        log_file.write(buffer.getvalue())
        print(buffer.getvalue())
    buffer.close()

def summarize_device_distribution(model, inputs, buffer):
    """将模型参数和输入按设备统计并存储"""
    device_summary = {"cpu": [], "cuda": []}
    
    # 分类模型参数
    for name, param in model.named_parameters():
        device_type = str(param.device).split(":")[0]
        device_summary[device_type].append(name)

    # 分类输入张量
    for key, value in inputs.items():
        device_type = str(value.device).split(":")[0]
        device_summary[device_type].append(f"Input {key}")

    # 写入缓冲区
    for device, layers in device_summary.items():
        log_to_buffer(buffer, f"\n--- {device.upper()} ---")
        for layer in layers:
            log_to_buffer(buffer, layer)

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def check_device(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name} is on {param.device}")
    print(f"Model is on {next(model.parameters()).device}")

def check_inputs(inputs):
    for key, value in inputs.items():
        print(f"Inputs {key}: on device {value.device}")

def assign_devices(model, all_layers, default_device="cuda:0"):
    """
    根据分配字典 `all` 将模型的层分配到对应设备。

    Args:
        model (torch.nn.Module): 模型实例。
        all (dict): 分配规则字典，例如：
            {
                "cpu": {0, 1, 2},
                "crypto": {3, 4, 5},
                "cuda:0": {6, 7},
                "cuda:1": {8}
            }
        default_device (str): 默认设备（未分配层的设备）。
    """
    layer_device_map = {}

    # 构建 layer -> device 的映射
    for device, layers in all_layers.items():
        for layer in layers:
            layer_device_map[layer] = device

    # 遍历模型所有模块
    for full_name, module in model.named_modules():
        # 提取层编号
        layer_id = None
        if "layer" in full_name:
            try:
                layer_id = int(full_name.split(".")[-1])  # 假设格式如 "transformer.h.0"
            except ValueError:
                pass

        # 确定目标设备
        device = layer_device_map.get(layer_id, default_device)

        # 移动模块到设备
        print(f"Assigning {full_name} to {device}")
        module.to(device)

def log_device_info(model, inputs, outputs):
    """记录模型、输入和输出的设备信息"""
    print("\n--- Device Info ---")

    # 输入设备
    print("\n[Input Tensors]")
    for key, value in inputs.items():
        print(f"{key}: {value.device}")

    # 模型层设备
    print("\n[Model Layers]")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")

    # 输出设备
    print("\n[Output Tensors]")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            print(f"{key}: {value.device}")
    elif isinstance(outputs, (list, tuple)):
        for idx, value in enumerate(outputs):
            print(f"Output {idx}: {value.device}")
    else:
        print(f"Output: {outputs.device}")

def run_demo(rank, world_size, model_path, input, all_layers):
    setup(rank, world_size)
    print("loading model...")
    model = AutoModel.from_pretrained(model_path)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    assign_devices(model, all_layers)
    model = DDP(model, device_ids=[rank], output_device=rank)

    input_token = tokenizer(input, return_tensors="pt").to(rank)

    print(f"Rank {rank} running inference...")
    check_device(model)
    
    # 初始化缓冲区
    output_buffer = io.StringIO()

    # 按设备分类记录
    summarize_device_distribution(model, inputs, output_buffer)

    outputs = model(**input_token)

    dist.barrier()
    flush_buffer(output_buffer)
    log_device_info(model, input_token, outputs)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    inputs = "Hello! This is Weiyu!"
    model_path = "Qwen2.5-1.5B"
    # 分配规则
    all_l = {
    "cpu": {0, 1, 2},
    # "crypto": {3, 4, 5},
    "cuda:0": {6, 7},
    "cuda:1": {8},
    }
    torch.multiprocessing.spawn(run_demo, args=(world_size, model_path, inputs, all_l), nprocs=world_size, join=True)


