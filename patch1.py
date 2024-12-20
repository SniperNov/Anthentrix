import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def assign_devices(model, all_layers, default_device="cuda:0"):
    """
    根据分配字典 `all_layers` 将模型的层分配到对应设备。

    Args:
        model (torch.nn.Module): 模型实例。
        all_layers (dict): 分配规则字典，例如：
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


class HybridModel(nn.Module):
    """
    基于设备分配的混合模型，支持动态设备分配。
    """
    def __init__(self, model, all_layers, default_device="cuda:0"):
        super().__init__()
        self.model = model
        self.all_layers = all_layers
        self.default_device = default_device
        assign_devices(self.model, self.all_layers, self.default_device)

    def forward(self, **inputs):
        # 数据根据输入设备调整
        for key, value in inputs.items():
            inputs[key] = value.to(self.default_device)
        return self.model(**inputs)


def main():
    # 加载预训练模型
    model_path = "Qwen2.5-1.5B"
    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 分配规则
    all_layers = {
        "cpu": {0, 1, 2},  # 层 0, 1, 2 分配到 CPU
        "cuda:0": {3, 4, 5},  # 层 3, 4, 5 分配到 GPU0
        "cuda:1": {6, 7, 8},  # 层 6, 7, 8 分配到 GPU1
    }

    # 构建混合模型
    print("Building hybrid model...")
    hybrid_model = HybridModel(model, all_layers)

    # 输入测试
    input_text = "Hello! This is Weiyu!"
    print(f"Input text: {input_text}")

    # Tokenize 输入
    inputs = tokenizer(input_text, return_tensors="pt")
    print(f"Tokenized inputs: {inputs}")

    # 推理
    print("Running inference...")
    outputs = hybrid_model(**inputs)
    print(f"Outputs: {outputs}")


if __name__ == "__main__":
    main()
