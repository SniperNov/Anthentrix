def assign_devices(model, all, default_device="cuda:0"):
    """
    根据分配字典 `all` 将模型的层分配到对应设备。

    Args:
        model (torch.nn.Module): 模型实例。
        all (dict): 分配规则字典，例如：
            {
                "cpu": {0, 1, 2},
                "crypto": {3, 4, 5},
                "gpu1": {6, 7},
                "gpu2": {8}
            }
        default_device (str): 默认设备（未分配层的设备）。
    """
    layer_device_map = {}

    # 构建 layer -> device 的映射
    for device, layers in all.items():
        for layer in layers:
            layer_device_map[layer] = device

    # 遍历模型的子模块并分配设备
    for name, module in model.named_children():
        # 尝试解析层编号
        layer_id = None
        if "layer" in name:
            try:
                layer_id = int(name.split("_")[-1])  # 假设命名如 layer_0, layer_1
            except ValueError:
                pass

        # 获取目标设备
        if layer_id is not None and layer_id in layer_device_map:
            device = layer_device_map[layer_id]
        else:
            device = default_device  # 如果未明确分配，使用默认设备

        # 将模块移动到目标设备
        print(f"Assigning {name} to {device}")
        module.to(device)
