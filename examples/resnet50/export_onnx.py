import torch
import torchvision.models as models

# 加载预训练的 ResNet-50 模型
model = models.resnet50(pretrained=True)
model.eval()

# 创建一个示例输入张量
dummy_input = torch.randn(16, 3, 224, 224)

# 导出为 ONNX, static batch size
onnx_path = "resnet50.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True)
print(f"Exported model to {onnx_path}")

# 导出为 ONNX, dynamic batch size
onnx_path_dynamic = "resnet50_dynamic.onnx"
torch.onnx.export(model, dummy_input, onnx_path_dynamic, export_params=True, 
                  input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
print(f"Exported model to {onnx_path_dynamic}")