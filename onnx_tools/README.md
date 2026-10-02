# ONNX GPU 实验

当前主程序仍默认使用 PyTorch CUDA。这里的脚本用于验证替换推理后端，不会自动改变主程序。

## 导出

```powershell
python onnx_tools\export_model.py `
  --weights runs\fishing_yolo11n_v1\weights\best.pt `
  --output-dir D:\Temp\starvalley_onnx_export_dynamic
```

导出使用动态输入、`1280` 最大尺寸、不内置 NMS，便于复用当前矩形补边逻辑。

## 对比

需要把兼容 CUDA 12.x 的 `onnxruntime-gpu` 放在独立环境或目录中，并确保 CUDA/cuDNN DLL 可以被找到：

```powershell
python onnx_tools\benchmark_backends.py `
  --weights runs\fishing_yolo11n_v1\weights\best.pt `
  --onnx D:\Temp\starvalley_onnx_export_dynamic\best.onnx `
  --images dataset_final\images\val
```

如果需要验证不依赖 CUDA 运行库的 GPU 路线，可以安装 `onnxruntime-directml`，然后把
`--provider cuda` 改成 `--provider directml`。DirectML 的包体明显更小，但性能需要按
实际游戏画面实测，不能直接视为 CUDA 的等价替代。

当前 DirectML 打包探针约为 325MB，已实际启动验证；它是独立入口，不会改变主程序的
PyTorch CUDA 默认链路。

只有在框位置、置信度和触发样本一致后，才进入 EXE 后端切换和运行库裁剪。
