# 星布谷地自动钓鱼助手备份范围

这个仓库只保存可以复现和维护软件的必要内容：

- `fishing_assistant/`：界面、检测、输入控制和配置源码
- `run_app.py`：程序入口
- `templates/`：自动喂食红色数字模板
- `runs/fishing_yolo11n_v1/weights/best.pt`：当前运行模型
- `assets/app_icon.ico`：应用图标
- `build_exe.ps1` 和 `*.spec`：EXE 构建配置
- `fishing_assistant/onnx_detector.py` 和 `onnx_tools/`：ONNX GPU 实验代码
- `diagnose_fishing_assistant.py`、`opencv_feed_zero.py`、`rank_bite_frames.py`：诊断和数据分析工具
- `requirements.txt`、交接文档和软件说明

以下内容保留在本机但不上传：

- `tools/`：portable Python 和全部第三方运行库
- `dist/`、`build/`：构建产物
- `.codegraph/`：本地代码索引数据库
- `dataset*`、`Ultralytics/`、`runs/` 中的训练输出
- 录屏、标注压缩包、缓存和运行日志

这样备份仓库保持较小，同时不影响本机继续训练、测试和打包。

## 当前推理路线

当前默认运行链路仍是 PyTorch CUDA，作为准确率基线。下一步会在独立环境中导出 ONNX，并使用 ONNX Runtime GPU 做逐帧对比；确认预处理、框位置、置信度和触发逻辑一致后，再考虑裁剪 ONNX Runtime 运行库。
