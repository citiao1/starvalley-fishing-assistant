# 星布谷地自动钓鱼助手备份范围

这个仓库只保存可以复现和维护软件的必要内容：

- `fishing_assistant/`：界面、检测、输入控制和配置源码
- `run_app.py`：程序入口
- `templates/`：自动喂食红色数字模板
- `models/best.onnx`：当前发布模型
- `assets/app_icon.ico`：应用图标
- `build_exe_onnx.ps1` 和 `星布谷地钓鱼助手_ONNX_directml.spec`：EXE 构建配置
- `fishing_assistant/onnx_detector.py` 和 `onnx_tools/`：ONNX 推理代码
- `diagnose_fishing_assistant.py`、`opencv_feed_zero.py`、`rank_bite_frames.py`：诊断和数据分析工具
- `requirements.txt`、交接文档和软件说明

以下内容不属于项目运行必需内容，不上传也不保留在发布目录：

- `tools/`：portable Python 和全部第三方运行库
- `dist/`、`build/`：构建产物
- `.codegraph/`：本地代码索引数据库
- `dataset*`、`Ultralytics/`、`runs/` 中的训练输出
- 录屏、标注压缩包、缓存和运行日志

这样备份仓库保持较小，同时不影响本机继续训练、测试和打包。

## 当前推理路线

当前默认发行链路是 ONNX Runtime DirectML。它保留 GPU 推理能力，同时将发布目录控制在约 325 MB。
