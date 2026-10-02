# 星布谷地自动钓鱼助手

这是一个 Windows 桌面 GUI，使用 YOLO/ONNX 识别上钩图标，使用固定 ROI 的 OpenCV 模板识别红色体力数字。

当前发布版本使用 ONNX Runtime DirectML，体积约 325 MB，可使用 GPU 推理。旧的 PyTorch CUDA
版本体积约 4 GB，已不作为发行版本。

## 安全默认值

- 启动后总开关关闭。
- 自动收杆关闭。
- 自动喂食关闭。
- 预览 / 仅识别模式开启。
- F12 为全局紧急停止，会关闭自动输入并锁定本次会话的动作。
- 输入方式默认使用窗口消息；如果游戏不响应，可在“检测参数”中切换为
  `SendInput（扫描码）` 或 `SendInput（虚拟键）`。
- 屏幕采集优先使用 DXcam；如果显卡捕获不可用，会自动回退到 mss，并在日志中记录原因。
- 上钩模型默认置信度为 `0.02`。该模型在真实测试集上的有效分数明显低于通用
  YOLO 默认阈值，因此由连续多帧确认和只保留 `bite_icon` 类共同抑制误检。

## 使用发布版本

从 GitHub Releases 下载 `星布谷地钓鱼助手_ONNX_directml`，解压后运行目录中的
`星布谷地钓鱼助手_ONNX_directml.exe`。必须保留整个目录，不能只复制 exe。

## 从源码运行

```powershell
python .\run_app_onnx.py
```

启动后先保持预览模式，确认画面分辨率、上钩框和红色数字 ROI 正确，再由用户主动关闭预览模式并打开需要的自动动作。

## 构建 DirectML 版本

```powershell
.\build_exe_onnx.ps1 -Provider directml
```

构建结果为 `dist\星布谷地钓鱼助手_ONNX_directml`，模型和运行时依赖会一起放入目录。
请把整个目录一起分发，不能只复制 exe，因为 Qt、ONNX Runtime、DirectML 和 OpenCV 的 DLL
位于 `_internal` 目录。

构建脚本默认把 PyInstaller 缓存写入项目下的 `build\.pyinstaller-cache`，不会把项目依赖下载到
C 盘。

## 排障

- 界面右侧显示实时识别状态。
- 会话日志写入 `app_data\logs\session.log`。
- 预览画面会绘制 `bite_icon` 框和红色数字 ROI。
- 实时状态会显示预览 FPS、推理 FPS、推理耗时、输入目标窗口和最近一次
  Win32 输入返回结果，还会显示采集耗时和当前采集后端。
- 模型、模板或屏幕采集失败时会在界面和日志中显示异常类型与路径。
- 可以运行 `.\tools\python312_portable\python.exe .\diagnose_fishing_assistant.py` 生成完整诊断报告，报告位于 `app_data\diagnostics.json`。

训练数据、`runs`、`build` 和 pip 下载缓存不属于发行包；分发时只需要
`dist\星布谷地钓鱼助手` 文件夹。
