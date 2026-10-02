# 星布谷地自动钓鱼助手

这是一个 Windows 桌面 GUI 原型，使用现有 `bite_icon` YOLO 权重识别上钩图标，使用固定 ROI 的 OpenCV 模板识别红色体力数字。

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

## 启动

```powershell
.\tools\python312_portable\python.exe .\run_app.py
```

启动后先保持预览模式，确认画面分辨率、上钩框和红色数字 ROI 正确，再由用户主动关闭预览模式并打开需要的自动动作。

## 构建独立程序

```powershell
.\build_exe.ps1
```

构建结果为 `dist\星布谷地钓鱼助手\星布谷地钓鱼助手.exe`，是 onedir 发行目录，模型和运行时依赖会一起放入目录。日志和设置写入 exe 同目录下的 `app_data`。

请把整个 `dist\星布谷地钓鱼助手` 文件夹一起分发。不能只复制其中的 exe，因为 Qt、Torch 和 OpenCV 的 DLL 位于 `_internal` 目录。

当前构建是 GPU 版，约 4.1 GB，其中约 3.86 GB 是 CUDA 版 PyTorch、cuDNN 和 cuBLAS
运行库；模型本身只有约 5 MB。若要显著缩小发行包，需要另做 CPU 版运行时，或改为
ONNX Runtime 方案，不能只删除 `_internal\torch\lib` 中的 DLL。

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
