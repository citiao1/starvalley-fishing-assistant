# 星布谷地自动钓鱼助手

Windows 桌面自动钓鱼助手。程序使用 ONNX Runtime DirectML 进行 GPU 推理，识别上钩图标；使用 OpenCV 模板识别右下角红色体力数字，并根据状态执行自动收杆和自动喂食。

当前推荐使用小体积 DirectML 发行版。发布目录约 325 MB，不依赖本地 Python 或 IDE，解压后即可运行。

## 下载

正式版本：[v1.0.0 DirectML 小体积发布版](https://github.com/citiao1/starvalley-fishing-assistant/releases/tag/v1.0.0)

直接下载：[release-directml-v1.0.0.zip](https://github.com/citiao1/starvalley-fishing-assistant/releases/download/v1.0.0/release-directml-v1.0.0.zip)

训练集备份：[backup-dataset-final-2026-10-02.zip](https://github.com/citiao1/starvalley-fishing-assistant/releases/download/v1.0.0/backup-dataset-final-2026-10-02.zip)

下载程序包后：

1. 解压整个目录。
2. 运行 `星布谷地钓鱼助手_ONNX_directml.exe`。
3. 不要单独复制 exe，`_internal` 目录中的 DLL 和模型文件必须保留。

## 主要功能

- YOLO/ONNX 识别鱼上钩图标。
- OpenCV 固定 ROI 识别红色体力 `0` 和非 `0` 状态。
- 自动收杆和自动喂食可以分别开关。
- 支持预览 / 仅识别模式。
- 支持窗口消息、SendInput 扫描码和虚拟键等输入方式。
- 显示采集分辨率、识别置信度、推理 FPS、推理耗时、输入目标和最近动作。
- 会话日志记录模型加载、屏幕采集、识别异常和输入返回结果。
- F12 全局紧急停止。

## 安全默认值

程序首次启动时：

- 总开关关闭。
- 自动收杆关闭。
- 自动喂食关闭。
- 预览 / 仅识别模式开启。

建议先保持预览模式，确认屏幕分辨率、上钩框和红色数字 ROI 正确，再手动打开需要的自动动作。

## 自动喂食逻辑

自动喂食不是按照“按键发送成功”判定成功。程序会：

1. 连续多帧确认红色数字为 `0`。
2. 发送一次 `Z`。
3. 继续观察体力数字是否恢复为非 `0`。
4. 如果仍然是红 `0`，下一轮识别仍可再次尝试。

因此，日志中的“已发送”只表示 Windows 接收了输入请求，不等于游戏一定执行了喂食。

## 从源码运行

仓库不包含便携 Python 环境。需要自行准备 Python 3.12 或兼容版本，然后安装依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements_onnx_directml.txt
python .\run_app_onnx.py
```

源码运行默认使用 ONNX Runtime DirectML：

```text
FISHING_ASSISTANT_BACKEND=onnx
FISHING_ASSISTANT_ONNX_PROVIDER=directml
```

## 构建发行版

准备好 Python、PyInstaller 和项目依赖后运行：

```powershell
.\build_exe_onnx.ps1 -Provider directml
```

输出目录：

```text
dist\星布谷地钓鱼助手_ONNX_directml
```

构建脚本会把 PyInstaller 缓存放在项目下的 `build\.pyinstaller-cache`。发布时应分发整个 onedir 目录，而不是只分发 exe。

## 排障

日志位置：

```text
app_data\logs\session.log
```

诊断报告：

```text
app_data\diagnostics.json
```

重点查看：

- `模型已加载`：确认模型和推理后端是否正常。
- `屏幕采集已打开`：确认采集分辨率。
- `屏幕采集后端`：确认使用 DXcam 还是回退到 mss。
- `推理线程错误`：查看异常类型和依赖缺失信息。
- `发送 Z` 或 `Z 喂食发送失败`：确认 Windows 输入接口返回结果。
- `检测到红 0 已消失`：确认游戏是否实际恢复了体力。

## 项目结构

```text
fishing_assistant/       GUI、检测引擎、输入控制和配置
models/best.onnx         当前 DirectML 发布模型
templates/               红色体力数字模板
assets/app_icon.ico      应用图标
run_app_onnx.py          DirectML 启动入口
build_exe_onnx.ps1       DirectML 打包脚本
onnx_tools/              ONNX 导出和后端测试工具
```

训练数据、录屏、历史训练输出和本地运行环境不属于程序运行必需内容。旧 PyTorch CUDA 发行版约 4 GB，已不作为当前推荐版本。

## 当前限制

- 尚未完成完整 80 轮训练，当前模型以已验证的实机效果为准。
- 尚未验证所有分辨率、窗口模式和显卡驱动组合。
- 收杆后自动抛竿、关闭结算界面和完整无人值守循环不在当前版本范围内。
- 删除了本地便携 Python 环境后，重新打包需要重新安装 Python 和依赖。
