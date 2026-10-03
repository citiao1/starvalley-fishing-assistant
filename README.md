# 星布谷地自动钓鱼助手

Windows 桌面自动钓鱼助手。程序使用 ONNX Runtime DirectML 进行 GPU 推理，识别上钩图标；使用 OpenCV 模板识别右下角红色体力数字，并根据状态执行自动收杆和自动喂食。

当前正式版使用小体积 DirectML 发行版。解压后的目录约 330 MB，压缩包约 138 MB，不依赖本地 Python 或 IDE，解压后即可运行。

## 下载

正式版本：[v1.1.0 正式版（ONNX DirectML）](https://github.com/citiao1/starvalley-fishing-assistant/releases/tag/v1.1.0-optimized-directml)

直接下载：[starvalley-fishing-assistant-onnx-directml-optimized-20261003.zip](https://github.com/citiao1/starvalley-fishing-assistant/releases/download/v1.1.0-optimized-directml/starvalley-fishing-assistant-onnx-directml-optimized-20261003.zip)

训练集备份：[backup-dataset-final-2026-10-02.zip](https://github.com/citiao1/starvalley-fishing-assistant/releases/download/v1.0.0/backup-dataset-final-2026-10-02.zip)

下载程序包后：

1. 解压整个目录。
2. 运行 `星布谷地钓鱼助手_ONNX_directml.exe`。
3. 不要单独复制 exe，`_internal` 目录中的 DLL 和模型文件必须保留。
4. 不要直接在压缩包内运行，建议解压到本地磁盘后再启动。

## 主要功能

- YOLO/ONNX 识别鱼上钩图标。
- OpenCV 固定 ROI 识别红色体力 `0` 和非 `0` 状态。
- 自动收杆和自动喂食可以分别开关。
- 支持预览 / 仅识别模式。
- 支持窗口消息、SendInput 扫描码和虚拟键等输入方式。
- 显示采集分辨率、识别置信度、推理 FPS、推理耗时、输入目标和最近动作。
- 显示最近 2 秒滑动 FPS、推理 p50/p95、结果年龄、丢帧计数和 ONNX 实际 provider。
- 会话日志记录模型加载、屏幕采集、识别异常和输入返回结果。
- F12 全局紧急停止。

## 安全默认值

程序首次启动时：

- 总开关关闭。
- 自动收杆关闭。
- 自动喂食关闭。
- 预览 / 仅识别模式开启。

建议先保持预览模式，确认屏幕分辨率、上钩框和红色数字 ROI 正确，再手动打开需要的自动动作。

首次使用建议按以下顺序操作：

1. 启动游戏并确认游戏窗口可以正常显示。
2. 启动助手，保持“预览 / 仅识别”开启。
3. 选择实际显示游戏的屏幕，确认预览画面和红色数字 ROI 对齐。
4. 点击开始检测，观察模型设备、采集后端、目标窗口和识别结果。
5. 确认无误后，再关闭预览模式并分别打开自动收杆或自动喂食。

## 自动喂食逻辑

自动喂食不是按照“按键发送成功”判定成功。程序会：

1. 连续多帧确认红色数字为 `0`。
2. 发送一次 `Z`。
3. 继续观察体力数字是否恢复为非 `0`；`not_found` 也被视为数字变化中的有效过渡。
4. 发送后有最小间隔和较慢的重试间隔，连续输入失败会暂停自动喂食。
5. 收杆和喂食同时满足时，收杆优先，本轮不会同时发送两个动作。

因此，日志中的“已发送”只表示 Windows 接收了输入请求，不等于游戏一定执行了喂食。

## 性能与输入安全

- 默认实验参数为模型输入 `960`、推理 `25 FPS`、预览 `12 FPS`，可在设置中调整并通过 p95 指标校准。
- 咬杆检测始终覆盖全屏，不使用固定咬杆 ROI。
- 自动输入只在游戏窗口仍为前台窗口时发送，不会主动抢回焦点。
- 结果超过配置的最大年龄时只更新识别显示，不触发收杆或喂食。

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

### 先收集信息

程序运行目录下的日志位置：

```text
app_data\logs\session.log
app_data\startup_error.log
app_data\qt_dll_preload_error.log
```

其中：

- `session.log`：模型加载、屏幕采集、识别异常、前台窗口和输入返回结果。
- `startup_error.log`：程序启动阶段异常，例如 DLL、Qt 或资源加载失败。
- `qt_dll_preload_error.log`：Qt DLL 预加载失败时生成。

源码环境可以运行下面的诊断脚本：

```text
python .\diagnose_fishing_assistant.py
```

诊断报告：

```text
app_data\diagnostics.json
```

### 常见问题

#### 1. 双击后没有窗口，或程序立即退出

常见原因是只复制了 exe、没有完整解压，或者 `_internal` 中的 Qt / ONNX Runtime DLL 不完整。

处理方法：

1. 删除当前解压目录，重新下载并完整解压 ZIP。
2. 确认 exe 与 `_internal` 位于同一发布目录。
3. 不要重命名、移动或单独复制 exe。
4. 查看 `app_data\startup_error.log`；如果存在 `WinError 126`、`WinError 127` 或 Qt DLL 错误，优先重新解压并更新 Windows。

#### 2. 报“找不到 ONNX 权重”或“找不到红 0 模板”

确认以下文件存在：

```text
_internal\models\best.onnx
_internal\templates\feed_red_0.png
_internal\templates\feed_red_2.png
```

不要从 `_internal` 中单独复制文件，也不要只发送 exe。

#### 3. DirectML 不可用、推理启动失败或自动回退 CPU

DirectML 依赖 Windows 图形驱动。先更新核显 / 独显驱动并重启程序，然后在“检测参数”中确认 `ONNX 推理后端` 为 `DirectML`。

如果日志或界面中没有 `DmlExecutionProvider`：

1. 先选择 `CPU` 验证模型和画面采集是否正常。
2. 如果 CPU 可以运行，说明主要问题是 DirectML 或显卡驱动，不是模型文件。
3. CPU 模式速度较慢，属于兼容性备用方案。

#### 4. 预览黑屏、画面不对或识别的是另一块屏幕

在“检测参数”中选择实际显示游戏的屏幕。多显示器环境下，默认值会优先选择主屏，但不一定是游戏所在屏幕。

同时确认：

- 游戏没有被最小化。
- 游戏窗口或无边框窗口仍然可见。
- `session.log` 中存在“屏幕采集已打开”。
- `屏幕采集后端` 显示为 `dxcam` 或 `mss fallback`。回退到 mss 不一定是错误，但性能可能较低。

#### 5. 能识别，但不收杆、不喂食或输入测试失败

自动输入有安全限制，必须同时满足：

- 游戏进程名为 `PetitPlanet.exe`。
- 游戏窗口是当前前台窗口。
- 助手和游戏的权限级别一致；如果游戏以管理员身份运行，助手也要以管理员身份运行。
- 未开启“预览 / 仅识别”。
- 总开关和对应的自动收杆 / 自动喂食开关已打开。

日志中如果出现“目标游戏窗口未在前台”“目标为管理员，本程序非管理员”或“已拒绝发送”，先把游戏切到前台，再重试。输入测试会延迟约 2 秒发送，提交测试后请立即切回游戏窗口。

#### 6. 自动喂食提示“连续输入失败已暂停”

这表示程序连续多次没有得到 Windows 输入接口的成功返回，常见原因是游戏不在前台、权限不匹配或输入方式不兼容。

处理方法：

1. 关闭自动动作，确认游戏进程和前台窗口。
2. 在设置中切换输入方式，优先尝试 `SendInput（扫描码）`。
3. 修复权限或前台问题后，停止并重新启动检测会话。
4. 如果按过 F12，紧急停止会锁定本次会话的自动输入，必须重新启动检测会话才能解除。

#### 7. 识别框不准、没有检测到上钩或红色数字状态错误

先保持预览模式，检查：

- 游戏是否在正确显示器和正确分辨率上。
- 上钩图标是否出现在预览画面中。
- 右下角红色数字 ROI 是否覆盖实际体力数字。
- 游戏 UI 是否被缩放、裁剪或其他窗口遮挡。

可以在设置中降低或提高置信度、调整模型输入尺寸和推理 FPS。不要一开始就开启自动动作，先用预览确认识别结果。

#### 8. 修改设置后没有立即生效

设置会自动保存到：

```text
app_data\settings.json
```

部分推理参数会在下一次安全推理边界生效。如果设置文件损坏或启动后参数异常，关闭程序后删除 `app_data\settings.json`，重新启动即可恢复安全默认值；这不会删除模型和模板。

#### 9. 诊断显示 `torch`、`ultralytics` 或 `torchvision` 缺失

这是当前 ONNX DirectML 发布包的正常现象。发布包为了减小体积，不包含训练和 PyTorch 推理依赖。

发布版主要关注：

- `onnx_runtime.ok = true`
- `feed_templates.ok = true`
- `screen_capture.ok = true`
- `active_providers` 中有 `DmlExecutionProvider` 或 `CPUExecutionProvider`

`yolo_model` 和 `torchvision_nms` 只用于源码训练 / PyTorch 诊断，不影响已打包的 ONNX 运行版。

### 日志关键词

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
