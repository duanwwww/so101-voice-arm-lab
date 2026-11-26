# SenseVoiceSmall 本地语音与仿真链路

当前网页默认用 SenseVoiceSmall INT8 转写中文，模型在 Windows 控制电脑的 CPU 上运行。OpenCV 提取红块工位，校方 DeepSeek 或本地规则接收文字与观测，控制器校验后逐段执行，并把实际运动和落位结果反馈到网页。ArUco 贴纸作为后续候选，当前控制链仍以红色物体和已标定工位作视觉输入。[当前工作流](workflow.md) · [运行入口](../README.md)

## 本机运行设置

语音由 `sherpa-onnx==1.12.40` 运行，4 个 CPU 线程、中文 `zh`、INT8、`use_itn=True`。ITN 将识别出的数字等内容转成常用书写形式。音频通过 PyAV 解码为 16 kHz 单声道 PCM，再送入同一个识别入口，每条最长 30 秒。模型使用 `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` 固定提交 `2365baeacb507f821a0c8120fcee3d484dba7a07`。下载后检查模型、词表和许可文件的 SHA-256，缓存到 `outputs/models/sensevoice-small-int8-2024-07-17`。本地推理没有按次 API 费用。[模型发布页](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/2365baeacb507f821a0c8120fcee3d484dba7a07) · [本项目加载与解码](../arm_lab/sensevoice.py)

首次在工程根目录准备模型，随后启动网页。下载与每次转写使用各自的授权入口，已准备好的模型供后续进程复用。

~~~powershell
uv sync --frozen --extra voice --extra data
uv run --frozen --extra voice --extra data python -m arm_lab.sensevoice --approve-download
uv run --frozen --extra voice --env-file .env python -m arm_lab serve --asr-engine sensevoice
~~~

网页录音结束后，确认转写请求，看到文字后可校对再发送动作。启动参数 `--sensevoice-dir` 可指定模型目录，`--asr-engine whisper` 可复用旧的 Whisper 入口。SenseVoiceSmall 的模型格式与 Whisper 模型目录分别配置。[网页语音入口](../arm_lab/model_access.py)

## 采用相同录音作比较

语音基准复用此前 24 段、5 位说话人的 AISHELL-1 录音，并逐文件校验原有哈希。它们来自官方训练划分，本项目事先按说话人分成 9 段开发、15 段留出。这里的留出只表示本项目的划分。另有 4 段已有的 Windows 中文合成口令，涉及搬运、定量平移、缺失量值和否定。真实朗读与合成口令分别计分。[24 段清单](../results/safety/asr_inventory.json) · [合成口令及来源](../results/safety/prompt_trial_audio)

每段使用未编辑的识别文本。字符错误率沿用历史的繁简、空白和标点处理；中文数字与阿拉伯数字的字符差仍计入。方向、数字、量纲、否定词另按事前参考事件逐项核对，以免字符层面的书写差掩盖实际命令差异。耗时包含音频读取和转写，另列一次模型加载，下载排除在转写耗时之外。与旧 Whisper 的计时来自同一电脑不同时间的运行，可用来估计等待量级。

| 相同的 24 段真人录音 | Whisper Small beam=1 | Whisper Small beam=3 | SenseVoiceSmall INT8 |
| --- | ---: | ---: | ---: |
| 全部字符错误率 | 20.51%（73/356） | 17.70%（63/356） | 10.67%（38/356） |
| 全部整句相同 | 5/24 | 6/24 | 11/24 |
| 15 段组内留出字符错误率 | 23.85%（57/239） | 20.92%（50/239） | 13.39%（32/239） |
| 15 段组内留出整句相同 | 3/15 | 3/15 | 6/15 |
| 全部转写总耗时 | 57.033 s | 56.725 s | 3.841 s |
| 平均每段转写耗时 | 2.376 s | 2.364 s | 0.160 s |

SenseVoiceSmall 本轮另用 2.072 s 加载模型，24 段原音合计 111.267 s。每种设置的成绩对应一次固定顺序运行，首次请求还会受到模型加载影响。原样文本、原音哈希及逐段计时都保存在本轮报告。[SenseVoice 与历史 Whisper 的逐句配对](../results/safety/sensevoice/speech.json)

SenseVoice 把“一百九十厘米”写成了“190 厘米”，修复了旧模型写成“190 米”的量纲错误。但“右转弯”仍成了“又转弯”，“自然向右偏”仍成了“自然下右偏”。转写对方向的这两处错误必须继续保留在验收项中。合成口令的搬运句也仍把“右边的工位”写成“右干的工位”，需要规划器澄清或结合场景处理。

| 原参考文本中逐次标出的事件 | Whisper beam=1 | Whisper beam=3 | SenseVoice |
| --- | ---: | ---: | ---: |
| 运动方向 | 7/9 | 7/9 | 7/9 |
| 数值 | 12/12 | 12/12 | 12/12 |
| 对应单位 | 10/12 | 11/12 | 12/12 |
| 长度数值和单位同时正确 | 4/6 | 5/6 | 6/6 |
| 原有否定或停止标志 | 9/10 | 9/10 | 10/10 |

以上分母按原文事件次数统计。SenseVoice 保留了原本的“未消毒”，同一句却把“微整形”写成“未整形”，新增了一个可能误导语义的否定词。原有否定标志保留 10/10 仍需配合对新增词和整句的检查。自动词表也单列按“有此类词的句子”计算的结果，与这里按事件次数计分的分母不同。[逐事件复核、原文和输出](../results/safety/sensevoice/keyword_review.json)

| 已有合成口令 | SenseVoice 原始输出 | 语音结果 |
| --- | --- | --- |
| 把红色木块放到右边的工位 | 把红色木块放到右干的工位。 | 目标方向词错误 |
| 向左移动十二毫米 | 向左移动12毫米。 | 数字等价，动作量值保留 |
| 向前移动五 | 向前移动5。 | 原话缺单位，转写仍需后续追问 |
| 不要向前移动一厘米 | 不要向前移动1厘米。 | 否定与量值保留 |

本轮第一次模型加载遇到 Windows 选中了旧版 ONNX Runtime DLL，进程在生成任何转写前退出。修复为从当前 uv 环境的明确路径先加载已安装的 ONNX Runtime DLL，再加载 sherpa-onnx，随后成功跑完 28 段。这次故障记入环境加载失败。[失败原因与修复记录](../results/safety/sensevoice/load_failure.json)

默认语音切换与澄清修复完成后，99 项自动化回归全部通过，包含网页录音格式解码、模型授权、录音长度限制、原有规划和运动流程，以及带起点的工位补答。[本轮回归记录](../results/safety/sensevoice/verification.json)

经操作者批准，可以用下面的命令对相同录音生成一次新的本地结果。初次准备公开语料可参考原语音实验的数据下载步骤。

~~~powershell
uv run --frozen --extra voice --extra data python -m scripts.evaluate_sensevoice --approve-local-asr --out outputs/sensevoice_repeat.json
~~~

[语料准备和历史 Whisper 实验](asr_benchmark.md) · [本轮评测脚本](../scripts/evaluate_sensevoice.py)

## 同一仿真场地中的语音链路

四条已有合成录音依次通过真实 HTTP 上传接口、本地 SenseVoiceSmall、未经编辑的转写、本地规则规划、控制校验、MuJoCo 运动、相机监控和文字回执。从 A 工位开始，四条之间没有重置场地。仿真物理位置、关节和末端位移另行记录，用来核对回执；运动期间页面监控请求和心跳也走接口。[完整请求、回执和物理测量](../results/safety/sensevoice/flow_local.json)

| 次序 | 任务 | 实际响应 | 原始目标是否达到 |
| --- | --- | --- | --- |
| 1 | 把红块放到右边的工位 | “右干”无法由规则确认目标，追问 A/B/C；木块仍在 A | 否，搬运未执行 |
| 2 | 向左移动十二毫米 | 执行 +Y 12 mm；三维位移误差约 0.226 mm | 是 |
| 3 | 不要向前移动一厘米 | 忽略该禁止句，保持原位 | 是 |
| 4 | 向前移动五 | 追问距离单位，保持原位 | 是 |

本轮本地链路共达到 3/4 个预期结果，其中 1 次实际运动、2 次应当保持静止。对搬运的追问避免了猜错目的地，但原始搬运任务尚未完成。文字到 DeepSeek 的历史 54 对留出成绩和 12 条仿真重放仍单独保留。此处新链路的云端规划请求为 0，反馈终点为网页文字回执。[历史规划与运动验证](planner100_results.md)

随后单独运行恢复实验，仍使用那条搬运录音和真实 SenseVoice。识别到“右干”后，本地规则先询问目标工位，场地和关节保持原位。测试脚本模拟用户明确补答 `B`，并通过 `reply_to` 将补答绑定到上一问。第一次补答仍被重复追问，原因是旧澄清逻辑把 B 接在无法识别的“右干的工位”后面。修复后，单次搬运会以明确补答替换未知目的地，保留原任务和起点，再执行。[补答首次失败记录](../results/safety/sensevoice/recovery.json) · [澄清代码](../arm_lab/planner.py)

修复后的同一场地完成 A 到 B 搬运，三次独立物理采样均显示木块在 B 且接触桌面。恢复全过程没有重置场地，云端调用为 0。这个成功包含测试提供的明确人工补答，原始语音单次任务成绩仍为 3/4，不能把补答成功加回为 4/4。[修复后的逐轮回执和三帧测量](../results/safety/sensevoice/recovery_fixed.json)

获得本地语音授权后可以用下面命令重新从四份录音运行。新输出路径用于保存每次真实运行的记录。

~~~powershell
uv run --frozen --extra voice --extra data python -m scripts.evaluate_sensevoice_flow --approve-local-asr --out outputs/sensevoice_flow_repeat.json
uv run --frozen --extra voice --extra data python -m scripts.evaluate_sensevoice_recovery --approve-local-asr --out outputs/sensevoice_recovery_repeat.json
~~~
