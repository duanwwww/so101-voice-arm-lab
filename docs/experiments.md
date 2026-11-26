# 实验设计、结果与复现

本页汇总截至 2026 年 11 月 26 日已完成的实验。项目使用固定底座、A/B/C 三个工位和一个边长 24 mm 的红色木块；实际执行后端为 MuJoCo，尚无本组真机实验结果。其他颜色、任意形状及多目标选择未纳入本次验收。

先按 [README](../README.md) 配置一个 uv 环境，再按本文复现。系统设计见 [工作流与技术实现](workflow.md)，实机验证计划见 [真机接入与验收](hardware.md)。原始输入、模型回复和失败记录保存在 `configs/`、`results/`；分阶段的详细记录归入 `docs/archive/`。

## 1. 数据与评价方法

| 实验 | 数据与任务 | 评价依据 |
| --- | --- | --- |
| 语音转写 | AISHELL-1 的 24 段真人普通话，5 位说话人；另有 4 段 Windows 离线合成口令 | 字符错误率（CER）、整句一致、方向/数值/单位/否定词、完整转写耗时 |
| 文本规划 | 自建 100 条中文输入，含场景、持物状态、上下文及参考 JSON；40 条开发、60 条留出 | 原始 JSON 的格式、字段、顺序、坐标轴、符号与数值；追问是否针对缺失信息 |
| 视觉感知 | VigiClaw 的 100 张实拍与 264 张外部仿真图；本项目 100 个双视角摆位、144 个范围边界摆位 | 发布者标注与预测框的 IoU；唯一目标识别；独立仿真物理位置与视觉估计的差异 |
| 运动与反馈 | 偏位抓放、定量移动、遮挡、已登记障碍、失联和夹持反馈等仿真用例 | 独立读取木块位置、接触、关节和末端位移，核对回执 |
| 完整语音链路 | 4 段合成录音连续执行；同一误听文本在 A、C 初态配对复测 | 原始转写是否未经修改、真实模型 JSON 是否正确、物理任务是否完成 |

评价时区分三个结果：**模型提出的动作是否正确、控制器是否接受该动作、物理任务是否完成**。拒绝错误动作属于控制保护有效，不计为抓放成功；应当追问或忽略的输入保持静止，可计为该测试目标通过。不同分母不合并为统一“成功率”。

AISHELL-1 样本来自发布者的训练划分。本项目按说话人划出 9 段开发样本和 15 段留出样本，后者不是官方测试集。VigiClaw 实拍图也属于发布者的训练数据，且项目早期已使用；这些结果不代表未知设备或新场景的泛化能力。真实原始音频、外部图像和模型权重由复现脚本下载到本地缓存，不随仓库重新分发。

数据来源：[AISHELL-1 官方页面](https://www.openslr.org/33/) · [24 段音频清单与哈希](../results/safety/asr_inventory.json) · [VigiClaw 固定版本](https://huggingface.co/datasets/pengxianglu/vigiclaw-yolo-dataset/tree/e2e9acfbfdf827776c06673ee4103c970ac0d29b) · [100 条输入及参考答案](../configs/planner100.json)

## 2. 语音识别：SenseVoice 与 Whisper

三个设置在同一台 Windows 电脑上处理相同的 24 段录音，原音合计 111.267 s。Whisper Small 使用 CPU INT8，比较 beam=1 与 beam=3；当前 SenseVoiceSmall 使用 CPU INT8、4 线程和中文数字规范化。Whisper 耗时来自此前运行，各设置各测一遍，不能据此分析速度波动。

| 指标 | Whisper beam=1 | Whisper beam=3 | SenseVoiceSmall |
| --- | ---: | ---: | ---: |
| 全部 24 段 CER，越低越好 | 20.51% | 17.70% | **10.67%** |
| 全部整句一致 | 5/24 | 6/24 | **11/24** |
| 组内留出 15 段 CER | 23.85% | 20.92% | **13.39%** |
| 组内留出整句一致 | 3/15 | 3/15 | **6/15** |
| 24 段转写总耗时，不含加载 | 57.033 s | 56.725 s | **3.841 s** |
| 方向事件正确 | 7/9 | 7/9 | 7/9 |
| 数值事件正确 | 12/12 | 12/12 | 12/12 |
| 单位事件正确 | 10/12 | 11/12 | **12/12** |
| 原有否定或停止标志保留 | 9/10 | 9/10 | **10/10** |

CER 在统一繁简、空白和标点后计算，数字书写不同仍可能计错；语义事件接受“一百九十”和“190”等价。SenseVoice 的模型加载另耗时 2.072 s。上表关键词按原文出现次数统计，不能与按句计分混用。[逐句输出与计时](../results/safety/sensevoice/speech.json) · [关键词逐项复核](../results/safety/sensevoice/keyword_review.json)

SenseVoice 修复了“一百九十厘米”被旧模型写成“190 米”的单位错误，但“右转弯”仍成为“又转弯”。合成搬运口令中的“右边”被识别为“右干”。另外，“微整形”被新增识别为“未整形”，说明原有否定词全部保留，也不等于没有引入新否定。当前瓶颈是方向和语义误听；发令前仍需展示并核对转写。

这批公开朗读不含 A/B/C 工位、夹爪术语和现场麦克风噪声；4 条合成口令单独报告。两种语音运行环境曾出现 PyAV 接口、Windows ONNX Runtime DLL 冲突，已通过锁定依赖及明确加载当前环境 DLL 修复，失败记录保留。[Whisper 解码故障](../results/safety/asr_first_audio_decode_error.json) · [SenseVoice 加载故障](../results/safety/sensevoice/load_failure.json)

## 3. 文本规划：100 条输入与提示词对照

100 条输入包含 70 条动作、20 条追问和 10 条本地控制口令。留出集为 42 条动作、12 条追问和 6 条本地口令，云端比较的分母因此是 **54**。文本、初态、参考答案和分组先冻结，在开发集比较四版提示词后锁定终选版，再进行留出配对测试。

提示词从原版 554 字符变为终选版 519 字符。主要调整是保留完整 `steps` 示例、明确 `transfer` 已包含抓取与放置、说明观测坐标不是移动指令，以及补全单位和方向规则。第一次精简曾降低正确率，最终选择依据开发结果，而非提示词越短越好。[现行提示词](../configs/planner100_finalist.txt) · [版本选择记录](../results/safety/planner100/selection.json)

| 相同留出输入上的结果 | 原版 | 终选版 |
| --- | ---: | ---: |
| 协议结构、字段及数值范围合格 | 38/54 | **49/54** |
| 动作 JSON 与参考答案严格一致 | 15/42 | **27/42** |
| 动作语义正确，接受等价的抓取＋放置 | 24/42 | **27/42** |
| 应追问时输出追问 | 4/12 | **9/12** |
| 追问能够消除关键障碍，逐句复核 | 4/12 | **7/12** |
| 模型提出的错误物理动作 | 17 | **11** |
| 控制校验放行的错误物理动作 | 2 | **0** |
| 预选 12 条任务的仿真目标通过 | 5/12 | **10/12** |

严格一致按解析后的 JSON 字段和值判断，包含外层协议、动作顺序、起讫工位、坐标轴、符号及量值。问句焦点的自动评分与逐句复核分别保存。12 条仿真子集在查看留出回答前确定，含 9 个动作与 3 个应追问的任务；终选版通过的 10 条为 7 次动作和 3 次追问。[逐题原始回复及评分](../results/safety/planner100/by_case.json) · [控制校验重放](../results/safety/planner100/guard_replay.json) · [仿真重放](../results/safety/planner100/simulation_finalist.json)

主要失败仍包括方向反转、应追问时直接运动，以及本地校验拒绝正确同义表达。例如“向右平移两点五厘米”被终选版生成 `y=+25`，控制器拒绝；“X 正方向”等正确计划也曾被过窄的本地语言规则拒绝。此次 11 个错误动作全部被拦截仅适用于这批留出输入，不能推断所有模型错误均可拦截。

该实验累计尝试 250 次，其中 248 次取得正常结束的回复，2 次读取超时；原版和终选版对 54 条留出规划输入均有结果。早期的 4 段 Whisper 合成口令试验另保留在 [历史提示词记录](archive/prompt_trial.md)，不混入此处的 100 条统计。

## 4. 视觉感知：检测、工位与位置误差

当前 OpenCV 实现按 HSV 红色阈值提取连通域，不包含物体类别模型。公开图像没有本项目台面标定，评价预测框与发布者红块标注的 IoU，阈值为 0.5；同图多余候选计为假阳性。

| 图像数据 | 正确框 / 误框 / 漏检 | 唯一且正确的目标 | 主要问题 |
| --- | ---: | ---: | --- |
| 实拍 100 张，全部标有红块 | 95 / 187 / 5 | **67/100** | 32 张有多个候选；1 张只有一个候选但位置不合格 |
| 外部仿真 187 张，标有红块 | 176 / 58 / 11 | **129/187** | 红色干扰物、分割及定位错误 |
| 外部仿真 77 张，无红块 | 0 / 57 / 0 | 不适用 | **57/77** 张仍输出一个错误候选 |

实拍按框精确率为 33.69%、召回率为 95.00%。95 是检出红块的张数，正样本总数是 100。外部仿真中的苹果、红碗、红盘可满足同一颜色阈值；“只有一个候选”不能单独证明物体身份。[逐图预测、标注及统计](../results/safety/vision100.json)

本项目另生成 100 个双视角摆位，覆盖正常、偏位、工位边缘、同色/异色干扰、遮挡、曝光和反光。其中 72 个含唯一且完整处于工位的目标：固定相机正确确认 **42/72**，腕部相机 **56/72**。固定相机正确确认的 42 例中，台面 XY 误差中位数为 **0.828 mm**、P95 为 **1.897 mm**。这些是仿真中可确认样本的条件统计，不是真机定位精度。

进一步在 144 个独立边界摆位比较抓取准入半径。视觉门槛由 8 mm 收紧到 7 mm 后，72 个物理越界样本中的误放从 **15 降至 2**，72 个物理合格样本中的正确放行从 **56 降至 26**。误放下降伴随明显拒真，生产阈值未因此替换。面积和长宽比过滤也曾让多候选变成错误的唯一候选，未接入控制链。[形状过滤对照](../results/safety/vision_ablation.json)

视觉是当前最明显的泛化瓶颈。下一步应先验证工作区、目标尺寸和现场标定，再考虑身份标记或类别检测。ArUco 贴纸保留为备选；本页没有多颜色或通用物体识别成绩。

## 5. 运动控制、反馈与异常处理

| 验证任务 | 已有结果 | 适用范围 |
| --- | --- | --- |
| 偏位抓放对照 | 同一 63 种摆位各运行固定抓点与视觉对准，共 126 次。两版各有 24 种保持原位；实际尝试的 39 种中落位从 **36/39 提升至 38/39**，按全部摆位为 **38/63**；两版均未误报成功 | 当前物块、摩擦和工位；剩余失败出现于握持中丢失指尖接触 |
| 定量平移与转腕 | 18 次平移误差中位 **0.245 mm**、最大 **0.272 mm**；6 次转腕角误差约 **0.006°**；24/24 通过软件门槛 | 仿真门槛为 5 mm、2°，需真机独立量测 |
| 路径及持物碰撞预检 | 加登记支架时，端点检查放行 25 条，整段检查放行 13 条；持物挡针测试在搬运前拒绝，关节不动 | 覆盖已登记静态障碍与近似几何，不覆盖人员、线缆或未知移动障碍 |
| 双视角结果确认 | 目标位遮挡时固定视角不能确认，启用腕部后 3/3 帧确认；双视角均遮挡时保持“未确认” | 只能使用当前画面可提供的证据，不能以已发出放置指令代替结果观察 |
| 断图与关页 | 测试期限为 1.2 s 时，关页后 1.27 s 收到锁止回执；断图期间不续期，未执行松爪，重放旧请求不运动 | 本机 HTTP/浏览器与仿真测试；实际默认期限为 3 s，尚非真机制动时延 |
| 多场景验收与软件回归 | 28/28 场景符合各自预期，含 15 成功、7 追问、3 未确认、2 拒绝、1 已在目标；最新独立 IK 回归 **106 项通过** | 场景通过包含正确拒绝及未确认，不能写为 28 次抓取成功 |

证据：[偏位与精度对照](../results/safety/control_benchmark.json) · [路径检查](../results/safety/route_ablation.json) · [断连验证](../results/safety/link_loss.json) · [28 项场景](../results/safety/simulation/report.json) · [106 项回归](../results/safety/sensevoice/verification_live.json)

夹爪测试采用接触后停止继续位置闭合、有界握持与低高度试抬。0.30 N·m 是仿真中测试通过的握持参数，不可直接作为真机设定。公开 SO-101 海绵抓放数据包含 128 段、74,887 帧位置记录；全部都存在“开度低且稳定”的窗口，其中 123 段从首帧开始就满足，说明单靠位置稳定不足以证明已抓住物体。该数据未提供本项目需要的双指接触与落位标签，因此不报告抓取成功率。[真实轨迹来源](https://huggingface.co/datasets/aswinkumar99/LeRobot-SO101-Pick-Place/tree/6f1ceec7b23629d9fcdaf819a7d312f019172977) · [位置证据回放](../results/safety/real_jaw_evidence.json)

## 6. 完整链路与“右干”纠偏

四条已有合成录音依次经真实 SenseVoice、正常 HTTP 接口、规划与本地分流、MuJoCo、相机复核和文字回执，期间不修改转写、不重置场地。HTTP 客户端在进程内运行；未测跨网络传输或在线 TTS。

| 原始识别文本 | 规划或分流 | 独立验证结果 |
| --- | --- | --- |
| 把红色木块放到右干的工位。 | DeepSeek 生成 A→B 搬运 JSON | 双指夹持、抬起超过 20 mm；放开后三次采样均完整位于 B 且仅接触桌面 |
| 向左移动12毫米。 | DeepSeek 生成 `move(y,12)` | 位移约 [−0.046,12.001,−0.221] mm，误差 **0.226 mm** |
| 不要向前移动1厘米。 | 本地忽略 | 关节和仿真时间不变 |
| 向前移动5。 | 本地追问单位 | 关节和仿真时间不变 |

完整链路目标通过 **4/4**，其中 2 次运动、2 次原位响应；实际云端请求 2 次。随后固定提示词，在 A、C 两种初态各比较原始“右干”转写与正确“右边”原文，4 次均严格生成正确 JSON，并完成物理搬运。本轮合计使用 **6 次交大调用**，无重试。[主链路原始记录](../results/safety/sensevoice/flow_live.json) · [原始 JSON 与物理状态审计](../results/safety/sensevoice/live_audit.json) · [配对实验](../results/safety/sensevoice/pairs_live.json)

同一“右干”表述在主链路及两次配对中共成功 3/3 次，但后两次复用同一份转写，只有一个误听音频案例。模型使用现有“A 左、B 右、C 前中”定义，没有添加“右干”特例。本地规则无法独立确认这个词对应 B，正确性由预定实验目标核验；方向被误听为另一个合法方向时，JSON 和路径检查仍不足以恢复原意。

在线运动结果使用本机课程 IK。公开仓库默认使用独立 IK，随后离线重放上述 4 份真实回复，**4/4 通过相同物理检查**，无新增模型调用。两套后端的具体位移误差应分别记录。[独立 IK 重放](../results/safety/sensevoice/pairs_standalone.json) 此前本地规则处理“右干”需追问，明确补答 B 后完成搬运，原始失败及恢复过程仍可查阅。[本地规则记录](../results/safety/sensevoice/flow_local.json) · [补答恢复记录](../results/safety/sensevoice/recovery_fixed.json)

## 7. 复现命令

以下命令均在项目根目录的 PowerShell 执行，使用 README 中同一个 `.venv`。新结果写入 `outputs/reproduce/`，该目录不提交。公开版显式使用独立 IK；历史课程 IK 的逐项数值以原始记录为准。没有新模型调用的步骤可先运行；本地模型推理及新增云端请求均须取得相应授权。

### 7.1 不调用模型：回归、评分与仿真重放

```powershell
$env:VOICE_ARM_KINEMATICS = 'standalone'
uv run --frozen --extra voice --extra data python -m unittest discover -s tests -v
uv run --frozen --extra voice --extra data python -m arm_lab evaluate --out outputs/reproduce/scenarios
uv run --frozen --extra voice --extra data python -m arm_lab.control_benchmark --out outputs/reproduce/control
uv run --frozen --extra voice --extra data python -m scripts.replay_sensevoice_pairs --out outputs/reproduce/sensevoice_pairs.json
uv run --frozen --extra voice --extra data python -m scripts.compare_asr_passes --out outputs/reproduce/whisper_comparison.json
```

下面从已保存的校方原始回复复算 100 条输入的成绩和控制校验，再重放终选版的 12 个预定仿真任务。只复制评分输入，保留仓库原始结果。

```powershell
New-Item -ItemType Directory -Force outputs/reproduce/planner100 | Out-Null
Copy-Item results/safety/planner100/responses.json,results/safety/planner100/local.json outputs/reproduce/planner100/
uv run --frozen --extra voice --extra data python -c "from pathlib import Path; from scripts.evaluate_planner100 import dataset; from scripts.score_planner100 import report; report(dataset(), Path('outputs/reproduce/planner100'))"
uv run --frozen --extra voice --extra data python -m scripts.replay_planner100_guard --directory outputs/reproduce/planner100
uv run --frozen --extra voice --extra data python -c "from pathlib import Path; from scripts.replay_planner100_sim import run; run('finalist', Path('outputs/reproduce/planner100/simulation_finalist.json'), False)"
```

### 7.2 公开图像与真实位置数据

首次视觉复现加 `--download` 下载固定版本数据；已有缓存时去掉该选项即可离线重算。图像处理不调用模型。

```powershell
uv run --frozen --extra voice --extra data python -m scripts.evaluate_vision100 --download --out outputs/reproduce/vision100.json
uv run --frozen --extra voice --extra data python -m arm_lab evaluate-real --out outputs/reproduce/public_data.json
uv run --frozen --extra voice --extra data python -m arm_lab.jaw_evidence_replay --out outputs/reproduce/jaw_evidence.json
```

`evaluate-real` 首次准备实拍图和 SO-101 数值轨迹，已有缓存时添加 `--offline`。`jaw_evidence_replay` 读取前一步的轨迹缓存，不下载视频，也不连接机械臂。

### 7.3 经授权运行本地语音模型

准备 AISHELL-1 缓存不调用模型。下一条 SenseVoice 命令会对 **24 段真人音频＋4 段合成音频**运行本地推理，`--download-model` 允许准备固定版本权重；已有权重可去掉该参数。

```powershell
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark prepare --out outputs/reproduce/asr_inventory.json
uv run --frozen --extra voice --extra data python -m scripts.evaluate_sensevoice --approve-local-asr --download-model --out outputs/reproduce/sensevoice_speech.json
```

如需重新测量历史 Whisper 两种设置，而非仅复算已有结果，可在另行批准 48 段本地推理后执行：

```powershell
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark run --beam 1 --approve-local-asr --download-model --out outputs/reproduce/whisper_beam1.json
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark run --beam 3 --approve-local-asr --offline --out outputs/reproduce/whisper_beam3.json
```

### 7.4 经授权运行完整语音与在线规划链路

先准备 SenseVoice 权重及本机 `.env`。下面第一条执行 4 次本地转写；两条合计的交大尝试上限为 8，失败也计数。配对脚本从 `--source` 扣除主链路已发生的尝试，复用转写，不再调用 ASR。不要把两个参数理解为两份独立的 8 次预算。新增实验应使用新的输出路径和调用授权。

```powershell
uv run --frozen --extra voice --extra data --env-file .env python -m scripts.evaluate_sensevoice_flow --approve-local-asr --approved-campus-attempts 8 --out outputs/reproduce/flow_live.json
uv run --frozen --extra voice --extra data --env-file .env python -m scripts.evaluate_sensevoice_pairs --source outputs/reproduce/flow_live.json --approved-campus-attempts 8 --out outputs/reproduce/pairs_live.json
```

100 条输入的历史模型回复足以复核本页评分。重新开展提示词比较需要重新冻结方案、准备独立调用日志并获得配额授权，不应直接将本次留出题用于迭代后再宣称同一留出成绩。[历史比较方法与调用记录](archive/planner100_method.md)

## 8. 结论与后续优先级

当前证据支持限定场景下的语音规划、仿真执行和反馈闭环。语音速度已有改善，主要风险集中在方向误听、规划符号错误、颜色检测缺乏物体身份信息，以及仿真夹持证据与真实电机反馈之间的差异。

本次保留红色规则木块、固定工位、有限动作集和失败后停止/澄清；不扩展多颜色多物体识别、任意三维抓取、失败后自动重抓或实时连续语音控制。后续首先完成真机标定、夹持反馈和独立位移测量，其次补现场方向/单位/否定最小对及无目标视觉负样本，再依据新数据决定视觉替换与控制范围。任务边界及颜色扩展说明见 [README](../README.md)，现场实验顺序见 [真机接入与验收](hardware.md)。
