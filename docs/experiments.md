# 实验设计、结果与复现

本页汇总截至 2026 年 11 月 26 日的实验。当前网页采用本地 SenseVoiceSmall 和上海交大“致远一号”`deepseek-reasoner`；此前 `deepseek-chat` 的成绩保留为历史基线，不作为当前模型的成绩。项目使用固定底座、A/B/C 三个工位和一个边长 24 mm 的红色木块；实际执行后端为 MuJoCo，尚无本组真机实验结果。其他颜色、任意形状及多目标选择未纳入本次验收。

先按 [README](../README.md) 配置一个 uv 环境，再按本文复现。系统设计见 [工作流与技术实现](workflow.md)，实机验证计划见 [真机接入与验收](hardware.md)。原始输入、模型回复和失败记录保存在 `configs/`、`results/`；分阶段的详细记录归入 `docs/archive/`。

## 1. 数据与评价方法

| 实验 | 数据与任务 | 评价依据 |
| --- | --- | --- |
| 语音转写 | AISHELL-1 的 24 段真人普通话，5 位说话人；另有 4 段 Windows 离线合成口令 | 字符错误率（CER）、整句一致、方向/数值/单位/否定词、完整转写耗时 |
| 文本规划 | 历史 Chat：100 条中文输入，40 条开发、60 条留出；当前模型：从已有数据选取 22 题进行 Reasoner/Chat 配对诊断 | 原始 JSON 的格式、字段、顺序、坐标轴、符号与数值；追问是否针对缺失信息。22 题不是新的留出集 |
| 误听纠偏 | 16 条固定诊断文本：14 条人工扰动文本、1 条已有 ASR 误听、1 条已有缺单位转写 | 分别评价可恢复错误、应追问输入、信息已经丢失的合法文本；不冒充新音频测试 |
| 视觉感知 | VigiClaw 的 100 张实拍与 264 张外部仿真图；本项目 100 个双视角摆位、144 个范围边界摆位 | 发布者标注与预测框的 IoU；唯一目标识别；独立仿真物理位置与视觉估计的差异 |
| 运动与反馈 | 偏位抓放、定量移动、遮挡、已登记障碍、失联和夹持反馈等仿真用例 | 独立读取木块位置、接触、关节和末端位移，核对回执 |
| 完整语音链路 | 历史 Chat：4 段合成录音；当前 Reasoner：10 段合成录音的连续场景实验 | 分开评价原始转写、模型 JSON、控制回执及独立物理状态；合成录音不能代表真人识别准确率 |

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

## 3. 文本规划：当前 Reasoner 与历史 Chat

### 3.1 当前模型：22 题配对诊断

固定同一份 519 字符提示词，比较交大网关的 `deepseek-reasoner` 和 `deepseek-chat`。22 题由已有 100 题中的 20 题，以及同一份“右干”转写在 A、C 两种初态下的 2 题组成，共 16 个动作、6 个应追问任务。题目与旧结果已经查看过，**这是模型切换诊断，不是新留出成绩**；也没有重新转写音频或执行物理动作。

| 同一批 22 题 | Reasoner，当前模式 | Chat，对照模式 |
| --- | ---: | ---: |
| 协议结构、字段及数值范围合格 | 22/22 | 22/22 |
| 动作 JSON 严格正确 | **15/16** | 10/16 |
| 动作语义正确，接受等价表达 | **15/16** | 11/16 |
| 应追问时输出追问 | 6/6 | 6/6 |
| 追问焦点符合参考条件 | **6/6** | 5/6 |
| 模型提出的错误物理动作 | 1 | 3 |
| 控制校验重放放行的错误动作 | 0 | 0 |
| 正确动作通过控制校验 | **8/16** | 6/16 |
| 模型正确、但被本地校验拒绝 | 7/15 个正确动作 | 5/11 个正确动作 |
| 单次请求耗时中位数 | 14.382 s | 3.547 s |
| 单次请求耗时 P95 | 39.790 s | 9.023 s |

Reasoner 在 5 题上语义正确而 Chat 不正确，没有出现相反的配对；两者都将“往右平移半厘米”错误地生成 `y=+5`，随后被控制校验拒绝。“右干”的两个初态下，两者均生成正确 JSON。**格式全部合格仍不代表动作语义全部正确**。原始回复、逐题评分和校验结果见 [22 题配对记录](../results/safety/reasoner/paired.json)。

Reasoner 的 15 个正确动作中，只有 8 个通过现行控制校验，另 7 个因表达、单位或上下文解析过窄被拒绝，包括“往后退 2 毫米”“两度半”“X 正方向”和补答单位。校验没有放行已知错误动作，但明显损失了可用性。这是与模型正确率分开的瓶颈，不能把 15 个正确动作加 6 个正确追问，写成完整链路成功 21/22。这里仅重放校验，未运行运动仿真。[独立逐题复核与拒绝清单](../results/safety/reasoner/manual_review.json)

这轮发出 44 次请求，加上单独保存的 1 次连通性验证共 45 次，无自动重试。Reasoner 的 `max_tokens=4096`，Chat 为 160；Reasoner 的 22 次回复均含网关返回的推理字段，两个模式均无截断。使用不同预算是为了给推理模式留出空间，因此结果比较的是当前两种配置，不能称为等算力比较。Reasoner 在这批诊断中改善了语义正确率，代价是约 4 倍的中位请求耗时。

### 3.2 误听纠偏边界：16 条文本诊断

固定 [16 条输入及参考目标](../configs/reasoner_asr_noise.json)，不更改提示词。6 条可恢复的误写、6 条应追问输入和 4 条信息丢失后的合法指令分组统计。只向模型发送转写文本和场景；原始意图、参考答案、分组标签不发送。14 条是人工构造的文本扰动，另 2 条复用此前合成音频的真实 ASR 输出；本实验没有新增录音或 ASR 推理。

16 次请求全部完成，无重试、无截断，协议格式 **16/16 合格**；请求耗时中位数 12.917 s、P95 66.019 s。[原始回复、评分与校验记录](../results/safety/reasoner/noise.json)

| 诊断组 | 模型语义结果 | 控制校验与含义 |
| --- | --- | --- |
| 可恢复误写，6 条 | 原始动作 JSON 严格与语义正确均为 **5/6** | 正确动作实际通过校验 **4/6**；“十二豪米”被模型恢复为 12 mm，但本地单位校验拒绝 |
| 应追问输入，6 条 | 输出追问及自动焦点评分均为 **5/6**；逐句语义复核为 **4/6** | “向下移动一厘”被直接当作 1 cm 运动，控制校验拒绝；仍需追问残缺单位 |
| 信息丢失后的合法文本，4 条 | 符合当前转写 **3/4** | 这 3 条均背离隐藏的原始意图，且 **3/4 通过控制校验**；属于语义风险，未进行物理执行 |

这轮“右干”得到追问，未恢复目标；同一文本在 22 题配对中曾成功，说明单次成功不能证明稳定纠偏。应追问组的 N08 输入“两点毫米”丢失了小数位，模型问“2 毫米还是 2 厘米”，没有确认缺失的数值；因此逐句复核仅 4/6。这里保留自动评分 5/6，不以修订参考答案掩盖评分器的问题。逐句复核由 Codex 完成，原始回答供组员复核。[逐条语义复核](../results/safety/reasoner/manual_review.json)

信息丢失组分别模拟方向替换、否定词丢失、数字替换和单位替换。N14—N16 按当前文字生成正确且在限额内的动作，却违背原始意图，校验无法拦截。N13 的方向再次被模型翻转，偶然与隐藏原意相同，但与实际输入冲突，不能计为可靠纠偏。模型仅凭文本无法确定这些原意；转写核对、关键数值确认和保留暂停入口仍然必要。以上 16 条均为诊断，不是新的盲测，也不是 16 次语音识别或物理任务。

### 3.3 历史基线：Chat 的 100 条输入与提示词对照

以下结果使用旧的 `deepseek-chat`，并非 Reasoner 的 100 题成绩。100 条输入包含 70 条动作、20 条追问和 10 条本地控制口令。留出集为 42 条动作、12 条追问和 6 条本地口令，云端比较的分母因此是 **54**。文本、初态、参考答案和分组先冻结，在开发集比较四版提示词后锁定终选版，再进行留出配对测试。

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
| 多场景验收与软件回归 | 历史 28/28 场景符合各自预期，含 15 成功、7 追问、3 未确认、2 拒绝、1 已在目标；本轮独立 IK 软件回归 **129 项通过**，耗时 39.866 s | 场景通过包含正确拒绝及未确认，不能写为 28 次抓取成功；软件回归不调用真实模型 |

证据：[偏位与精度对照](../results/safety/control_benchmark.json) · [路径检查](../results/safety/route_ablation.json) · [断连验证](../results/safety/link_loss.json) · [28 项场景](../results/safety/simulation/report.json) · [本轮 129 项回归](../results/safety/reasoner/verification.json) · [历史 106 项回归](../results/safety/sensevoice/verification_live.json)

夹爪测试采用接触后停止继续位置闭合、有界握持与低高度试抬。0.30 N·m 是仿真中测试通过的握持参数，不可直接作为真机设定。公开 SO-101 海绵抓放数据包含 128 段、74,887 帧位置记录；全部都存在“开度低且稳定”的窗口，其中 123 段从首帧开始就满足，说明单靠位置稳定不足以证明已抓住物体。该数据未提供本项目需要的双指接触与落位标签，因此不报告抓取成功率。[真实轨迹来源](https://huggingface.co/datasets/aswinkumar99/LeRobot-SO101-Pick-Place/tree/6f1ceec7b23629d9fcdaf819a7d312f019172977) · [位置证据回放](../results/safety/real_jaw_evidence.json)

## 6. 完整链路与“右干”纠偏

### 6.1 当前 Reasoner：10 段音频连续场景

本轮使用 [10 段任务清单](../configs/reasoner_flow10.json)，依次覆盖 A→B 搬运、左移 12 mm、禁止动作、缺单位追问、补答 5 mm、B→C 搬运、平移后转腕、C→A 搬运、超范围追问，以及执行中暂停。全部是 Windows 离线合成语音，前 4 段复用既有录音，后 6 段新增合成；与 AISHELL-1 真人语音成绩分开。网页产品已经移除 TTS 播报，离线合成仅用于准备可复现的测试输入。

实际链路为 SenseVoice 原始转写 → 网页 HTTP 接口 → 本地停止/禁止/缺单位处理或 Reasoner → 控制校验 → MuJoCo → 相机复核与物理状态取证；期间不修改转写、不重置场地。暂停测试另有 1 条文字辅助运动，用于确保播放暂停音频时机械臂正在运动，不计为第 11 段音频。

本轮真实模型链路目标通过 **10/10**：6 个运动任务、3 个保持静止的响应、1 个运动中暂停。发生 10 次本地 ASR、8 次交大规划，其中 7 次来自主音频任务、1 次来自暂停测试的辅助文字任务；无重试、0 次重置。7 个主任务原始 JSON 均格式正确且语义正确，另 3 条由本地确定性处理，不能写为“模型 10/10”。[完整原始链路与物理证据](../results/safety/reasoner/flow10.json)

| 顺序与原始转写 | 规划或处理结果 | 独立物理验证 |
| --- | --- | --- |
| 1. 把红色木块放到右干的工位。 | Reasoner：A→B | 双指夹持试抬超过 20 mm；释放后连续 3 次位于 B，仅接触台面 |
| 2. 向左移动12毫米。 | Reasoner：`y=+12` | 位移误差 **0.194 mm** |
| 3. 不要向前移动1厘米。 | 本地忽略 | 关节、木块位置与仿真时间不变 |
| 4. 向前移动5。 | 本地追问单位 | 同上，保持原位 |
| 5. 5毫米。 | Reasoner 结合上文：`x=+5` | 位移误差 **0.198 mm**，未重置场地 |
| 6. 把红色木块从B区送到C区。 | Reasoner：B→C | 同一试抬及连续 3 次落位检查通过 |
| 7. 先向左移动5毫米，再把手腕向下转3度。 | Reasoner：平移后转腕 | 平移误差 **0.200 mm**；独立读取腕角 **3.006°** |
| 8. 把C区的红色木块放到左面的工位。 | Reasoner：C→A | 同一试抬及连续 3 次落位检查通过 |
| 9. 向前移动60厘米。 | Reasoner：追问超范围距离 | 保持原位 |
| 10. 暂停当前任务。 | 本地暂停正在执行的辅助运动 | 观察到实际位移后再识别暂停；运动中断、后续步骤未执行、现场和木块保留 |

以上使用公开版独立 IK。位移与转角从 MuJoCo 状态独立读取，不以执行回执中的“成功”直接判分；3 次搬运还检查双指接触、试抬高度、释放及连续落位。暂停只验证本机进程内 HTTP 和仿真，不代表远程网络或真机制动时延。

ASR 的首条接口耗时 2.807 s，含模型初次加载；其余 9 条为 0.108–0.243 s。自动关键字段对照虽为 10/10，但其中数条没有提取到可比较字段，且未覆盖“右边→右干”，因此不作为 100% 识别准确率。原始转写没有人工修正；“右干”在本次链路成功，在上一组文本诊断中却触发追问，应按不同运行分别报告。10 段均为合成语音，未评估新说话人或现场噪声。本轮连通性、配对、误听及语音链路合计 **69 次交大请求**（1＋44＋16＋8）。

### 6.2 历史 Chat：4 段音频及“右干”配对

以下为此前 `deepseek-chat` 的证据。四条已有合成录音依次经真实 SenseVoice、正常 HTTP 接口、规划与本地分流、MuJoCo、相机复核和文字回执，期间不修改转写、不重置场地。HTTP 客户端在进程内运行；未测跨网络传输或在线 TTS。

| 原始识别文本 | 规划或分流 | 独立验证结果 |
| --- | --- | --- |
| 把红色木块放到右干的工位。 | DeepSeek 生成 A→B 搬运 JSON | 双指夹持、抬起超过 20 mm；放开后三次采样均完整位于 B 且仅接触桌面 |
| 向左移动12毫米。 | DeepSeek 生成 `move(y,12)` | 位移约 [−0.046,12.001,−0.221] mm，误差 **0.226 mm** |
| 不要向前移动1厘米。 | 本地忽略 | 关节和仿真时间不变 |
| 向前移动5。 | 本地追问单位 | 关节和仿真时间不变 |

完整链路目标通过 **4/4**，其中 2 次运动、2 次原位响应；实际云端请求 2 次。随后固定提示词，在 A、C 两种初态各比较原始“右干”转写与正确“右边”原文，4 次均严格生成正确 JSON，并完成物理搬运。本轮合计使用 **6 次交大调用**，无重试。[主链路原始记录](../results/safety/sensevoice/flow_live.json) · [原始 JSON 与物理状态审计](../results/safety/sensevoice/live_audit.json) · [配对实验](../results/safety/sensevoice/pairs_live.json)

同一“右干”表述在主链路及两次配对中共成功 3/3 次，但后两次复用同一份转写，只有一个误听音频案例。模型使用现有“A 左、B 右、C 前中”定义，没有添加“右干”特例。本地规则无法独立确认这个词对应 B，正确性由预定实验目标核验；方向被误听为另一个合法方向时，JSON 和路径检查仍不足以恢复原意。

在线运动结果使用本机课程 IK。公开仓库默认使用独立 IK，随后离线重放上述 4 份真实回复，**4/4 通过相同物理检查**，无新增模型调用。两套后端的具体位移误差应分别记录。[独立 IK 重放](../results/safety/sensevoice/pairs_standalone.json) 此前本地规则处理“右干”需追问，明确补答 B 后完成搬运，原始失败及恢复过程仍可查阅。[本地规则记录](../results/safety/sensevoice/flow_local.json) · [补答恢复记录](../results/safety/sensevoice/recovery_fixed.json)

### 6.3 网页交互回归：模型替身，不计模型准确率

独立启动 Microsoft Edge 无界面浏览器及 MuJoCo，以内存替身提供 ASR 文本和规划 JSON，真实模型调用为 0。两组浏览器脚本通过，验证移除授权弹窗、规划器选择和 TTS 控件后，上传音频、停止、抓取与放置、遮挡确认、澄清回复仍可用。[界面与流程记录](../results/safety/reasoner/browser.json) · [最终界面截图](../results/safety/reasoner/console.png)

忙时交互测试上传 5 次音频，验证禁止指令和状态查询保留正在执行的任务，新的运动指令保留为草稿；暂停、恢复后草稿与场地均保留，审核草稿不会自动发送，明确提交后才执行。测试期间收到 14 次有效心跳、0 次场地重置、0 次看门狗超时。最后 4 mm 平移的仿真误差为 0.197 mm。这些成绩验证网页状态管理与执行链路，不验证 SenseVoice 或 Reasoner 的识别能力。[忙时语音交互记录](../results/safety/reasoner/browser_live_voice.json)

## 7. 复现命令

以下命令均在项目根目录的 PowerShell 执行，使用 README 中同一个 `.venv`。新结果写入 `outputs/reproduce/`，该目录不提交。公开版显式使用独立 IK；历史课程 IK 的逐项数值以原始记录为准。网页提交语音识别及规划请求直接执行，不再弹出逐次授权。批量实验用 `--live` 或 `--run` 明确启动；省略时仅检查实验输入，历史脚本的 `--approve-*` 参数保留为其命令行接口。

### 7.1 不调用模型：回归、评分与仿真重放

```powershell
$env:VOICE_ARM_KINEMATICS = 'standalone'
uv run --frozen --extra voice --extra data python -m unittest discover -s tests -v
uv run --frozen --extra voice --extra data python -m arm_lab evaluate --out outputs/reproduce/scenarios
uv run --frozen --extra voice --extra data python -m arm_lab.control_benchmark --out outputs/reproduce/control
uv run --frozen --extra voice --extra data python -m scripts.replay_sensevoice_pairs --out outputs/reproduce/sensevoice_pairs.json
uv run --frozen --extra voice --extra data python -m scripts.compare_asr_passes --out outputs/reproduce/whisper_comparison.json
uv run --frozen --extra voice --extra data --with playwright python tests/browser_diagram.py --out outputs/reproduce/browser
uv run --frozen --extra voice --extra data --with playwright python tests/browser_live_voice.py --out outputs/reproduce/browser_live_voice.json
```

浏览器脚本使用已安装的 Microsoft Edge、独立测试端口和内存模型替身；不连接真实规划 API。机器没有 Edge 时需先安装该浏览器。

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

### 7.3 本地语音模型

准备 AISHELL-1 缓存不调用模型。下一条 SenseVoice 命令会对 **24 段真人音频＋4 段合成音频**运行本地推理，`--download-model` 允许准备固定版本权重；已有权重可去掉该参数。

```powershell
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark prepare --out outputs/reproduce/asr_inventory.json
uv run --frozen --extra voice --extra data python -m scripts.evaluate_sensevoice --approve-local-asr --download-model --out outputs/reproduce/sensevoice_speech.json
```

如需重新测量历史 Whisper 两种设置，而非仅复算已有结果，下面两条合计运行 48 段本地推理：

```powershell
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark run --beam 1 --approve-local-asr --download-model --out outputs/reproduce/whisper_beam1.json
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark run --beam 3 --approve-local-asr --offline --out outputs/reproduce/whisper_beam3.json
```

### 7.4 当前 Reasoner：文本诊断与 10 段语音链路

先检查输入和实验计划；以下三条不调用模型。10 段合成录音已随实验资源保存，检查会核对来源与文件哈希。

```powershell
uv run --frozen --extra voice --extra data python -m scripts.evaluate_reasoner --dry-run
uv run --frozen --extra voice --extra data python -m scripts.evaluate_reasoner_noise --dry-run
uv run --frozen --extra voice --extra data python -m scripts.evaluate_reasoner_flow10
```

下面从已保存回复离线重算两组文本评分。复制到新路径，避免覆盖发布证据；不请求模型，也不执行机械臂动作。

```powershell
New-Item -ItemType Directory -Force outputs/reproduce/reasoner | Out-Null
Copy-Item results/safety/reasoner/paired.json,results/safety/reasoner/noise.json outputs/reproduce/reasoner/
uv run --frozen --extra voice --extra data python -m scripts.evaluate_reasoner --score-only --out outputs/reproduce/reasoner/paired.json
uv run --frozen --extra voice --extra data python -m scripts.evaluate_reasoner_noise --score-only --out outputs/reproduce/reasoner/noise.json
```

准备好 SenseVoice 权重和本机 `.env` 后，使用以下命令重新运行真实模型实验。22 题比较有 44 个计划请求；16 题诊断有 16 个计划请求；10 段连续链路最多运行 10 次本地转写、12 次云端规划。三条应依次运行，避免校方限流；重试及中断均保留在结果文件中，不为提升成绩重复抽样。`--live`/`--run` 才会开始调用，不会弹出网页授权框。

```powershell
$env:VOICE_ARM_KINEMATICS = 'standalone'
uv run --frozen --extra voice --extra data --env-file .env python -m scripts.evaluate_reasoner --live --max-attempts 46 --out outputs/reproduce/reasoner_paired_live.json
uv run --frozen --extra voice --extra data --env-file .env python -m scripts.evaluate_reasoner_noise --live --max-attempts 17 --out outputs/reproduce/reasoner_noise_live.json
uv run --frozen --extra voice --extra data --env-file .env python -m scripts.evaluate_reasoner_flow10 --run --cloud-limit 12 --out outputs/reproduce/reasoner_flow10_live.json
```

Reasoner/Chat 比较及误听诊断支持 `--resume` 继续同一冻结计划，但不覆盖已取得的回复；10 段连续链路为保留真实场景顺序，不支持自动续跑或重试。重复运行须用新的输出文件。若需要重新合成后 6 段录音，可在副本配置中将这些音频路径改到新的项目子目录，再用 Windows PowerShell 执行 `scripts/prepare_reasoner_flow10_audio.ps1 -Config <副本配置路径>`；脚本拒绝覆盖已有录音。合成输入只服务于实验，网页没有 TTS 播报功能。

### 7.5 历史 Chat 链路复现

下面保留旧实验脚本的复现入口，用于与历史 `deepseek-chat` 结果对照。第一条执行 4 次本地转写；两条合计的交大尝试上限为 8，失败也计数。配对脚本从 `--source` 扣除主链路已发生的尝试，复用转写，不再调用 ASR。两个参数共用一份预算，不能理解为两份独立的 8 次额度。

```powershell
uv run --frozen --extra voice --extra data --env-file .env python -m scripts.evaluate_sensevoice_flow --approve-local-asr --approved-campus-attempts 8 --out outputs/reproduce/flow_live.json
uv run --frozen --extra voice --extra data --env-file .env python -m scripts.evaluate_sensevoice_pairs --source outputs/reproduce/flow_live.json --approved-campus-attempts 8 --out outputs/reproduce/pairs_live.json
```

100 条输入的历史模型回复足以复核本页评分。重新开展提示词比较需要重新冻结方案、准备独立调用日志并核对校方额度，不应直接将本次留出题用于迭代后再宣称同一留出成绩。[历史比较方法与调用记录](archive/planner100_method.md)

## 8. 结论与后续优先级

当前证据支持限定场景下的语音规划、仿真执行和反馈闭环。22 题诊断支持将网页规划切换为 Reasoner，但尚不足以证明未知指令下的泛化率。语音速度已有改善，主要风险集中在方向误听、合法但背离原意的转写、规划符号错误、颜色检测缺乏物体身份信息，以及仿真夹持证据与真实电机反馈之间的差异。Reasoner 的额外请求延迟和本地校验对正确动作的误拒也是当前瓶颈。

本次保留红色规则木块、固定工位、有限动作集和失败后停止/澄清；不扩展多颜色多物体识别、任意三维抓取、失败后自动重抓或实时连续语音控制。软件后续先修复数值、同义表达及澄清上下文校验，并建立独立回归集，保留原有距离与碰撞约束；不能靠放宽全部校验换取通过率。实机阶段优先完成标定、夹持反馈和独立位移测量，其次补现场方向/单位/否定最小对及无目标视觉负样本，再依据新数据决定视觉替换与控制范围。任务边界及颜色扩展说明见 [README](../README.md)，现场实验顺序见 [真机接入与验收](hardware.md)。
