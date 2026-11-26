# 公开普通话录音怎样用于机械臂课程

原计划先自录 36 条普通话。现在已有 **24 条真人录音**可检查语音识别把方向、数值、单位和否定词听成什么。数据来自希尔贝壳公开的 AISHELL-1，有参考转写，Apache-2.0 许可。下载与评测都在本机完成。原始录音保留在忽略提交的缓存目录，逐条文件编号、说话人、参考文本、时长和 SHA-256 见清单。[官方语料库](https://www.openslr.org/33/) · [发布者的固定版本](https://huggingface.co/datasets/AISHELL/AISHELL-1/tree/bbe295d530192a4cd41644b711c9aecd087df653) · [本组的 24 条清单](../results/safety/asr_inventory.json)

## 两遍实测结果

同一台 Windows 电脑（Intel Core Ultra 9 275HX），同一个 Whisper Small，先后对同一顺序的 24 段 WAV 各转写一次。只改变解码时保留的候选数量（beam）。音频共 111.267 秒。下表的“整句对”要求规范化后逐字相同。字符错误率（CER）越低越好。[两遍逐句对照与计时](../results/safety/asr_comparison.json) · [第一遍原始输出](../results/safety/asr_beam1.json) · [第二遍原始输出](../results/safety/asr_beam3.json)

| 样本 | beam=1 字符错误率 | beam=3 字符错误率 | 整句对，beam=1 → 3 | 解码总耗时，beam=1 → 3 |
| --- | ---: | ---: | ---: | ---: |
| 开发，2 人 9 段 | 13.68%（16/117） | 11.11%（13/117） | 2/9 → 3/9 | 21.056 → 21.350 秒 |
| 留出，3 人 15 段 | 23.85%（57/239） | 20.92%（50/239） | 3/15 → 3/15 | 35.977 → 35.375 秒 |
| 全部，5 人 24 段 | 20.51%（73/356） | 17.70%（63/356） | 5/24 → 6/24 | 57.033 → 56.725 秒 |

平均每段用时 2.376 秒和 2.364 秒。每秒原音分别需要约 0.513 秒和 0.510 秒处理。计时包括读取音频和生成完整转写，排除下载、加载权重及评分。两种设置各测一遍，顺序固定且可能受缓存和机器负载影响，0.308 秒的总差距不足以判断哪种设置通常更快。

### 哪些词会影响机械臂

先根据参考原文标出语义位置，再把两份转写逐项对照。下表按原文出现次数计分。同一句可以有两个方向或两个数量。数值接受“一百九十”和“190”等等价写法，单位另判。否定或停止标志检查是否保留“不、未、停止”等含义；最后一行再核对预先标出的短语，如“不相符”。[逐项语义判分、原文和实际转写](../results/safety/asr_keyword_review.json)

| 核对项 | beam=1 | beam=3 | 观察 |
| --- | ---: | ---: | --- |
| 运动方向清楚且正确 | 7/9 | 7/9 | “右转弯”两遍都成了“又转弯”；“向右偏”都成了“下右偏”，多了一个方向。另有 4 处左右脚，两遍均正确。 |
| 数值 | 12/12 | 12/12 | 包含 6 个长度和 6 个其他量值。 |
| 所有对应单位 | 10/12 | 11/12 | beam=3 将“八里米”纠正为“八厘米”。 |
| 其中长度的数值和单位都对 | 4/6 | 5/6 | 两遍都把“一百九十厘米”写成“190米”。数值对，换算后相差 100 倍。 |
| 否定或停止标志 | 9/10 | 9/10 | “未消毒”两遍都成了“位消毒”，否定消失。 |
| 预先标出的否定或停止短语完整 | 8/10 | 8/10 | “不相符”两遍都成了“不幸福”，虽有“不”，所指已经改变。 |

固定词表还可独立复算。含运动方向的 8 段中，两遍均有 6 段方向序列完整。含“不能／不要／停止”等预列词的 8 段均命中，扩展到上表的 10 处语义标注才显出漏掉的“未”。即使标志保留，后面的动作也可能写错，例如“不能饮水”的“饮”两遍都错。[固定词表逐句评分](../results/safety/asr_comparison.json)

**对控制流程的影响。** 本次优先推荐 beam=3 作为下一轮现场验证的设置。它在 6 段上减少了字符错误，其余 18 段的错误数持平，只修复了 1 处长度单位。网页仍需展示原始转写，让操作者在发出前核对方向、数值、单位与否定。平移只接受控制端支持的单位和每步 50 mm 的范围；听成“米”或缺方向时不能靠模型猜单位、猜方向。24 段公共朗读支持这一设计取舍。教室噪声、夹爪术语、指令规划和实际运动都需另行验证。

本次首次启动曾在读取首段 WAV 时遇到 faster-whisper 1.2.1 与 PyAV 19.0.2 的接口不兼容，没有生成转写。将语音环境的 PyAV 固定为 16.1.0 后，用同一份已校验权重离线跑完 24 + 24 段。CTranslate2 为 4.9.1，Python 为 3.12.13。失败记录单独保留。[首次失败记录](../results/safety/asr_first_audio_decode_error.json) · [音频解码回归测试](../tests/test_asr_benchmark.py)

## 选了什么，没覆盖什么

也核对了两种更接近控制口令的来源。HUTB 的机器人命令例程采用单一合成音色，适合验证软件链路。MagicHub 的中文命令语料含真人录音，但发布页要求联系数据方。本轮选择可以直接复现、有原音和参考转写的 AISHELL-1。[HUTB 实验说明](https://openhutb.github.io/ros2/humanoid/humanoid_kws_command/) · [MagicHub 发布页](https://magichub.com/datasets/mandarin-chinese-scripted-speech-corpus-command-and-query-keyword-spotting-sms/)

评测集在任何识别之前按发布者原文选择，涵盖“向左滑动”“向前”“七毫米”“不要”“停止”等词。在 24 条中，8 条带方向，6 条带数值和单位，6 条带运动词，8 条带否定或停止；同一条可以属于多个类别。单条时长 2.91 至 7.37 秒，音频合计 111.27 秒。经校验均为 16 kHz、单声道、16 位 PCM，24 个独立 WAV 共 3.56 MB。

五位说话人的文件在发布者的 train 目录下。本项目没有用它训练模型；Whisper 预训练材料与该语料是否有重叠，无法从本实验判断。项目把两个说话人的 9 条设为开发样本，另外三个说话人的 15 条设为最终报告样本。录音有体育、交通等上下文，同一说话人有相邻语段。这是一批针对关键词的抽样，比例不能外推成课堂口令的准确率。24 条原始参考文本经过离线规则，产生运动计划的次数为 0。本项只运行了文本解析。

原音没有 A/B/C 工位、夹爪闭合、持物后放置，也不包含本实验室的麦克风、网络和背景噪音。软件的文字组合测试可以覆盖语法和场景关联。预约时还值得顺带保存 3 到 4 条实际演示口令及人工转写，分别涉及搬运、厘米/毫米、转腕和否定；这些录音用于查现场链路和典型混淆，现有公开样本已承担基础转写测试。

另外调查了 Google FLEURS 普通话与 AISHELL-2。FLEURS 有正式测试划分，但所查普通话 dev/test 参考文本中“向左、向右、厘米、毫米”均为零，和本实验最关心的风险词错开。AISHELL-2 的发布者说明覆盖语音控制、智能家居和自动驾驶，这些领域可补充后续语料，本次使用的样本只来自 AISHELL-1。[Google FLEURS 论文与作者信息](https://research.google/pubs/fleurs-few-shot-learning-evaluation-of-universal-representations-of-speech/) · [FLEURS 普通话参考文本](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/cmn_hans_cn) · [AISHELL-2 官方介绍](https://aishelltech.com/aishell_2)

## 如何评分

数据准备阶段已离线复跑，清单逐字节一致。清单中的 `model_calls=0` 仅表示准备数据无需推理。只有标记为 `status=complete` 的对应模型报告才有可用的 CER 和关键内容识别结果。

第一轮使用网页相同的 CPU INT8、中文、beam=1、关闭 VAD。第二轮仅把搜索宽度改到 3，录音和模型完全一致。较宽的搜索可保留更多候选，也往往花费更多时间。两个参数在看本批识别结果之前确定。模型用 Systran/faster-whisper-small 的固定版本，程序在读取或下载时核对模型文件与词表哈希。[模型来源](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120)

- **字符错误率 CER** 按全组的插入、删除、替换次数除以参考字符总数。在 Windows 上先做 Unicode NFKC 和繁转简，再去空格、标点；保留字母、汉字和数字。数字写法不同仍会影响 CER。
- **关键内容逐句完全相同的数量** 单独比较顺序、方向词、运动词、否定/停止和“数值＋单位”。“十三毫米”和“13 mm”在此指标中相同，“十三厘米”会记错。新出现的反向词或多余数值也会造成该句不匹配。它测的是预先列出的词和数值，不能替代完整意图验收。
- **速度** 记录模型加载以后每条完整解码的墙钟时间。两轮分别记录，便于衡量更慢的解码是否换来了更少的关键听写错误。

现场口令是否执行，仍分别取决于用户核对转写、规划、执行器状态和实物结果。这个评测器只产生文本，无法对机械臂发指令。

## Windows 复现

在工程目录下执行。准备数据无需模型，首次下载 5 个指定说话人的归档（合计约 186 MB）及官方转写，核验 SHA-256 后只留下指定录音和文本索引。首次准备完毕即可脱网检查。

~~~powershell
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark prepare
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark prepare --offline
~~~

经操作者明确批准后，下面每一行各转写已选好的 24 条录音一次。首次命令允许从 Hugging Face 下载约 486 MB 固定本地模型文件。音频不离开电脑。第二次使用同一权重离线运行。若本机已准备好权重，第一行也可将 `--download-model` 替换为 `--offline`。默认写到 `results/safety/asr_beam1.json` 和 `asr_beam3.json`。已有报告时用 `--out` 指定新文件，避免覆盖既有测量。中断或失败时已尝试的次数仍记入报告。

~~~powershell
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark run --beam 1 --approve-local-asr --download-model
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark run --beam 3 --approve-local-asr --offline
~~~

已有本次结果时，直接执行下面的命令即可离线复算字符、词表和耗时指标。脚本核对每段的录音哈希、说话人、分组、原文和顺序，无需加载 Whisper。

~~~powershell
uv run --frozen --extra voice --extra data python -m scripts.compare_asr_passes
~~~
