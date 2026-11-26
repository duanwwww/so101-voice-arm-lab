# SO-101 桌面机械臂语音遥操作

操作者一边看远程画面，一边说“把 A 的红木块放到 B”，也可以说“向右平移两厘米”。Whisper Small 把语音转成可校对的文字，OpenCV 提供可见工位，交大 DeepSeek 将两者转成控制 JSON。控制电脑校验动作，分段执行抓取、试抬和放置，再用画面复核落点。等待澄清或暂停期间保留当前场地。当前执行后端是 MuJoCo，实机记录为 0。

[工作流与各节点成绩](docs/workflow.md) · [真机改造、场地和 90 分钟预约方案](docs/hardware_handoff.md) · [3 分 14 秒仿真演示](results/safety/simulation_demo.mp4)

![从录音、相机观测到动作执行与回执的工作流](docs/workflow.png)

### 组员先看什么

| 想了解 | 已有证据与主要限制 |
| --- | --- |
| 口令是否听清 | 同一批 24 段公开真人普通话两遍测试，beam=3 字符错误率 17.70%。这是通用普通话语料，另有 4 条合成机械臂口令的小规模验证。[语音实验](docs/asr_benchmark.md) · [口令纠错](docs/prompt_trial.md) |
| JSON 能否控制正确 | 100 条输入先冻结金标。在相同的 42 条留出动作题中，严格匹配由 15 条增至 27 条。终选版仍会把向右的 −25 mm 写成 +25 mm。[原始回复与失败分析](docs/planner100_results.md) |
| 画面是否指对物体 | 100 张有红块的实拍图，67 张得到唯一正确候选。77 张无红块的外部仿真图，57 张仍报出一个候选。[视觉验证与数据来源](docs/vision100.md) |
| 机械臂是否做到 | 预选 12 个任务经真实模型回复和仿真物理重放，终选版 10 个达标，其中 7 次运动达标、3 次按要求追问且保持静止。两种本地表述误拦了正确动作。[逐例回放](results/safety/planner100/simulation_finalist.json) |

新组员可以先运行下面的离线命令，复现一次 A 到 B 的搬运。需要查看相机和语音交互时再启动网页。[课程模型出处](vendor/so101_course/NOTICE.md) · [项目贡献与材料来源](docs/project_summary.md)

## Windows 启动

[3 分 14 秒本机仿真演示](results/safety/simulation_demo.mp4) · [固定相机和腕部相机现场取证](docs/field_cameras.md) · [24 段真实普通话的两遍 Whisper 实测、错误举例与复算](docs/asr_benchmark.md)

网页在当前状态和所选相机画面可更新时才允许发令。指令执行期间关页或断图，默认超过 3 秒未续期就锁住仿真控制。恢复网络后先核对回执，手动重置仿真才能发起新动作。[接口说明](docs/deployment.md)

任务进行中可以继续录音。说“不要向前移动一厘米”仅忽略这一句，说“暂停当前任务”会中止剩余步骤并保留现场。“恢复控制”后可发新任务；在忙时发送的新动作会保留在网页草稿中。可在原位查看、确认后再执行。[交互用法、真实模型纠错及提示词对照](docs/prompt_trial.md)

~~~powershell
git clone https://github.com/duanwwww/so101-voice-arm-lab.git
cd so101-voice-arm-lab
uv sync --frozen --extra voice --extra data
uv run --frozen --extra voice --extra data python -m arm_lab run --layout A "把 A 区的木块搬到 B 区"
# 首次克隆后创建配置，本机已有 .env 时保留现有配置
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
uv run --frozen --extra voice --env-file .env python -m arm_lab serve
~~~

服务默认在控制电脑的 [localhost:8768](http://localhost:8768/) 打开。端口被占用时在启动命令末尾加 `--port 8770`。前台服务按 Ctrl+C 退出。私有仓库成员需要先用各自获授权的 GitHub 账号克隆。

## 使用控制台

1. 在“仿真场景”中选一个摆位，点击“重置仿真”。右侧可以输入自己的任务，也可以点“填入场景任务”再编辑。选择“本地规则 · 离线”即可复现各场景的示例。语音输入支持录音和上传短音频，经 Whisper 转写后可在发出前修改。
2. 默认规划方式为“交大 DeepSeek · 自然语言”。输入“把前面的木块夹起来，再放到右边的工位”，点“执行指令”。本机授权窗口列出本次传输的内容和接收方。模型可返回一到四步动作，或直接追问。
3. 如果模型问要放到哪里，可以在同一输入框回复“右边那个”。系统将上一轮口令、追问、这一句补充和最新观测一起送入规划。点“另起任务”会结束本轮澄清。
4. 每步结束后读取实际结果。释放后退开夹爪，在一个仿真秒内复看三次。勾选“执行后用 DeepSeek 解读并用 Edge-TTS 播报”即可依次进行回执解读和语音合成。也可单独使用回执下方的两个按钮，分别授权。

| 可以自己输入的任务 | 对应动作 |
| --- | --- |
| 向前抓取木块 | 摄像头定位木块后抓取和提升，双侧指尖接触通过后保持夹持。下一句可说“放到 B 区”。 |
| 把 A 区的木块搬到 B，接着把手腕向上转三度 | 校验工位和关节限制，按序搬运、观察、转腕并测量角度。 |
| 先向左挪十二毫米，再向前移动一厘米 | 以底座为参考执行两段 IK 平移，分别给出测量值。 |
| 把A区的红快放到C | 离线规则也能识别这一同音误字。在 A 时执行搬运。在 C 时说明你指定的起点和画面不一致。 |
| 桌上的木块在哪里 | 读取相机观测，返回工位或夹持状态。 |

“本地规则 · 离线”支持抓放、观察、平移、转腕及组合指令。控制台有 14 个可选场景，包括“放料位被遮挡”和“两个相机均无法复核”。从监控画面的菜单切换视角。运动时可以点“停止仿真运动”，锁止后须查看回执并手动重置仿真。当前使用一个 24 mm 红木块、三个限定工位和固定底座，每步平移上限 50 mm，腕部俯仰上限 15°。腕部复核和网页停止均已通过仿真浏览器测试。

## 模型配置与调用批准

| 模块 | 运行位置与输入 |
| --- | --- |
| Whisper Small | 控制电脑 CPU INT8。使用 Systran/faster-whisper-small。在本机网页首次批准转写时可下载权重到 outputs/models/faster-whisper-small，以后复用。音频留在控制电脑。 |
| 交大 DeepSeek | 上海交大“致远一号”，模型 deepseek-chat。规划输入为原话、可见物体编号和工位、台面标定得到的 XY 毫米坐标，以及适用的持物状态和追问。反馈输入为已执行的步骤、实际状态、回执文本及最终工位。每次最多输出 160 tokens。[校方接入指南](https://apps.hpc.sjtu.edu.cn/posts/claw-guide-sjtu-api) |
| OpenCV、IK 和 MuJoCo | 在控制电脑运行。视觉分割 RGB 图像，位置 IK 生成平移路径。工位抓放通过位置伺服、自由木块和接触摩擦实现。 |
| Edge-TTS | 微软在线合成普通话音色 zh-CN-XiaoxiaoNeural，只接收操作者批准播报的文字。 |

在本机 .env 中设置 SJTU_DEEPSEEK_API_KEY，或者用 SJTU_DEEPSEEK_ENV_FILE 指向已有课程配置。程序核对该文件中的 SJTU_API_BASE_URL 与学校官方地址，再读取 SJTU_API_KEY。校外按校方说明连接交大 VPN。

在 127.0.0.1 或 localhost 打开的本机控制台可逐次批准。规划、回执解读、转写、播报分别计数，取消确认不会触发调用。远程浏览器使用控制电脑在交互终端预先批准的次数。局域网麦克风需要受浏览器信任的 HTTPS，也可上传音频。启动远程服务的例子如下。

~~~powershell
uv run --frozen --extra voice --env-file .env python -m arm_lab serve --host 0.0.0.0 --port 8769 --deepseek-budget 4 --tts-budget 2
# 使用已经下载的 Whisper 权重时
uv run --frozen --extra voice --env-file .env python -m arm_lab serve --port 8769 --asr-budget 2 --whisper-dir "D:\models\faster-whisper-small"
~~~

## 复现实验

本轮新增 100 条自建中文输入和控制 JSON 参考答案。留出动作题中，原版与终选版分别在 15/42、27/42 条上精确生成了金标 JSON；预选的 12 条 MuJoCo 任务分别有 5 条、10 条达到物理或追问目标。[完整实验报告](docs/planner100_results.md) · [100 条输入、金标和实际回复](results/safety/planner100/by_case.json) · [提示词对照方法](docs/planner100_method.md) · [OpenCV 校验结果](docs/vision100.md)

新版本支持已登记障碍 JSON，坐标按底座系、单位为米。示例文件是仿真支架。下列命令分别复跑对照实验、验证载有支架时拒绝 A 路线，以及离线校验已缓存的 24 段普通话录音。语音识别批准和专用复跑命令见[语音实验](docs/asr_benchmark.md)。

~~~powershell
uv run --frozen --extra voice --extra data python -m arm_lab.control_benchmark --out outputs/control_benchmark
uv run --frozen --extra voice --extra data python -m arm_lab run --layout A --keepouts configs/keepouts.example.json "把木块放到 B 区"
uv run --frozen --extra voice --extra data python -m arm_lab.asr_benchmark prepare --offline
~~~

~~~powershell
uv run --frozen --extra voice python -m arm_lab run "先向左挪十二毫米，再把手腕向下转三度"
uv run --frozen --extra voice python -m arm_lab run --layout cover_goal
uv run --frozen --extra voice python -m arm_lab evaluate --out outputs/evaluation
uv run --frozen --extra voice python -m unittest discover -s tests -v
# 在独立端口启动测试服务器，使用已安装的 Edge。模型替身只在测试进程中注册。
uv run --frozen --extra voice --with playwright python tests/browser_diagram.py --out outputs/diagram-browser
# 录制一段真实仿真回合
uv run --frozen --extra voice python -m arm_lab record --layout B --out outputs/demo.mp4
# 两条固定的真实校方规划测试，在控制台批准后才发送
uv run --frozen --extra voice --env-file .env python -m arm_lab check-deepseek --out outputs/deepseek_pilot.json
~~~

公开数据评测使用额外的 Parquet 读取依赖。首次联网运行下载两位发布者的固定版本数据至 outputs/public-data。保留缓存后，可断网复现相同的逐图与逐回合结果。两条评测命令均无需 API Key。

~~~powershell
uv sync --frozen --extra voice --extra data
uv run --frozen --extra voice --extra data python -m arm_lab evaluate-real --out outputs/real_data.json
uv run --frozen --extra voice --extra data python -m arm_lab evaluate-real --offline --out outputs/real_data_repeat.json
~~~

- [八个新增场景与两套真实数据的结果及局限](docs/scenarios_and_real_data.md) · [场景图](results/scenarios/scene_gallery.jpg) · [27 项仿真记录](results/scenarios/report.json) · [真实数据评测记录](results/scenarios/real_data.json)
- [自定义指令的首轮仿真数据](results/custom_commands/report.json) · [浏览器录音到播报的测试记录](results/custom_commands/browser.json) · [控制台截图](results/custom_commands/diagram_console.png)
- [部署与模块通信](docs/deployment.md) · [实验结果与失败分析](docs/validation.md) · [相关文献与已有系统](docs/related_work.md)
- [课程资源出处与许可](vendor/so101_course/NOTICE.md) · [早期仿真录像](results/pick_B_sim.mp4)

实测环境为 Windows 11、uv、Python 3.12、MuJoCo 4.0.0 和 OpenCV 4.11。执行后端目前为 SO-101 仿真，尚未接入实物。预约真机后须首先核对从臂型号、串口、校准文件、工位坐标及夹持中心。
