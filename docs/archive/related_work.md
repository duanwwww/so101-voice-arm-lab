> 历史归档：本页保留阶段性设计与实验记录，部分默认设置已更新。当前说明以[项目首页](../../README.md)、[系统设计](../workflow.md)、[实验与复现](../experiments.md)及[真机接入](../hardware.md)为准。

# 相关工作与本次实验

本实验面向两三天的课程原型。首版用例要求把红块从 A 或 B 移到 C。当前原型支持 A/B/C 间的双向搬运、单独抓起后持物、平移、转腕和多步口令。用一台固定相机判断工位并估计台面 XY，在 SO-101 的 MuJoCo 场景验证执行反馈，预约真机时再测实物。[当前设计](deployment.md)

| 工作 | 做到哪一步 | 对本实验的取舍 |
| --- | --- | --- |
| ExtendAble，2026 年实机开源项目 | SO-101 从语音解析目标颜色，通过机载相机扫描，按十个距离分区调用记录的轨迹，把物体放到标记处。作者记录了单目深度和 IK 对准问题。[原项目](https://github.com/nathanqiuUCSB/ExtendAble) | 用两个取料工位和一个放料工位降低定位和示教工作量。目标所在工位仍须根据当次图像确定。 |
| sortbot，SO-101 开源项目 | 顶视和腕部图像进入视觉语言规划器，通过工具调用执行抓取、放置、移动和转动。合爪前对准检查可配置，所查配置默认关闭。[原项目](https://github.com/mr-mph/sortbot) · [配置](https://github.com/mr-mph/sortbot/blob/main/sortbot/config.yaml) | 以单相机颜色检测给出区域，程序校验动作参数。最终复看图像判定目标是否到达。 |
| Huang 等，Inner Monologue，CoRL 2022 | 在注入动作扰动的两项真机桌面任务上，表 2 中初始观测一次的平均成功率为 20%，逐步提供物体与动作成败反馈的方案为 90%。[论文表 2](https://arxiv.org/html/2207.05608#S4.T2) | 复用执行后重看场景的机制。论文数字仅对应原作者的任务与平台。 |
| Cui 等，LILAC，HRI 2023 | 在 Franka Panda 上研究语言在线纠正如何调整摇杆控制空间，并开展了 12 人实验。[作者论文](https://nlp.stanford.edu/pubs/cui2023lilac.pdf) | 把纠正的表达限制在明确的距离和关节转角，可逐条评估方向与数值。 |

## 数据与课程贡献

本组使用两套公开真实数据做了外部评测。VigiClaw 的 real 子集提供四种颜色的方块标注，用于检查红块识别在不同相机和背景下的误检、漏检。Aswinkumar 的 SO-101 海绵抓放演示提供逐帧夹爪指令和位置反馈，用于验证以固定阈值识别开合顺序时能覆盖多少回合。两者的采集者与本组分开署名，公开样本仅缓存用于回放。[VigiClaw 数据卡](https://huggingface.co/datasets/pengxianglu/vigiclaw-yolo-dataset) · [SO-101 抓放数据卡](https://huggingface.co/datasets/aswinkumar99/LeRobot-SO101-Pick-Place) · [本组的实测方法与结果](scenarios_and_real_data.md)

LeRobot 的公开抓放数据提供 50 个回合、两路 30 FPS 视频和关节状态、动作。库名含 so101，元数据的 robot_type 标为 so100_follower。它适合参考日志结构。本工程的图像与回合由自己的仿真生成。[原始元数据](https://huggingface.co/datasets/lerobot/svla_so101_pickplace/raw/main/meta/info.json)

课程材料提供 SO-101 网格模型及位置 IK。这个求解器控制固定爪参考点的 XYZ，允许朝向变化。先前从模型零位分别请求六个方向的 50 mm 位移，均收敛，但夹爪朝向变化约 8°–13°。本实验把抓放工位与定量平移分开验证。实物“SO10”的具体型号、实际夹持中心和关节零位要在预约时核对。[上游模型](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)

本工程新增实验场景、单目工位判断、指令校验、模型调用批准机制、执行反馈和独立仿真测试。外部实机视频、论文百分比和公开数据均保留各自来源。
