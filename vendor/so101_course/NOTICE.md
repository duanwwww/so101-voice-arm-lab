# 课程资源说明

SO101 下两份 MuJoCo XML 和 13 个 STL 与 TheRobotStudio / SO-ARM100 固定提交 `141aea7bcac2744d92998e30268c413202a56f96` 的对应文件一致，保留了 CAD 来源注释与 [Apache 2.0 许可文本](UPSTREAM_LICENSE)。[上游模型与网格](https://github.com/TheRobotStudio/SO-ARM100/tree/141aea7bcac2744d92998e30268c413202a56f96/Simulation/SO101)

本项目运行时增加相机、工位、自由木块和指尖接触垫。原实验曾使用授课方提供的 `so101_core.py`；该文件及讲义按课程授权另行取得。GitHub 版本自带独立的仿真运动学实现。已有合法课件的组员如需对照历史实验，可把原文件放在本目录（历史版本的 SHA-256 为 `ff6caf4583c5556d26c14904dcf08d0047ca3b2286e6c9993b5da957fb71920d`），随后显式选择 `$env:VOICE_ARM_KINEMATICS='course'`。选择 `'standalone'` 可始终使用独立实现。[共享版运动学](../../arm_lab/kinematics.py)
