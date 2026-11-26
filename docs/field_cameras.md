# 预约时保存固定相机和腕部相机的画面

模块 `arm_lab.field_cameras` 只读取操作者指定的本机摄像头或已有视频，保存 JPEG 与时间记录。它没有串口、模型或动作执行接口。没有腕部 USB 相机时只指定固定相机即可。实际从臂是否配有腕部相机、设备索引和腕部安装标定仍需现场核对。

## 采一回合

先由操作者在控制电脑确认 Windows 相机索引。以下 0 和 1 仅作例子，程序不扫描、不猜测设备，也不占用未指定的摄像头。第一条只采初始位置，第二条采松爪退开后的双视角。一次任务内部如果要持续观察，可选择 during。单次最多 30 组 JPEG、10 Hz、15 秒采样窗口，归档图最长边限制在 1280×720 的包围框内。

~~~powershell
uv run --frozen --extra voice --extra data python -m arm_lab.field_cameras --fixed-device 0 --trial 1 --phase before --samples 3 --fps 3
uv run --frozen --extra voice --extra data python -m arm_lab.field_cameras --fixed-device 0 --wrist-device 1 --trial 1 --phase after --samples 3 --fps 3
~~~

默认 Windows 后端 DirectShow；已确认设备索引但驱动无法打开时，可显式选择 `--backend msmf`。输出目录形如 `outputs/field-camera/01/after`。同一回合、同一阶段的已有记录会保留，重测可指定另一个 `--out` 目录。把前后两份 capture.json 的路径填进 15 回合登记表的 evidence 列，实物落点由现场观察者填写。[空表](../results/safety/field_trials_blank.csv)

已有两份录制时，从相同的**现场事件**确定起点，例如两机都拍到的同步标记。再任选下列参数，分别指定文件内的起始秒数。工具以视频的标称帧率选择最近的帧，可变帧率或不同时钟的录制需要人工核对播放器时间。单独回放固定录像时省略腕部参数。

~~~powershell
uv run --frozen --extra voice --extra data python -m arm_lab.field_cameras --fixed-video "D:/lab/fixed.avi" --wrist-video "D:/lab/wrist.avi" --start-s 8.0 --wrist-start-s 9.2 --trial 1 --phase after --samples 3 --fps 3 --out outputs/replay
~~~

## 怎样读报告

| 字段 | 含义 |
| --- | --- |
| trial、phase、start_utc、finish_utc | 登记表行号、与动作的先后关系、采集软件启动及结束的 UTC 时间。 |
| sources | 明确使用的设备索引及后端，或输入文件名、起点和标称 FPS。reported_size 为来源报告的宽高。 |
| samples.views | 每个采样时刻两幅图分别记录收到和保存的分辨率、相对文件路径、SHA-256、抓帧完成和解码完成的主机单调时钟。录像另有源帧号及后端报告的媒体时间。 |
| grab_skew_ms | 先抓取两个源，再各自解码保存。数值是同一台电脑两个抓帧调用结束时刻的差。摄像头驱动可能缓存图像；这个数不能证明两次曝光点的时间差，更不是远程网络延迟。 |
| counts、failure | 逐视角统计保存数和缺失采样数。失效理由区分录像到头、定位失败、抓帧失败、解码失败和赶不上下一次采样。录像以过高频率采样导致同一帧重复时另有 reused_video_samples。硬件在驱动内部丢了几帧无法由本接口测出。 |
| status | complete 表示每一组都收到指定的视角。partial 保存已有画面并由命令行返回 1。中断或设备错误时保留已有的报告和画面。 |

## 本机实际验证

使用同一次 MuJoCo 仿真产生的固定相机和腕部视角制作了两个 640×400、5 Hz 的六帧测试视频。从第 2 帧起，命令行连续读取两路各 4 帧，得到 4 对有效图；两次抓帧完成时刻的差最大为 1.221 ms。这是已经录好的本地视频在本次电脑上读出的数字。另一段回放取最后阶段的 3 对，均为不同源帧。还专门测了一长一短的测试视频，较短的视角在第 4 次采样正确记为 past_video_end，固定视角其余画面继续保存。上述验证均未打开真实摄像头。[4 对画面的回放报告](../results/safety/field_camera_replay/01/during/capture.json) · [3 对画面的回放报告](../results/safety/field_camera_replay/01/after/capture.json) · [自动化测试](../tests/test_field_cameras.py)
