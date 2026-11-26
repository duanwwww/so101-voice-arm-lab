import argparse
import json
from pathlib import Path
import sys

from .model_access import DEEPSEEK_ENDPOINT, DEEPSEEK_MODEL, PLAN_MAX_TOKENS, WEB_PLANNER_MODEL, ModelAccess, ModelDenied, ModelFailure
from .scene import LAYOUTS


def main():
    parser=argparse.ArgumentParser(description="SO-101 无实机指令实验")
    sub=parser.add_subparsers(dest="mode",required=True)
    check=sub.add_parser("evaluate",help="运行固定仿真用例并保存证据")
    check.add_argument("--out",type=Path,default=Path("outputs/evaluation"))
    real=sub.add_parser("evaluate-real",help="用固定版本的公开实拍图像与实机夹爪记录做离线回放评测")
    real.add_argument("--cache",type=Path,default=Path("outputs/public-data"))
    real.add_argument("--out",type=Path,default=Path("outputs/real_data.json"))
    real.add_argument("--offline",action="store_true",help="只读本机已缓存的数据文件")
    pilot=sub.add_parser("check-deepseek",help="在上海交大网关验证两条固定文字用例，缺少校方凭据时只写状态")
    pilot.add_argument("--out",type=Path,default=Path("outputs/deepseek_pilot.json"))
    one=sub.add_parser("run",help="执行单条文字指令")
    film=sub.add_parser("record",help="从实际仿真回合录制 H.264 视频")
    film.add_argument("--layout",choices=LAYOUTS,default="B")
    film.add_argument("--out",type=Path,default=Path("outputs/demo.mp4"))
    one.add_argument("text",nargs="?",help="省略时运行所选场景的示例任务")
    serve=sub.add_parser("serve",help="启动控制台与 HTTP 接口")
    serve.add_argument("--host",default="127.0.0.1")
    serve.add_argument("--port",type=int,default=8768)
    for p in (one,serve):
        p.add_argument("--keepouts",type=Path,help="按底座坐标系和米登记的固定障碍 JSON")
        p.add_argument("--layout",choices=LAYOUTS,default="A")
    one.add_argument("--deepseek-budget",type=int,default=0)
    serve.add_argument("--asr-engine",choices=("sensevoice","whisper"),help="默认 sensevoice；仅提供旧 --whisper-dir 时兼容选择 whisper")
    serve.add_argument("--sensevoice-dir",type=Path)
    serve.add_argument("--whisper-dir",type=Path)
    args=parser.parse_args()
    if args.mode=="evaluate-real":
        from .real_data import evaluate_real
        evaluate_real(args.out,args.cache,args.offline)
        return 0
    if args.mode=="check-deepseek":
        from .deepseek_pilot import run_pilot
        result=run_pilot(args.out)
        print(result['status'],f"{result['attempts']}/2",args.out)
        return 0 if result['status']=='passed' else 2
    if args.mode=="record":
        from .record import record
        receipt=record(args.layout,args.out)
        print(receipt['message'],args.out)
        return int(receipt['status']!='success')
    if args.mode=="evaluate":
        from .evaluate import evaluate
        report=evaluate(args.out)
        return int(report["passed"]!=report["total"])
    access=ModelAccess(planner_model=WEB_PLANNER_MODEL if args.mode=="serve" else DEEPSEEK_MODEL)
    if args.mode=="run" and args.deepseek_budget:
        access.authorize("deepseek",args.deepseek_budget,
                         f"上海交通大学 {DEEPSEEK_MODEL}（{DEEPSEEK_ENDPOINT}），发送口令和工位摘要，"
                         f"每次最多输出 {PLAN_MAX_TOKENS} tokens，使用校方分配的配额。")
    if args.mode=="run":
        from .flow import execute
        from .simulator import Simulator
        sim=Simulator(args.layout)
        try:
            if args.keepouts is not None:
                from .motion_guard import MotionGuard
                sim.guard = MotionGuard.from_config(sim.model,args.keepouts)
            result=execute(sim,args.text or LAYOUTS[args.layout].command,planner=access.plan if access.remaining["deepseek"] else None)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        finally:
            sim.close()
        return 0
    if args.sensevoice_dir and args.whisper_dir:
        parser.error("请只指定所选引擎的模型目录。")
    access.asr_engine = args.asr_engine or ("whisper" if args.whisper_dir else "sensevoice")
    if args.whisper_dir:
        if access.asr_engine != "whisper":
            parser.error("--whisper-dir 需要 --asr-engine whisper。")
        access.whisper_dir = args.whisper_dir.resolve()
    if args.sensevoice_dir:
        if access.asr_engine != "sensevoice":
            parser.error("--sensevoice-dir 需要 --asr-engine sensevoice。")
        access.sensevoice_dir = args.sensevoice_dir.resolve()
    import uvicorn
    from .web import create_app
    uvicorn.run(create_app(access,args.layout,keepouts=args.keepouts),host=args.host,port=args.port)
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except (ModelDenied,ModelFailure,ValueError) as exc:
        sys.exit(str(exc))
