"""Shared action vocabulary for the campus planner, offline controls and simulator."""
from dataclasses import dataclass, replace
import math
import re
import unicodedata


@dataclass(frozen=True)
class Intent:
    kind: str
    axis: str = ""
    amount: float = 0
    message: str = ""
    source: str = ""
    target: str = ""
    steps: tuple = ()


SLOTS = ("A", "B", "C")
MAX_STEPS = 4
DIGITS = "零一二三四五六七八九"
NUMBER = r"(?:\d+(?:\.\d+)?|半|[零〇一二两三四五六七八九十百千]+(?:点[零〇一二三四五六七八九]+)?)"
UNIT = r"(毫米|厘米|公分|mm|cm|米|m)"
DIR = {"前":("x",1), "后":("x",-1), "左":("y",1), "右":("y",-1), "上":("z",1), "下":("z",-1)}
MOVE = re.compile(rf"(?:让)?(?:机械臂|夹爪|末端|爪子)?(?:向|往|朝)?([前后左右上下])(?:边|方)?(?:平移|移动|挪动|挪|移|走|抬高|降低)?({NUMBER}){UNIT}")
WRIST = re.compile(rf"(?:让)?(?:手腕|腕部|腕关节|夹爪)?(?:向|往|朝)?([上下])(?:俯仰|旋转|转动|转)?({NUMBER})(?:度|°)")
SLOT = r"(?:工位|区域|位置)?([abc])(?:号)?(?:工位|区域|区|点|位)?"
BLOCK = r"(?:(?:桌面上|桌上|前面|面前|那|这|左边|右边)(?:的|个|块)?)*(?:小|红色的|红色|红)?(?:木块|方块|积木|物块|立方体|块|它)"
TAKE = r"(?:抓取|抓住|抓起|夹起|夹住|拿起|拾起|抓|拿|夹)"
SEND = r"(?:放置|放入|放进|放|搬运|搬|送|挪动|挪|移动|移|推)(?:到|入|进|至|回|在|往)?"
OBJECT = rf"(?:(?:从)?{SLOT}(?:里|上|中)?的?)?(?:把|将)?{BLOCK}(?:从{SLOT})?"


def number(token):
    token = token.replace("两","二").replace("〇","零")
    if token == "半":
        return .5
    if token[0].isascii():
        return float(token)
    integer, _, frac = token.partition("点")
    if "百" in integer or "千" in integer:
        # Never guess colloquial shorthand (“一百九” can mean 190).
        if not re.fullmatch(r"(?:[一二三四五六七八九]千)?(?:零?[一二三四五六七八九]百)?(?:零?[一二三四五六七八九]十[一二三四五六七八九]?|零[一二三四五六七八九])?", integer):
            raise ValueError("请补充明确数值")
        value = sum((1000 if u=="千" else 100 if u=="百" else 10 if u=="十" else 1)*DIGITS.index(n)
                    for n,u in re.findall(r"([一二三四五六七八九])([千百十]?)", integer))
    elif "十" in integer:
        tens, ones = integer.split("十")
        value = (DIGITS.index(tens) if tens else 1)*10 + (DIGITS.index(ones) if ones else 0)
    elif len(integer)==1:
        value = DIGITS.index(integer)
    else:
        raise ValueError("请补充数字和单位")
    return float(str(value)+"."+"".join(str(DIGITS.index(c)) for c in frac)) if frac else float(value)


def actions(intent):
    return intent.steps if intent.kind == "sequence" else (intent,)


def joined(items):
    return items[0] if len(items) == 1 else Intent("sequence", steps=tuple(items))


def ask(message):
    return Intent("ask", message=message)


# Only complete requests in this vocabulary interrupt a live trajectory. For example,
# “不要停” is a prohibition and “拿稳以后暂停” is conditional, not an immediate stop.
COURTESY = r"麻烦你?|请(?:你|问)?|帮我|能不能|可不可以|能否"
POLITE_PREFIX = re.compile(rf"^(?:{COURTESY})+")
PAUSE_REQUEST = re.compile(
    r"(?:(?:现在|马上|立即|先|给我|把机械臂|让机械臂|机械臂))*"
    r"(?:暂停|停止|停下(?:来)?|停住|停一停|停一下|停|取消|"
    r"(?:别|不要)(?:再)?(?:动|继续(?:执行|移动|操作)?))"
    r"(?:一下)?(?:当前|正在执行的|正在进行的|这个|这次|现在的)?"
    r"(?:动作|任务|运动|操作|指令|移动|执行)?(?:一下)?(?:了|吧|好吗|可以吗)?"
)
PROHIBITION = re.compile(
    r"不要|不能|不可以|不许|不准|不得|不可|不用|不必|不需要|不允许|不应该|"
    r"不再|不想|不是|禁止|不[抓拿放走转移挪去动]|莫要|勿|"
    r"(?<!识)(?<!分)(?<!特)(?<!区)(?<!个)别(?!的|人)"
)
CONTROL_CLAUSES = re.compile(r"[,;。！？!?]|但是|而是|改为|改成|改去|然后|接着|随后|并且|最后")
CONTROL_SCRIPT = str.maketrans("請問煩幫暫別動現繼續執務當個讓機許應該轉識區後馬",
                               "请问烦帮暂别动现继续执务当个让机许应该转识区后马")
COMMAND_SCRIPT = str.maketrans("紅塊邊點釐兩數單腕關節輕體這裡夾爪從調歸順號頭幾遠離與進處確認嗎",
                               "红块边点厘两数单腕关节轻体这里夹爪从调归顺号头几远离与进处确认吗")


def command_text(text):
    """Keep the original transcript for audit; normalize the control vocabulary for validation."""
    text = unicodedata.normalize("NFKC", text).lower().translate(CONTROL_SCRIPT).translate(COMMAND_SCRIPT)
    return "".join(text.split())


def classify_control_text(text):
    """Local control for completed utterances; None passes an ordinary request to planning."""
    normalized = command_text(text).strip("。.!?,;")
    clauses = [POLITE_PREFIX.sub("", clause) for clause in CONTROL_CLAUSES.split(normalized) if clause]
    if not clauses:
        return ask("请说出要做的动作。")
    if any(PAUSE_REQUEST.fullmatch(clause) for clause in clauses):
        return Intent("pause", message="收到暂停请求。")
    negated = [bool(PROHIBITION.search(clause)) for clause in clauses]
    if all(negated):
        return Intent("ignore", message="收到，本条禁止指令不产生新动作。")
    if any(negated) or any(re.search(r"暂停|停止|取消|停下", clause) for clause in clauses):
        return ask("这句话里有停止条件或不同要求，请单独说要执行的动作。")
    return None


def local_step(text):
    text = re.sub(rf"^(?:{COURTESY}|先|再|然后|把|将)+", "", text)
    text = re.sub(r"(?:一下|谢谢|好吗|吧|可以吗)+$", "", text)
    for word, letter in (("左边", "a"), ("右边", "b"), ("中间", "c")):
        text = re.sub(word + r"(?:的)?(?:那个)?(?:工位|区域|格子)", letter + "区", text)
    if m := MOVE.fullmatch(text):
        direction, n, unit = m.groups()
        axis, sign = DIR[direction]
        scale = {"厘米":10,"公分":10,"cm":10,"毫米":1,"mm":1,"米":1000,"m":1000}[unit]
        return Intent("move", axis, sign*number(n)*scale)
    if m := WRIST.fullmatch(text):
        return Intent("wrist", "wrist_flex", number(m[2])*(1 if m[1]=="下" else -1))
    if m := re.fullmatch(OBJECT + SEND + SLOT, text):
        return Intent("transfer", source=(m[1] or m[2] or "").upper(), target=m[3].upper())
    if m := re.fullmatch(r"(?:向前|往前)?" + TAKE + OBJECT + r"(?:起来|起)?", text):
        return Intent("pick", source=(m[1] or m[2] or "").upper())
    if m := re.fullmatch(OBJECT + TAKE + r"(?:起来|起)?", text):
        return Intent("pick", source=(m[1] or m[2] or "").upper())
    if m := re.fullmatch(SEND + SLOT, text):
        return Intent("place", target=m[1].upper())
    if re.fullmatch(r"(?:看一下|查看|看看|观察)(?:桌面|工位|木块|红块)?|(?:红块|木块|方块|它)(?:现在)?在(?:哪儿|哪里|哪|哪个工位)", text):
        return Intent("observe")
    if text in ("归位","回零","回到初始位置","回到待机位","回到原位"):
        return Intent("home")
    if re.search(r"(?:木块|红块|方块|积木|它).*(?:放|送|搬|挪|移)", text):
        return ask("要把木块放到哪个工位？请回复 A、B 或 C。")
    if re.search(r"手腕|腕部|几度|俯仰|旋转", text):
        return ask("手腕往哪个方向转多少度？例如“向下转五度”。")
    if re.search(r"[前后左右上下].*(?:移|挪|走|一点|厘米|毫米)", text):
        return ask("需要移动多远？请补充方向、距离和单位。")
    return ask("这句需要进一步解释。请说清目标或动作，也可以使用交大 DeepSeek 规划原话。")


def local_intent(text):
    control = classify_control_text(text)
    if control:
        # Legacy offline callers represent non-motion as ask. The live controller
        # handles the distinct ignore/pause intents before invoking any planner.
        return ask(control.message)
    normalized = command_text(text).strip("。.!！?")
    normalized = re.sub(r"(红(?:色)?(?:木|方)?|木|方)快(?=放|送|搬|从|移|挪|抓|夹|拿|$)", r"\1块", normalized)
    parts = [p for p in re.split(r"(?:然后|接着|随后|最后|并且|并|再|[,，;；。])+",
                                 normalized) if p]
    if not 1 <= len(parts) <= MAX_STEPS:
        return ask("每条任务可连续执行一到四个动作，请分次发送。")
    try:
        items = [local_step(part) for part in parts]
    except (ValueError, OverflowError):
        return ask("数字写法不明确，请补充数值和单位。")
    return next((item for item in items if item.kind == "ask"), None) or joined(items)


def source_slot(scene):
    objects = scene["objects"]
    if not objects:
        return None, "当前画面没有找到木块。"
    if len(objects) != 1:
        return None, "可见目标不唯一，请重新摆放或确认目标。"
    if objects[0]["slot"] not in SLOTS:
        return None, "木块没有完整位于已示教工位。"
    return objects[0]["slot"], ""


def check_steps(intent, scene):
    """Check the whole plan before moving, including the predicted effect of earlier steps."""
    slot, reason = source_slot(scene)
    held = bool(scene.get("holding"))
    for item in actions(intent):
        if item.kind in ("move", "wrist"):
            limit = 50 if item.kind == "move" else 15
            if not math.isfinite(item.amount) or not 0 < abs(item.amount) <= limit:
                return "rejected", f"单步{'平移' if item.kind=='move' else '转腕'}限于 {limit} {'mm' if item.kind=='move' else '度'}以内。"
        if item.kind == "home" and held:
            return "clarify", "夹爪里还有木块，请先指定放置工位。"
        if item.kind == "place" and not held:
            return "clarify", "夹爪里没有木块。请先抓取，或说明要搬运的木块。"
        if item.kind in ("transfer","pick"):
            if held and (item.kind == "pick" or item.source):
                return "clarify", "夹爪已经持有木块，请指定放置工位。"
            if not held and (not slot or (item.source and item.source != slot)):
                at_goal = "木块已经在目标工位。" if slot and item.target==slot else ""
                return "clarify", reason or f"你指定从 {item.source} 区取，相机看到木块在 {slot} 区。{at_goal}请确认是否以当前观测为准。"
        if item.kind == "pick":
            held, slot = True, None
        if item.kind in ("transfer","place"):
            if item.target not in SLOTS:
                return "clarify", "请选择 A、B 或 C 工位。"
            held, slot = False, item.target
    return None


def decode_action(row):
    if not isinstance(row, dict):
        raise ValueError("动作须为 JSON 对象")
    kind = row.get("action")
    fields = {"transfer":{"action","object","from","to"}, "pick":{"action","object","from"},
              "place":{"action","to"}, "move":{"action","axis","mm"}, "wrist":{"action","deg"},
              "observe":{"action"}, "home":{"action"}}
    if not isinstance(kind, str) or kind not in fields or set(row) != fields[kind]:
        raise ValueError("未知动作或参数")
    if "from" in row and (row["object"] != "red_1" or row["from"] not in SLOTS):
        raise ValueError("起点或物体无效")
    if "to" in row and row["to"] not in SLOTS:
        raise ValueError("目标工位无效")
    if kind == "move" and row["axis"] not in ("x","y","z"):
        raise ValueError("未知坐标轴")
    amount = row.get("mm", row.get("deg", 0))
    if type(amount) not in (int,float) or not math.isfinite(amount):
        raise ValueError("数值无效")
    return Intent(kind, axis=row.get("axis", "wrist_flex" if kind == "wrist" else ""),
                  amount=amount, source=row.get("from",""), target=row.get("to",""))


def decode_model_plan(response, scene):
    if not isinstance(response, dict):
        raise ValueError("规划须为 JSON 对象")
    if set(response) == {"action","question"} and response["action"] == "ask":
        question = response["question"]
        if isinstance(question,str) and 0 < len(question.strip()) <= 160:
            return ask(question.strip())
        raise ValueError("澄清问题无效")
    rows = response["steps"] if set(response) == {"steps"} else [response]
    if not isinstance(rows,list) or not 1 <= len(rows) <= MAX_STEPS:
        raise ValueError("动作序列长度无效")
    result = joined([decode_action(row) for row in rows])
    issue = check_steps(result, scene)
    if issue:
        raise ValueError(issue[1])
    return result


def missing_motion_unit(text):
    """Catch a spoken distance/angle cut off at its number, including inside a sequence."""
    return re.search(rf"(?:[前后左右上下](?:边|方)?(?:平移|移动|挪动|挪|移|走|转动|转)?|平移|移动|转动)"
                     rf"(?:了|大约)?{NUMBER}(?:左右)?(?:吧|好吗)?(?=$|[?,!;。]|\.(?!\d)|再|接着|然后|随后|最后)",
                     command_text(text))


def check_command_match(text, proposed, scene):
    """A clear command must keep its directions, amounts and destinations in the model plan."""
    if proposed.kind == "ask":
        return
    if control := classify_control_text(text):
        raise ValueError(control.message)
    if missing_motion_unit(text):
        raise ValueError("指令中的数值缺少单位。")
    dialogue = scene.get("dialogue")
    reference = offline_reply(dialogue, text) if dialogue else local_intent(text)
    if reference.kind == "ask":
        # For free-form commands the model can ground a transfer in the scene, but
        # relative displacements must also match a literal direction, number and unit.
        quantitative = [s for s in actions(proposed) if s.kind in ("move", "wrist")]
        if quantitative:
            literal = command_text(text)
            literal = re.sub(r"(?:手腕|腕部|腕关节)(?:轻轻|稍微)?(抬高|抬起|压低|下压)",
                             lambda m: "手腕向上转" if m[1] in ("抬高","抬起") else "手腕向下转", literal)
            matches = sorted([*MOVE.finditer(literal), *WRIST.finditer(literal)], key=lambda m:m.start())
            expected = [local_step(match[0]) for match in matches]
            if quantitative != expected:
                raise ValueError("请明确确认每段平移或转腕的方向、数值和单位。")
        return

    def primitives(plan):
        slot, _ = source_slot(scene)
        held = bool(scene.get("holding"))
        series = []
        for step in actions(plan):
            if step.kind in ("pick","transfer") and not held:
                if step.kind == "transfer" and slot == step.target:
                    continue
                series.append(("pick",step.source or slot))
                held = True
            if step.kind in ("place","transfer"):
                series.append(("place",step.target))
                slot, held = step.target, False
            elif step.kind != "pick":
                series.append((step.kind,step.axis,round(step.amount,6)))
        return series

    if primitives(reference) != primitives(proposed):
        raise ValueError("动作与指令中的对象、距离、方向或顺序不一致")


def offline_reply(previous, answer):
    short = command_text(answer).strip("。.!！")
    old_command = command_text(previous["command"]).rstrip("。.!?")
    if missing_motion_unit(old_command):
        if re.fullmatch(rf"(?:{UNIT}|度|°)", short):
            return local_intent(old_command + short)
        if re.fullmatch(rf"{NUMBER}(?:{UNIT}|度|°)", short):
            return local_intent(re.sub(NUMBER + "$", "", old_command) + short)
    if "以当前观测为准" in previous["question"] and short in ("确认","是","是的","对","以当前观测为准"):
        original = local_intent(previous["command"])
        if original.kind in ("transfer","pick"):
            return replace(original,source="")
    if m := re.fullmatch(SLOT, short):
        if "哪个工位" in previous["question"]:
            # A station answer replaces an unresolved destination noun phrase.
            # Keep the original transfer prefix and source; do not discard a second action.
            prefix = re.match(OBJECT + SEND, old_command)
            if prefix:
                destination = old_command[prefix.end():]
                label = re.fullmatch(r"[\u4e00-\u9fff]*(?:工位|区域|位置)", destination)
                extra_action = re.search(
                    rf"{NUMBER}|{SEND}|{TAKE}|然后|接着|随后|最后|再|并|转|停|不要|不能|不许|别",
                    destination)
                if label and not extra_action:
                    proposal = local_intent(old_command[:prefix.end()] + m[1] + "区")
                    if proposal.kind == "transfer":
                        return proposal
            return local_intent(re.sub(r"(?:过去|走|一下|到哪里|到哪儿|到哪|到|$)$", "", old_command, count=1)
                                + "到" + m[1] + "区")
    if re.fullmatch(NUMBER + r"(?:度|°|厘米|毫米|公分|mm|cm|m|米)", short):
        old = re.sub(r"(?:几度|几厘米|多少(?:度|厘米|毫米)?|一点(?:点)?|一下|[？?。])+$", "", previous["command"])
        return local_intent(old + short)
    return local_intent(answer)
