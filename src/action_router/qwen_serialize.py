"""Serialization ported from the `submit_0707_qwen3_ls` submission's script.py.

This is intentionally a byte-for-byte copy of that script's `serialize()` /
`SERIALIZE_VARIANTS` (which its own docstring says mirrors an external
`src/data.py` not present in this repo). Do NOT "clean up" or merge this with
`action_router.features` -- a text-format mismatch between training and
inference silently tanks the score, so this stays a faithful, standalone port.

`richargs` is the variant used by the qwen3-0.6b full-FT + label-smoothing
checkpoint in `submit_0707_qwen3_ls`: rich meta line (adds `codelang`, `loc`,
`files=<open file basenames>`) + `arg_basenames` (strips directories from
every history action's `args` values, not just `open_files`) + full,
untruncated history.
"""
import re

_NO_VERDICT_ACTIONS = ("ask_user", "plan_task")
_EMPTY_RESULT_PREFIXES = ("0 ", "no matches", "empty directory", "found 0 ",
                          "no relevant results")
_LINT_ERRORS_RE = re.compile(r"[1-9]\d* errors?,")


def _result_ok(rs):
    if rs.startswith(("ERROR", "FAIL")):
        return False
    if rs.startswith("exit="):
        return rs.startswith("exit=0")
    if rs.startswith(_EMPTY_RESULT_PREFIXES):
        return False
    if _LINT_ERRORS_RE.match(rs):
        return False
    return True


def _budget_bucket(tokens):
    if tokens < 2_000:
        return "very_low"
    if tokens < 10_000:
        return "low"
    if tokens < 50_000:
        return "medium"
    return "high"


def _elapsed_bucket(seconds):
    if seconds < 120:
        return "early"
    if seconds < 900:
        return "mid"
    return "late"


def _strip_dirs(v):
    if "/" not in v:
        return v
    tail = "/" if v.endswith("/") else ""
    return v.rstrip("/").rsplit("/", 1)[-1] + tail


SERIALIZE_VARIANTS = {
    "v1":        {},
    "nometa":    {"drop_meta": True},
    "leanact":   {"lean_actions": True},
    "dupprompt": {"dup_prompt": True},
    "bareact":   {"lean_actions": True, "bare_actions": True},
    "combo":     {"drop_meta": True, "lean_actions": True, "dup_prompt": True,
                  "bare_actions": True},
    "richmeta":  {"rich_meta": True},
    "richargs":  {"rich_meta": True, "arg_basenames": True},
}


def serialize(r, max_hist=None, drop_meta=False, lean_actions=False,
              dup_prompt=False, bare_actions=False, rich_meta=False,
              arg_basenames=False, strip_history=False):
    sm = r["session_meta"]
    ws = sm["workspace"]
    parts = []
    if rich_meta:
        mix = ws.get("language_mix") or {}
        codelang = max(mix.items(), key=lambda kv: kv[1])[0] if mix else "-"
        names = [f.rsplit("/", 1)[-1] for f in ws["open_files"][:6]]
        parts.append(
            f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
            f"budget={_budget_bucket(sm['budget_tokens_remaining'])} "
            f"elapsed={_elapsed_bucket(sm['elapsed_session_sec'])} "
            f"codelang={codelang} loc={ws.get('loc', '-')} ci={ws['last_ci_status']} "
            f"dirty={ws['git_dirty']} open={len(ws['open_files'])} "
            f"files={','.join(names) or '-'}]"
        )
    elif not drop_meta:
        parts.append(
            f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
            f"budget={sm['budget_tokens_remaining']} ci={ws['last_ci_status']} "
            f"dirty={ws['git_dirty']} open={','.join(ws['open_files']) or '-'}]"
        )
    hist = [] if strip_history else (r["history"] if max_hist is None else r["history"][-max_hist:])
    for t in hist:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        elif lean_actions:
            if t["name"] in _NO_VERDICT_ACTIONS:
                line = t["name"]
            else:
                ok = "ok" if _result_ok(t.get("result_summary", "")) else "fail"
                line = f"{t['name']} {ok}"
            parts.append(line if bare_actions else
                         "ACTION " + line.replace(" ", " -> ", 1))
        else:
            args = t.get("args", {}) or {}
            if arg_basenames:
                args = {k: _strip_dirs(v) if isinstance(v, str) else v
                        for k, v in args.items()}
            parts.append(
                f"ACTION {t['name']}({args}) -> {t.get('result_summary', '')}"
            )
    if dup_prompt:
        parts.append(f"PROMPT: {r['current_prompt']} {r['current_prompt']}")
    else:
        parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)
