import json
from pathlib import PurePath


def _safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _budget_bucket(tokens):
    try:
        tokens = int(tokens)
    except (TypeError, ValueError):
        return "unknown"
    if tokens < 2_000:
        return "very_low"
    if tokens < 10_000:
        return "low"
    if tokens < 50_000:
        return "medium"
    return "high"


def _elapsed_bucket(seconds):
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 120:
        return "early"
    if seconds < 900:
        return "mid"
    return "late"


def _open_files_value(open_files, mode="count"):
    if mode == "count":
        return str(len(open_files))
    if mode == "basename":
        names = [PurePath(_safe_text(path)).name for path in open_files]
        return ",".join(name for name in names if name) or "-"
    if mode == "basename_space":
        names = [PurePath(_safe_text(path)).name for path in open_files]
        return " ".join(name for name in names if name) or "-"
    if mode == "ext":
        exts = []
        for path in open_files:
            suffix = PurePath(_safe_text(path)).suffix.lstrip(".")
            if suffix:
                exts.append(suffix)
        return ",".join(exts) or "-"
    if mode == "path":
        return ",".join(_safe_text(path) for path in open_files) or "-"
    raise ValueError(f"unknown open_files mode: {mode}")


def render_sample(sample, max_history=8):
    """Render one JSONL sample into a compact text input for a classifier.

    The public baseline only uses current_prompt. This representation adds the
    recent agent trajectory and workspace state because next actions often
    depend on what was just read, edited, or tested.
    """
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    recent_history = history[-max_history:]

    parts = [
        "task: predict next ai coding agent action",
        f"user_tier: {_safe_text(meta.get('user_tier'))}",
        f"language_pref: {_safe_text(meta.get('language_pref'))}",
        f"budget_bucket: {_budget_bucket(meta.get('budget_tokens_remaining'))}",
        f"turn_index: {_safe_text(meta.get('turn_index'))}",
        f"elapsed_bucket: {_elapsed_bucket(meta.get('elapsed_session_sec'))}",
        f"workspace_languages: {_compact_json(workspace.get('language_mix') or {})}",
        f"workspace_loc: {_safe_text(workspace.get('loc'))}",
        f"git_dirty: {_safe_text(workspace.get('git_dirty'))}",
        f"open_files: {_compact_json(workspace.get('open_files') or [])}",
        f"last_ci_status: {_safe_text(workspace.get('last_ci_status'))}",
    ]

    for item in recent_history:
        role = item.get("role", "")
        if role == "user":
            parts.append(f"history_user: {_safe_text(item.get('content'))}")
        elif role == "assistant_action":
            name = _safe_text(item.get("name"))
            args = _compact_json(item.get("args") or {})
            result = _safe_text(item.get("result_summary"))
            parts.append(f"history_action: {name} args={args} result={result}")
        else:
            parts.append(f"history_{role}: {_compact_json(item)}")

    parts.append(f"current_prompt: {_safe_text(sample.get('current_prompt'))}")
    return "\n".join(parts)


def render_granite_sample(sample, max_history_events=12, open_files_mode="count"):
    """Compact serialization used for the granite reproduction run.

    `max_history_events=12` corresponds to the last six user/action pairs in
    the released data format where history alternates user and assistant_action.
    """
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    recent_history = history[-max_history_events:]

    open_files = workspace.get("open_files") or []
    language_mix = workspace.get("language_mix") or {}
    main_lang = ""
    if language_mix:
        main_lang = max(language_mix.items(), key=lambda item: item[1])[0]

    meta_text = " ".join(
        [
            f"tier={_safe_text(meta.get('user_tier'))}",
            f"pref={_safe_text(meta.get('language_pref'))}",
            f"turn={_safe_text(meta.get('turn_index'))}",
            f"budget={_budget_bucket(meta.get('budget_tokens_remaining'))}",
            f"elapsed={_elapsed_bucket(meta.get('elapsed_session_sec'))}",
            f"lang={main_lang}",
            f"ci={_safe_text(workspace.get('last_ci_status'))}",
            f"git={'dirty' if workspace.get('git_dirty') else 'clean'}",
            (
                f"openfiles={_open_files_value(open_files, open_files_mode)}"
                if open_files_mode == "basename_space"
                else f"open={_open_files_value(open_files, open_files_mode)}"
            ),
            f"loc={_safe_text(workspace.get('loc'))}",
        ]
    )

    hist_parts = []
    for item in recent_history:
        role = item.get("role", "")
        if role == "user":
            hist_parts.append(f"U: {_safe_text(item.get('content'))}")
        elif role == "assistant_action":
            name = _safe_text(item.get("name"))
            args = _compact_json(item.get("args") or {})
            result = _safe_text(item.get("result_summary"))
            hist_parts.append(f"A[{name}] {args} -> {result}")

    return " ".join(
        [
            "[META]",
            meta_text,
            "[HIST]",
            " | ".join(hist_parts),
            "[CUR]",
            _safe_text(sample.get("current_prompt")),
        ]
    )


def session_group(sample_id):
    return _safe_text(sample_id).split("-step_", 1)[0]
