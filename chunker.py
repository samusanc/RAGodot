"""
chunk_godot.py — Godot RST → cleaned chunks
Three-tier routing with DEDICATED WORKERS PER TIER.

Model tiers (each tier has its own queue + worker pool):
  LARGE  : gemini-2.5-pro, gemini-3.1-pro-preview, gemini-3-pro-preview
  MID    : gemini-2.5-flash, gemini-3-flash-preview
  LITE   : gemini-2.5-flash-lite, gemini-3.1-flash-lite-preview
  QWEN-A : local host A (faster)  — gets files up to QWEN_A_MAX_SIZE chars
  QWEN-B : local host B (slower)  — gets files up to QWEN_B_MAX_SIZE chars

When a model fails N times in a row it is blacklisted for the session.
Failed jobs cascade DOWN the tier chain automatically.
If all Gemini models in a tier are blacklisted, jobs fall to the next tier.
If all Gemini tiers exhausted → Qwen.

Usage:
    python chunk_godot.py
    python chunk_godot.py --large-workers 2 --mid-workers 2 --lite-workers 2
    python chunk_godot.py --qwen-a-workers 1 --qwen-b-workers 1
    python chunk_godot.py --dry-run
    python chunk_godot.py --fail-threshold 3
"""

import re
import sys
import time
import heapq
import argparse
import threading
import requests
from pathlib import Path
from collections import defaultdict
import ollama

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

# Qwen hosts — A is faster (desktop GPU), B is slower (3080 notebook)
OLLAMA_HOST_A    = "http://192.168.0.36:11434"
OLLAMA_HOST_B    = "http://192.168.0.190:11434"
OLLAMA_MODEL     = "qwen3:14b"
QWEN_NUM_CTX     = 3072

# Files larger than these limits get re-routed away from that Qwen host.
QWEN_A_MAX_SIZE  = 5_000   # Fast host handles anything up to SMALL_THRESHOLD
QWEN_B_MAX_SIZE  = 3_000   # Slow host only takes tiny files

GEMINI_BASE_URL  = "http://localhost:8317/v1/chat/completions"
GEMINI_API_KEY   = "your-api-key-3"

# Per-tier model lists (fallback order within each tier)
GEMINI_LARGE_TIER = ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3-pro-preview"]
GEMINI_MID_TIER   = ["gemini-2.5-flash", "gemini-3-flash-preview"]
GEMINI_LITE_TIER  = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"]

# Full fallback chain (used for status reporting)
GEMINI_FALLBACK_CHAIN = GEMINI_LARGE_TIER + GEMINI_MID_TIER + GEMINI_LITE_TIER

DEFAULT_FAIL_THRESHOLD = 5

# Size thresholds (pre-filtered chars) — controls initial tier assignment
SMALL_THRESHOLD  =  5_000   # <= → qwen
MID_THRESHOLD    = 15_000   # <= → lite tier
LARGE_THRESHOLD  = 45_000   # <= → mid tier
                             # >  → large tier

# Section splitter caps
QWEN_MAX_SECTION   =  3_000
GEMINI_MAX_SECTION = 20_000

# I/O
INPUT_ROOT  = Path("./godot")
OUTPUT_ROOT = Path("./godot_chk")

PATH_ROUTING = [
    ("classes/**/*.rst",         "class"),
    ("getting_started/**/*.rst", "tutorial"),
    ("tutorials/**/*.rst",       "tutorial"),
]

# Default workers per tier
DEFAULT_LARGE_WORKERS  = 2
DEFAULT_MID_WORKERS    = 2
DEFAULT_LITE_WORKERS   = 2
DEFAULT_QWEN_A_WORKERS = 1
DEFAULT_QWEN_B_WORKERS = 1

# Print model stats every N completions
STATS_EVERY = 25

# ═══════════════════════════════════════════════════════════════════════
# MODEL HEALTH TRACKER
# ═══════════════════════════════════════════════════════════════════════

class ModelHealth:
    def __init__(self, threshold: int):
        self._threshold   = threshold
        self._failures    = defaultdict(int)
        self._blacklisted = set()
        self._success_ct  = defaultdict(int)
        self._fail_ct     = defaultdict(int)
        self._active      = defaultdict(int)
        self._lock        = threading.Lock()

    def mark_start(self, model: str):
        with self._lock:
            self._active[model] += 1

    def mark_end(self, model: str):
        with self._lock:
            self._active[model] = max(0, self._active[model] - 1)

    def record_success(self, model: str):
        with self._lock:
            self._failures[model] = 0
            self._success_ct[model] += 1

    def record_failure(self, model: str) -> bool:
        with self._lock:
            self._failures[model] += 1
            self._fail_ct[model]  += 1
            if (self._failures[model] >= self._threshold
                    and model not in self._blacklisted):
                self._blacklisted.add(model)
                return True
            return False

    def is_available(self, model: str) -> bool:
        with self._lock:
            return model not in self._blacklisted

    def tier_available(self, tier: list) -> list:
        with self._lock:
            return [m for m in tier if m not in self._blacklisted]

    def status_line(self) -> str:
        with self._lock:
            bad = list(self._blacklisted)
            ok  = [m for m in GEMINI_FALLBACK_CHAIN if m not in self._blacklisted]
            return (f"Gemini available: {len(ok)} | blacklisted: {len(bad)}"
                    + (f" ({', '.join(bad)})" if bad else ""))

    def model_stats_table(self) -> str:
        with self._lock:
            display = GEMINI_FALLBACK_CHAIN + ["qwen-a", "qwen-b", "qwen_fallback"]
            lines = [f"\n{'─'*72}",
                     f"  {'MODEL':<37} {'OK':>6} {'FAIL':>6} {'IN-FLIGHT':>10}  STATUS",
                     f"{'─'*72}"]
            for m in display:
                ok  = self._success_ct.get(m, 0)
                bad = self._fail_ct.get(m, 0)
                inf = self._active.get(m, 0)
                if ok == 0 and bad == 0:
                    continue
                bl = " [BLACKLISTED]" if m in self._blacklisted else ""
                lines.append(f"  {m:<37} {ok:>6} {bad:>6} {inf:>10}{bl}")
            lines.append(f"{'─'*72}")
        return "\n".join(lines)


health: ModelHealth = None   # set in main()

# ═══════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a documentation pre-processor for a Godot Engine RAG system. Your sole job is to clean and restructure raw RST documentation sections into dense, embedding-friendly plain text chunks.

RULES — follow every one exactly:

1. OUTPUT FORMAT: Return only the cleaned chunk. No preamble, no "Here is the cleaned chunk:", no markdown code fences, no explanation after. No <answer> tags. No ### headers.

2. ALWAYS START with this metadata block (fill in the values):
   Document: [class or tutorial name]
   Path: [original .rst file path]
   Summary: [one sentence describing what this chunk covers]

3. STRIP completely — do not keep, mention, or reference:
   - RST directives starting with ".." (.. code-block::, .. note::, .. warning::, etc.)
   - Substitution definitions: lines like ".. |virtual| replace:: ..."
   - Internal hyperlink syntax: :ref:`SomeClass <class_SomeClass>` → write just "SomeClass"
   - GitHub URLs, "DO NOT EDIT" headers, generator comments, XML source links
   - ASCII table borders: lines made of "+---+" or "|   |" patterns
   - Image references: lines like ".. image:: ..." or "|imageN|"
   - Empty lines used only for RST spacing

4. CONVERT to plain prose/lists:
   - RST tables → bullet list of "name (Type): description. Default: value."
   - ".. note::" blocks → inline sentence starting with "Note:"
   - ".. warning::" blocks → inline sentence starting with "Warning:"
   - ".. tip::" blocks → inline sentence starting with "Tip:"
   - Method signatures → "Method: name(params) → ReturnType" then description below

5. KEEP intact, with plain labels (no markdown fences):
   - All GDScript examples — label as "GDScript:" on its own line, then indented code
   - All C# examples — label as "C#:" on its own line, then indented code
   - All class names, method names, property names (exact casing)
   - Inheritance chain ("Inherits: A < B < C")
   - Enum values and their integer mappings

6. CHUNK SIZE target: 300–600 words. Very short sections still need the metadata header.

7. BOOLEAN CONTEXT notes and edge cases → rewrite as "Key Rule: [plain sentence]".

8. Do NOT add information not present in the input. Do NOT hallucinate methods, use cases, or best practices.

Output ONLY the cleaned chunk. No ---CHUNK--- separators."""

USER_CLASS_DESC    = "Process this Godot CLASS reference section — class identity, description, inheritance.\nFile path: {path}\n\nRST:\n{content}"
USER_CLASS_PROPS   = "Process this Godot CLASS reference section — properties/member variables.\nFile path: {path}\n\nRST:\n{content}"
USER_CLASS_METHODS = "Process this Godot CLASS reference section — methods (group {group}).\nFile path: {path}\n\nRST:\n{content}"
USER_TUTORIAL      = "Process this Godot TUTORIAL section.\nFile path: {path}\n\nRST:\n{content}"

# ═══════════════════════════════════════════════════════════════════════
# PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════

_STRIP_RE = [
    re.compile(r"^\.\.\s*\|.*$"),
    re.compile(r"^[+|][-=+|]+[+|]\s*$"),
    re.compile(r"^:github_url:.*$"),
    re.compile(r"^:gitref:.*$"),
    re.compile(r".*DO NOT EDIT THIS FILE.*"),
    re.compile(r".*Generated automatically.*"),
    re.compile(r"^\.\. Generated.*$"),
    re.compile(r"^\.\. XML source.*$"),
    re.compile(r"^\.\. image::.*$"),
    re.compile(r"^\|image\d+\|.*$"),
]

def pre_filter(text: str) -> str:
    kept = [l for l in text.splitlines()
            if not any(p.match(l.strip()) for p in _STRIP_RE)]
    result, blanks = [], 0
    for line in kept:
        if line.strip() == "":
            blanks += 1
            if blanks <= 2:
                result.append(line)
        else:
            blanks = 0
            result.append(line)
    return "\n".join(result)

# ═══════════════════════════════════════════════════════════════════════
# RST SPLITTER
# ═══════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(r"^(.+)\n([=\-~^\"'`#*+])\2{2,}\s*$", re.MULTILINE)

def _split_headings(text: str) -> list:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    pre = text[:matches[0].start()].strip()
    if pre:
        sections.append(("__preamble__", pre))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        body  = text[start:end].strip()
        sections.append((title, body))
    return sections

def _split_class(text: str, max_chars: int) -> list:
    sections = _split_headings(text)
    result, mbuf, mchars, mgroup = [], [], 0, 1

    def flush():
        nonlocal mgroup, mbuf, mchars
        if mbuf:
            result.append((f"methods_{mgroup}", "\n\n".join(mbuf)))
            mgroup += 1; mbuf.clear(); mchars = 0

    for title, body in sections:
        if not body:
            continue
        label = title.lower()
        chunk = f"{title}\n{body}" if title not in ("__preamble__", "") else body
        if label in ("__preamble__", "description", ""):
            flush(); result.append(("description", chunk))
        elif any(k in label for k in ("propert", "member")):
            flush(); result.append(("properties", chunk))
        elif any(k in label for k in ("method", "constructor", "operator",
                                       "signal", "enum", "theme")):
            if mchars + len(chunk) > max_chars:
                flush()
            mbuf.append(chunk); mchars += len(chunk)
        else:
            flush(); result.append(("other", chunk))
    flush()
    return result

def _split_tutorial(text: str, max_chars: int) -> list:
    result = []
    for title, body in _split_headings(text):
        content = f"{title}\n{body}" if title not in ("__preamble__", "") else body
        if len(content) <= max_chars:
            result.append((title, content))
        else:
            paras = re.split(r"\n{2,}", content)
            buf, part = "", 1
            for para in paras:
                if len(buf) + len(para) > max_chars and buf:
                    result.append((f"{title} (part {part})", buf.strip()))
                    part += 1; buf = para
                else:
                    buf += "\n\n" + para
            if buf.strip():
                result.append((f"{title} (part {part})", buf.strip()))
    return result

# ═══════════════════════════════════════════════════════════════════════
# LLM BACKENDS
# ═══════════════════════════════════════════════════════════════════════

_tl = threading.local()

def _qwen_client(host: str) -> ollama.Client:
    key = f"client_{host}"
    if not hasattr(_tl, key):
        setattr(_tl, key, ollama.Client(host=host))
    return getattr(_tl, key)

def call_qwen(user_msg: str, host: str, stat_key: str) -> str:
    client = _qwen_client(host)
    health.mark_start(stat_key)
    for attempt in range(1, 4):
        try:
            resp = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                options={"temperature": 0.1, "num_ctx": QWEN_NUM_CTX},
            )
            health.mark_end(stat_key)
            health.record_success(stat_key)
            return resp["message"]["content"].strip()
        except Exception as e:
            _log(f"    [{stat_key}] attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(4 * attempt)
    health.mark_end(stat_key)
    health.record_failure(stat_key)
    return ""

def _call_gemini_model(model: str, user_msg: str) -> str:
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
    }
    t0 = time.monotonic()
    r  = requests.post(GEMINI_BASE_URL, json=payload,
                       headers=headers, timeout=240)
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    result = r.json()["choices"][0]["message"]["content"].strip()
    _log(f"    [gemini] {model} {elapsed:.1f}s → {len(result)} chars")
    return result

def _try_tier(tier_models: list, user_msg: str, logs: list):
    """Try each available model in a tier. Returns (result, model) or None."""
    available = health.tier_available(tier_models)
    if not available:
        return None
    for model in available:
        health.mark_start(model)
        try:
            result = _call_gemini_model(model, user_msg)
            health.record_success(model)
            health.mark_end(model)
            return result, model
        except Exception as e:
            health.mark_end(model)
            newly_bl = health.record_failure(model)
            status   = "BLACKLISTED" if newly_bl else "failed"
            logs.append(f"    {model} → {status}: {e}")
            if newly_bl:
                _log(f"[!] {model} blacklisted. {health.status_line()}")
            time.sleep(2)
    return None

def call_gemini_with_cascade(user_msg: str, start_tier: str, logs: list):
    """
    Try start_tier first, then cascade to other tiers, then Qwen fallback.
    Returns (result, model_used).
    """
    if start_tier == "large":
        tier_order = [GEMINI_LARGE_TIER, GEMINI_MID_TIER, GEMINI_LITE_TIER]
    elif start_tier == "mid":
        tier_order = [GEMINI_MID_TIER, GEMINI_LITE_TIER, GEMINI_LARGE_TIER]
    else:  # lite
        tier_order = [GEMINI_LITE_TIER, GEMINI_MID_TIER, GEMINI_LARGE_TIER]

    for tier in tier_order:
        result = _try_tier(tier, user_msg, logs)
        if result:
            return result

    logs.append("    all Gemini unavailable → Qwen fallback")
    _log(f"[!] All Gemini down — falling back to Qwen. {health.status_line()}")
    for host, key in [(OLLAMA_HOST_A, "qwen-a"), (OLLAMA_HOST_B, "qwen-b")]:
        res = call_qwen(user_msg, host, "qwen_fallback")
        if res:
            return res, f"qwen_fallback@{host}"
    return "", "qwen_fallback_failed"

# ═══════════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════════

def assign_tier(filtered_size: int) -> str:
    if filtered_size <= SMALL_THRESHOLD:
        return "qwen"
    if filtered_size <= MID_THRESHOLD:
        return "lite"
    if filtered_size <= LARGE_THRESHOLD:
        return "mid"
    return "large"

# ═══════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def _make_prompt(label: str, content: str, path: str, group: int) -> str:
    if label == "description":
        return USER_CLASS_DESC.format(path=path, content=content)
    if label == "properties":
        return USER_CLASS_PROPS.format(path=path, content=content)
    if label.startswith("methods_"):
        return USER_CLASS_METHODS.format(path=path, content=content, group=group)
    return USER_TUTORIAL.format(path=path, content=content)

# ═══════════════════════════════════════════════════════════════════════
# FILE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════

def process_file(raw: str, prompt_type: str, file_path: Path,
                 tier: str, qwen_host, logs: list) -> str:

    path_str  = str(file_path)
    filtered  = pre_filter(raw)
    use_qwen  = (tier == "qwen") and (qwen_host is not None)
    max_chars = QWEN_MAX_SECTION if use_qwen else GEMINI_MAX_SECTION

    pct = 100 * (1 - len(filtered) / max(len(raw), 1))
    logs.append(f"pre-filter: {len(raw)}→{len(filtered)} chars ({pct:.0f}% stripped)")

    splitter = _split_class if prompt_type == "class" else _split_tutorial
    sections = splitter(filtered, max_chars)
    logs.append(f"split: {len(sections)} section(s) [tier={tier}]")

    chunks, method_group = [], 1
    for label, content in sections:
        if not content.strip():
            continue
        prompt = _make_prompt(label, content, path_str, method_group)
        if label.startswith("methods_"):
            method_group += 1

        short = (label[:35] + "…") if len(label) > 35 else label
        logs.append(f"  • {short} ({len(content)} chars)")

        if use_qwen:
            stat_key = "qwen-a" if qwen_host == OLLAMA_HOST_A else "qwen-b"
            result   = call_qwen(prompt, qwen_host, stat_key)
            used     = stat_key
        else:
            result, used = call_gemini_with_cascade(prompt, tier, logs)

        if result:
            chunks.append(result)
            logs[-1] += f" → ok [{used}]"
        else:
            logs[-1] += " → EMPTY"

    if not chunks:
        logs.append("! no chunks — using pre-filtered fallback")
        return filtered

    return "\n---CHUNK---\n".join(chunks)

# ═══════════════════════════════════════════════════════════════════════
# PRIORITY QUEUE
# ═══════════════════════════════════════════════════════════════════════

class PriorityJobQueue:
    def __init__(self):
        self._heap   = []
        self._seq    = 0
        self._lock   = threading.Lock()
        self._cv     = threading.Condition(self._lock)
        self._closed = False

    def put(self, priority: int, job: dict):
        with self._cv:
            heapq.heappush(self._heap, (priority, self._seq, job))
            self._seq += 1
            self._cv.notify()

    def get(self, timeout: float = 3.0):
        with self._cv:
            deadline = time.monotonic() + timeout
            while not self._heap:
                if self._closed:
                    return None
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return None
                self._cv.wait(timeout=rem)
            _, _, job = heapq.heappop(self._heap)
            return job

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def __len__(self):
        with self._lock:
            return len(self._heap)

# ═══════════════════════════════════════════════════════════════════════
# WORKER LOOPS
# ═══════════════════════════════════════════════════════════════════════

_print_lock  = threading.Lock()
_done_count  = 0
_done_lock   = threading.Lock()
_stats_count = 0
_stats_lock  = threading.Lock()
_total       = 0   # set in main()

def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)

def _maybe_print_stats():
    global _stats_count
    with _stats_lock:
        _stats_count += 1
        do_print = (_stats_count % STATS_EVERY == 0)
    if do_print:
        _log(health.model_stats_table())

def _run_job(job: dict, worker_id: str):
    global _done_count
    idx         = job["idx"]
    file_path   = job["file_path"]
    prompt_type = job["prompt_type"]
    output_path = job["output_path"]
    tier        = job["tier"]
    qwen_host   = job.get("qwen_host")
    raw         = job["raw"]

    rel = file_path.relative_to(INPUT_ROOT)
    _log(f"[{idx}/{_total}] START  {rel}  [{worker_id}] tier={tier}")

    logs = []
    try:
        content = process_file(raw, prompt_type, file_path,
                               tier, qwen_host, logs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        n = len([p for p in content.split("---CHUNK---") if p.strip()])

        with _done_lock:
            _done_count += 1

        _log(f"[{idx}/{_total}] DONE   {file_path.name} → {n} chunk(s)")
        for line in logs:
            _log(f"  {line}")

        _maybe_print_stats()

    except Exception as e:
        with _done_lock:
            _done_count += 1
        _log(f"[{idx}/{_total}] FAIL   {file_path.name}: {e}")


def gemini_worker_loop(queue_obj: PriorityJobQueue, worker_id: str):
    """Worker dedicated to a Gemini tier queue."""
    while True:
        job = queue_obj.get(timeout=5.0)
        if job is None:
            break
        _run_job(job, worker_id)


def qwen_worker_loop(queue_obj: PriorityJobQueue, worker_id: str,
                     host: str, host_max_size: int,
                     overflow_queue):
    """
    Qwen worker for a specific host.
    Jobs exceeding host_max_size are re-routed to overflow_queue (Gemini lite).
    """
    while True:
        job = queue_obj.get(timeout=5.0)
        if job is None:
            break
        size = job["filtered_size"]
        if size > host_max_size and overflow_queue is not None:
            _log(f"[{job['idx']}/{_total}] REROUTE {job['file_path'].name} "
                 f"({size} chars > {host_max_size}) → lite [{worker_id}]")
            job["tier"] = "lite"
            job.pop("qwen_host", None)
            overflow_queue.put(-size, job)
        else:
            job["qwen_host"] = host
            _run_job(job, worker_id)

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    global health, _total

    parser = argparse.ArgumentParser()
    parser.add_argument("--large-workers",  type=int, default=DEFAULT_LARGE_WORKERS,
                        help=f"Workers for large Gemini tier (default {DEFAULT_LARGE_WORKERS})")
    parser.add_argument("--mid-workers",    type=int, default=DEFAULT_MID_WORKERS,
                        help=f"Workers for mid Gemini tier (default {DEFAULT_MID_WORKERS})")
    parser.add_argument("--lite-workers",   type=int, default=DEFAULT_LITE_WORKERS,
                        help=f"Workers for lite Gemini tier (default {DEFAULT_LITE_WORKERS})")
    parser.add_argument("--qwen-a-workers", type=int, default=DEFAULT_QWEN_A_WORKERS,
                        help=f"Workers for fast Qwen host A (default {DEFAULT_QWEN_A_WORKERS})")
    parser.add_argument("--qwen-b-workers", type=int, default=DEFAULT_QWEN_B_WORKERS,
                        help=f"Workers for slow Qwen host B (default {DEFAULT_QWEN_B_WORKERS})")
    parser.add_argument("--fail-threshold", type=int, default=DEFAULT_FAIL_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    health = ModelHealth(threshold=args.fail_threshold)

    if not INPUT_ROOT.exists():
        print(f"Error: '{INPUT_ROOT}' not found."); sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Collect & route jobs ────────────────────────────────────────
    all_jobs, seen = [], set()
    for pattern, prompt_type in PATH_ROUTING:
        for file_path in sorted(INPUT_ROOT.glob(pattern)):
            if file_path in seen:
                continue
            seen.add(file_path)
            rel         = file_path.relative_to(INPUT_ROOT)
            output_path = OUTPUT_ROOT / rel.with_suffix(".chk")
            if output_path.exists():
                print(f"SKIP  {rel}")
                continue
            raw      = file_path.read_text(encoding="utf-8")
            filtered = pre_filter(raw)
            fsize    = len(filtered)
            tier     = assign_tier(fsize)
            all_jobs.append({
                "file_path":     file_path,
                "prompt_type":   prompt_type,
                "output_path":   output_path,
                "raw":           raw,
                "filtered_size": fsize,
                "tier":          tier,
            })

    _total = len(all_jobs)
    if _total == 0:
        print("Nothing to do."); return

    # ── Routing table ───────────────────────────────────────────────
    tier_counts: dict = defaultdict(int)
    print(f"\n{'─'*65}")
    print(f"{'FILE':<48} {'SIZE':>7}  TIER")
    print(f"{'─'*65}")
    for job in all_jobs:
        rel = job["file_path"].relative_to(INPUT_ROOT)
        t   = job["tier"]
        tier_counts[t] += 1
        print(f"{str(rel):<48} {job['filtered_size']:>7}  {t}")
    print(f"{'─'*65}")
    print(f"Total: {_total} files")
    print(f"Tier counts:  large={tier_counts['large']}  mid={tier_counts['mid']}  "
          f"lite={tier_counts['lite']}  qwen={tier_counts['qwen']}")
    print(f"\nWorkers:      large={args.large_workers}  mid={args.mid_workers}  "
          f"lite={args.lite_workers}  qwen-a={args.qwen_a_workers}  "
          f"qwen-b={args.qwen_b_workers}  fail_threshold={args.fail_threshold}")
    print(f"Qwen A (fast, max {QWEN_A_MAX_SIZE:,} chars): {OLLAMA_HOST_A}")
    print(f"Qwen B (slow, max {QWEN_B_MAX_SIZE:,} chars): {OLLAMA_HOST_B}")
    print(f"Size thresholds: qwen<={SMALL_THRESHOLD:,}  lite<={MID_THRESHOLD:,}  "
          f"mid<={LARGE_THRESHOLD:,}  large=rest")
    print(f"Gemini large: {GEMINI_LARGE_TIER}")
    print(f"Gemini mid  : {GEMINI_MID_TIER}")
    print(f"Gemini lite : {GEMINI_LITE_TIER}")

    if args.dry_run:
        return

    # ── Build separate queues per tier ─────────────────────────────
    q_large  = PriorityJobQueue()
    q_mid    = PriorityJobQueue()
    q_lite   = PriorityJobQueue()
    q_qwen_a = PriorityJobQueue()
    q_qwen_b = PriorityJobQueue()

    for idx, job in enumerate(all_jobs, 1):
        job["idx"] = idx
        size = job["filtered_size"]
        tier = job["tier"]
        if tier == "large":
            q_large.put(-size, job)    # largest first
        elif tier == "mid":
            q_mid.put(-size, job)
        elif tier == "lite":
            q_lite.put(-size, job)
        else:
            # Split between Qwen hosts:
            # tiny files (≤ QWEN_B_MAX_SIZE) go to slow host B
            # larger qwen files go to fast host A
            if size <= QWEN_B_MAX_SIZE:
                q_qwen_b.put(+size, job)
            else:
                q_qwen_a.put(+size, job)

    # ── Launch workers ──────────────────────────────────────────────
    t_start = time.time()
    threads = []
    all_queues = [q_large, q_mid, q_lite, q_qwen_a, q_qwen_b]

    for i in range(args.large_workers):
        t = threading.Thread(target=gemini_worker_loop,
                             args=(q_large, f"large-{i+1}"), daemon=True)
        t.start(); threads.append(t)

    for i in range(args.mid_workers):
        t = threading.Thread(target=gemini_worker_loop,
                             args=(q_mid, f"mid-{i+1}"), daemon=True)
        t.start(); threads.append(t)

    for i in range(args.lite_workers):
        t = threading.Thread(target=gemini_worker_loop,
                             args=(q_lite, f"lite-{i+1}"), daemon=True)
        t.start(); threads.append(t)

    # Qwen A: handles medium qwen files; re-routes oversized → lite
    for i in range(args.qwen_a_workers):
        t = threading.Thread(target=qwen_worker_loop,
                             args=(q_qwen_a, f"qwen-a-{i+1}",
                                   OLLAMA_HOST_A, QWEN_A_MAX_SIZE, q_lite),
                             daemon=True)
        t.start(); threads.append(t)

    # Qwen B: handles tiny files; re-routes oversized → host A queue
    for i in range(args.qwen_b_workers):
        t = threading.Thread(target=qwen_worker_loop,
                             args=(q_qwen_b, f"qwen-b-{i+1}",
                                   OLLAMA_HOST_B, QWEN_B_MAX_SIZE, q_qwen_a),
                             daemon=True)
        t.start(); threads.append(t)

    # ── Wait for completion ─────────────────────────────────────────
    last_stats = time.time()
    while True:
        with _done_lock:
            done = _done_count
        if done >= _total:
            break
        if time.time() - last_stats > 120:
            _log(health.model_stats_table())
            last_stats = time.time()
        time.sleep(1)

    for q in all_queues:
        q.close()
    for t in threads:
        t.join(timeout=10)

    elapsed = time.time() - t_start
    print(f"\n{'─'*65}")
    print(f"Done in {elapsed/60:.1f} min — {_done_count}/{_total} processed")
    print(health.status_line())
    print(health.model_stats_table())
    print(f"Output in '{OUTPUT_ROOT}/'")


def parse_chunks(chk_content: str) -> list:
    return [p.strip() for p in chk_content.split("---CHUNK---") if p.strip()]


if __name__ == "__main__":
    main()
