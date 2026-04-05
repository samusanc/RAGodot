"""
chunk_godot.py — Godot RST → cleaned chunks
Three-tier routing + cascading fallback chain.

Model priority (highest to lowest):
  gemini-2.5-pro / gemini-3.1-pro-preview
  → gemini-3-pro-preview
  → gemini-2.5-flash / gemini-3-flash-preview
  → gemini-2.5-flash-lite / gemini-3.1-flash-lite-preview
  → qwen (local, always available, round-robined across hosts)

When a model fails N times in a row it is blacklisted for the session.
Failed jobs cascade down the chain automatically.
If all Gemini models are blacklisted, Qwen handles everything.

Usage:
    python chunk_godot.py                         # auto workers
    python chunk_godot.py --qwen-workers 2 --gemini-workers 4
    python chunk_godot.py --dry-run               # routing table only
    python chunk_godot.py --fail-threshold 3      # blacklist after 3 failures
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

# Multiple Qwen hosts — round-robined per worker
OLLAMA_HOSTS = [
    "http://192.168.0.36:11434",
    "http://192.168.0.190:11434",
]
OLLAMA_MODEL = "qwen3:14b"
QWEN_NUM_CTX = 3072

GEMINI_BASE_URL = "http://localhost:8317/v1/chat/completions"
GEMINI_API_KEY  = "your-api-key-3"

# Full fallback chain — ordered from most to least capable.
# ALL models are included and used for initial routing (not just 2.5 family).
GEMINI_FALLBACK_CHAIN = [
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
]

# Tier groupings for initial routing (maps size → candidate pool)
GEMINI_LARGE_TIER = ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3-pro-preview"]
GEMINI_MID_TIER   = ["gemini-2.5-flash", "gemini-3-flash-preview"]
GEMINI_LITE_TIER  = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"]

# How many consecutive failures before a model is blacklisted for the session
DEFAULT_FAIL_THRESHOLD = 5

# Size thresholds (pre-filtered chars) for initial routing
SMALL_THRESHOLD  =  5_000   # ≤ this → qwen
LARGE_THRESHOLD  = 15_000   # > this → prefer large tier

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

DEFAULT_QWEN_WORKERS   = 2   # now 2 by default (one per host)
DEFAULT_GEMINI_WORKERS = 4

# ═══════════════════════════════════════════════════════════════════════
# MODEL HEALTH TRACKER
# ═══════════════════════════════════════════════════════════════════════

class ModelHealth:
    """Thread-safe failure counter with blacklisting + per-model stats."""

    def __init__(self, threshold: int):
        self._threshold   = threshold
        self._failures    = defaultdict(int)   # model → consecutive failures
        self._blacklisted = set()
        self._success_ct  = defaultdict(int)   # model → total successes
        self._fail_ct     = defaultdict(int)   # model → total failures
        self._active      = defaultdict(int)   # model → currently in-flight
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
        """Returns True if the model just got blacklisted."""
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

    def available_gemini(self) -> list[str]:
        with self._lock:
            return [m for m in GEMINI_FALLBACK_CHAIN
                    if m not in self._blacklisted]

    def status_line(self) -> str:
        with self._lock:
            ok  = [m for m in GEMINI_FALLBACK_CHAIN if m not in self._blacklisted]
            bad = list(self._blacklisted)
            return (f"Gemini available: {len(ok)} | blacklisted: {len(bad)}"
                    + (f" ({', '.join(bad)})" if bad else ""))

    def model_stats_table(self) -> str:
        """Return a formatted table of per-model call stats."""
        with self._lock:
            all_models = list(GEMINI_FALLBACK_CHAIN) + ["qwen_fallback", "qwen"]
            lines = [f"\n{'─'*70}",
                     f"  {'MODEL':<35} {'OK':>6} {'FAIL':>6} {'IN-FLIGHT':>10}  STATUS",
                     f"{'─'*70}"]
            for m in all_models:
                ok  = self._success_ct.get(m, 0)
                bad = self._fail_ct.get(m, 0)
                inf = self._active.get(m, 0)
                if ok == 0 and bad == 0:
                    continue
                bl = " [BLACKLISTED]" if m in self._blacklisted else ""
                lines.append(f"  {m:<35} {ok:>6} {bad:>6} {inf:>10}{bl}")
            lines.append(f"{'─'*70}")
        return "\n".join(lines)


# Global instance — set in main()
health: ModelHealth = None

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

def _split_headings(text: str) -> list[tuple[str, str]]:
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

def _split_class(text: str, max_chars: int) -> list[tuple[str, str]]:
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

def _split_tutorial(text: str, max_chars: int) -> list[tuple[str, str]]:
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

# Round-robin counter for Qwen host assignment
_qwen_host_counter = 0
_qwen_host_lock    = threading.Lock()

def _assign_qwen_host() -> str:
    """Pick the next Qwen host in round-robin order."""
    global _qwen_host_counter
    with _qwen_host_lock:
        host = OLLAMA_HOSTS[_qwen_host_counter % len(OLLAMA_HOSTS)]
        _qwen_host_counter += 1
    return host

def _qwen_client(host: str):
    """Return a thread-local ollama.Client for the given host."""
    key = f"client_{host}"
    if not hasattr(_tl, key):
        setattr(_tl, key, ollama.Client(host=host))
    return getattr(_tl, key)

def call_qwen(user_msg: str, host: str | None = None) -> str:
    """Always-available local fallback. Retries 3 times.
    If host is None, picks one via round-robin."""
    if host is None:
        host = _assign_qwen_host()
    client = _qwen_client(host)
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
            return resp["message"]["content"].strip()
        except Exception as e:
            _log(f"    [qwen@{host}] attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(4 * attempt)
    return ""

def _call_gemini_model(model: str, user_msg: str) -> str:
    """Single attempt on a specific Gemini model. Raises on failure."""
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
    r = requests.post(GEMINI_BASE_URL, json=payload,
                      headers=headers, timeout=180)
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    result = r.json()["choices"][0]["message"]["content"].strip()
    _log(f"    [gemini] {model} responded in {elapsed:.1f}s "
         f"({len(result)} chars out)")
    return result

def call_with_fallback(user_msg: str, preferred_model: str,
                       logs: list) -> tuple[str, str]:
    """
    Try preferred_model first, then cascade down GEMINI_FALLBACK_CHAIN,
    then fall back to Qwen. Returns (result_text, model_used).
    Logs all attempts.
    """
    chain = [preferred_model] + [
        m for m in GEMINI_FALLBACK_CHAIN
        if m != preferred_model
    ]

    for model in chain:
        if not health.is_available(model):
            logs.append(f"    skip {model} (blacklisted)")
            continue
        health.mark_start(model)
        try:
            result = _call_gemini_model(model, user_msg)
            health.record_success(model)
            health.mark_end(model)
            return result, model
        except Exception as e:
            health.mark_end(model)
            newly_blacklisted = health.record_failure(model)
            status = "BLACKLISTED" if newly_blacklisted else "failed"
            logs.append(f"    {model} → {status}: {e}")
            if newly_blacklisted:
                _log(f"[!] {model} blacklisted after repeated failures. "
                     f"{health.status_line()}")
            time.sleep(2)

    # All Gemini exhausted → Qwen
    logs.append("    all Gemini models unavailable → falling back to Qwen")
    _log(f"[!] All Gemini unavailable — using Qwen. {health.status_line()}")
    return call_qwen(user_msg), "qwen_fallback"

# ═══════════════════════════════════════════════════════════════════════
# ROUTING — now distributes across ALL model tiers
# ═══════════════════════════════════════════════════════════════════════

_tier_counters: dict[str, int] = {"large": 0, "mid": 0, "lite": 0}
_tier_lock = threading.Lock()

def _round_robin_from(tier_key: str, candidates: list[str],
                      fallback: str) -> str:
    """Pick next available model from a tier in round-robin order."""
    with _tier_lock:
        available = [m for m in candidates if health.is_available(m)]
        if not available:
            return fallback   # will cascade in call_with_fallback
        idx   = _tier_counters[tier_key] % len(available)
        model = available[idx]
        _tier_counters[tier_key] += 1
    return model

def initial_model(filtered_size: int) -> str:
    """Choose starting model based on file size, rotating across all tiers."""
    if filtered_size <= SMALL_THRESHOLD:
        return "qwen"
    if filtered_size <= LARGE_THRESHOLD:
        return _round_robin_from("lite", GEMINI_LITE_TIER,
                                 GEMINI_LITE_TIER[0])
    if filtered_size <= LARGE_THRESHOLD * 3:
        return _round_robin_from("mid", GEMINI_MID_TIER,
                                 GEMINI_MID_TIER[0])
    return _round_robin_from("large", GEMINI_LARGE_TIER,
                             GEMINI_LARGE_TIER[0])

def backend_label(model: str) -> str:
    if model == "qwen":
        return "qwen"
    if "pro" in model:
        return "gemini_large"
    if "lite" in model:
        return "gemini_lite"
    return "gemini_mid"

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
                 preferred_model: str, qwen_host: str | None,
                 logs: list) -> str:

    path_str  = str(file_path)
    filtered  = pre_filter(raw)
    use_qwen  = (preferred_model == "qwen")
    max_chars = QWEN_MAX_SECTION if use_qwen else GEMINI_MAX_SECTION

    pct = 100 * (1 - len(filtered) / max(len(raw), 1))
    logs.append(f"pre-filter: {len(raw)}→{len(filtered)} chars ({pct:.0f}% stripped)")

    splitter = _split_class if prompt_type == "class" else _split_tutorial
    sections = splitter(filtered, max_chars)
    logs.append(f"split: {len(sections)} section(s) [preferred={preferred_model}]")

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
            result = call_qwen(prompt, host=qwen_host)
            used   = f"qwen@{qwen_host}"
        else:
            result, used = call_with_fallback(prompt, preferred_model, logs)

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

    def get(self, timeout: float = 3.0) -> dict | None:
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

_print_lock = threading.Lock()
_done_count = 0
_done_lock  = threading.Lock()

# Periodic stats — print every N completions
STATS_EVERY = 20
_stats_counter = 0
_stats_lock    = threading.Lock()

def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)

def _maybe_print_stats():
    global _stats_counter
    with _stats_lock:
        _stats_counter += 1
        do_print = (_stats_counter % STATS_EVERY == 0)
    if do_print:
        _log(health.model_stats_table())

def worker_loop(queue_obj: PriorityJobQueue, worker_id: str,
                total: int, qwen_host: str | None = None):
    """
    qwen_host: if set, this worker uses that specific Qwen host.
               Pass None for Gemini workers (they don't use Qwen directly).
    """
    global _done_count
    while True:
        job = queue_obj.get(timeout=5.0)
        if job is None:
            break

        idx             = job["idx"]
        file_path       = job["file_path"]
        prompt_type     = job["prompt_type"]
        output_path     = job["output_path"]
        preferred_model = job["preferred_model"]
        raw             = job["raw"]

        rel = file_path.relative_to(INPUT_ROOT)
        _log(f"[{idx}/{total}] START  {rel}  [{worker_id}]"
             + (f" preferred={preferred_model}" if preferred_model != "qwen" else ""))

        logs = []
        try:
            content = process_file(raw, prompt_type, file_path,
                                   preferred_model, qwen_host, logs)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            n = len([p for p in content.split("---CHUNK---") if p.strip()])

            with _done_lock:
                _done_count += 1

            _log(f"[{idx}/{total}] DONE   {file_path.name} → {n} chunk(s)")
            for line in logs:
                _log(f"  {line}")

            _maybe_print_stats()

        except Exception as e:
            with _done_lock:
                _done_count += 1
            _log(f"[{idx}/{total}] FAIL   {file_path.name}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    global health

    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-workers",    type=int, default=DEFAULT_QWEN_WORKERS,
                        help=f"Qwen worker threads (default {DEFAULT_QWEN_WORKERS}, "
                             f"split across {len(OLLAMA_HOSTS)} host(s))")
    parser.add_argument("--gemini-workers",  type=int, default=DEFAULT_GEMINI_WORKERS)
    parser.add_argument("--fail-threshold",  type=int, default=DEFAULT_FAIL_THRESHOLD,
                        help="Consecutive failures before blacklisting a model")
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
            pmodel   = initial_model(len(filtered))
            all_jobs.append({
                "file_path":       file_path,
                "prompt_type":     prompt_type,
                "output_path":     output_path,
                "raw":             raw,
                "filtered_size":   len(filtered),
                "preferred_model": pmodel,
            })

    total = len(all_jobs)
    if total == 0:
        print("Nothing to do."); return

    # ── Routing table ───────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"{'FILE':<48} {'SIZE':>7}  PREFERRED")
    print(f"{'─'*62}")
    for job in all_jobs:
        rel = job["file_path"].relative_to(INPUT_ROOT)
        print(f"{str(rel):<48} {job['filtered_size']:>7}  {job['preferred_model']}")
    print(f"{'─'*62}")
    print(f"Total: {total} files | qwen_workers={args.qwen_workers} "
          f"gemini_workers={args.gemini_workers} "
          f"fail_threshold={args.fail_threshold}")
    print(f"Qwen hosts: {OLLAMA_HOSTS}")
    print(f"Fallback chain: {' → '.join(GEMINI_FALLBACK_CHAIN)} → qwen")

    # Show tier distribution
    tier_counts: dict[str, int] = {"large": 0, "mid": 0, "lite": 0, "qwen": 0}
    for job in all_jobs:
        m = job["preferred_model"]
        if m == "qwen":
            tier_counts["qwen"] += 1
        elif any(m == x for x in GEMINI_LARGE_TIER):
            tier_counts["large"] += 1
        elif any(m == x for x in GEMINI_MID_TIER):
            tier_counts["mid"] += 1
        else:
            tier_counts["lite"] += 1
    print(f"Tier distribution: large={tier_counts['large']} "
          f"mid={tier_counts['mid']} lite={tier_counts['lite']} "
          f"qwen={tier_counts['qwen']}")

    if args.dry_run:
        return

    # ── Build priority queues ───────────────────────────────────────
    qwen_q   = PriorityJobQueue()
    gemini_q = PriorityJobQueue()

    for idx, job in enumerate(all_jobs, 1):
        job["idx"]   = idx
        job["total"] = total
        size = job["filtered_size"]
        if job["preferred_model"] == "qwen":
            qwen_q.put(+size, job)     # smallest first
        else:
            gemini_q.put(-size, job)   # largest first

    # ── Launch workers ──────────────────────────────────────────────
    t_start = time.time()
    threads = []

    # Assign Qwen hosts to workers in round-robin
    for i in range(args.qwen_workers):
        host = OLLAMA_HOSTS[i % len(OLLAMA_HOSTS)]
        t = threading.Thread(
            target=worker_loop,
            args=(qwen_q, f"qwen-{i+1}@{host.split('/')[-1]}", total),
            kwargs={"qwen_host": host},
            daemon=True,
        )
        t.start(); threads.append(t)

    for i in range(args.gemini_workers):
        t = threading.Thread(
            target=worker_loop,
            args=(gemini_q, f"gemini-{i+1}", total),
            kwargs={"qwen_host": None},  # Gemini workers pick host at fallback time
            daemon=True,
        )
        t.start(); threads.append(t)

    # ── Wait for completion ─────────────────────────────────────────
    last_stats = time.time()
    while True:
        with _done_lock:
            done = _done_count
        if done >= total:
            break
        # Also print stats periodically by time (every 2 min)
        if time.time() - last_stats > 120:
            _log(health.model_stats_table())
            last_stats = time.time()
        time.sleep(1)

    qwen_q.close()
    gemini_q.close()
    for t in threads:
        t.join(timeout=10)

    elapsed = time.time() - t_start
    print(f"\n{'─'*62}")
    print(f"Done in {elapsed/60:.1f} min — {_done_count}/{total} processed")
    print(health.status_line())
    print(health.model_stats_table())
    print(f"Output in '{OUTPUT_ROOT}/'")


def parse_chunks(chk_content: str) -> list[str]:
    return [p.strip() for p in chk_content.split("---CHUNK---") if p.strip()]


if __name__ == "__main__":
    main()
