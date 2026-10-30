from __future__ import annotations

import ast
from pathlib import Path

import pytest

from esme_pretrain.launch.budget import require_approval

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILES = sorted([*ROOT.glob("src/esme_pretrain/launch/*.py"), *ROOT.glob("scripts/*.py")])
INVOKE_ATTRS = {"remote", "spawn", "map", "starmap"}
STATE_WRITES = {
    *("open", "write_text", "write_bytes", "mkdir", "makedirs", "touch", "mkdtemp"),
    *("save", "commit", "init", "login"),
}


def _decorated(fn: ast.FunctionDef, name: str) -> bool:
    return any(name in ast.unparse(decorator) for decorator in fn.decorator_list)


def _modal_call_lines(fn: ast.FunctionDef) -> list[int]:
    lines = []
    for statement in fn.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            invokes = isinstance(node.func, ast.Attribute) and node.func.attr in INVOKE_ATTRS
            if invokes or (isinstance(root, ast.Name) and root.id in {"modal", "app"}):
                lines.append(node.lineno)
    return lines


def _state_write_lines(fn: ast.FunctionDef) -> list[int]:
    lines = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in STATE_WRITES or name.startswith("write_"):
            lines.append(node.lineno)
    return lines


def _approval_gate_line(fn: ast.FunctionDef) -> int | None:
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Call)
            and ast.unparse(test.operand.func) == "require_approval"
            and isinstance(node.body[0], ast.Return)
        ):
            return node.lineno
    return None


def _ungated_launches(source: str, label: str) -> list[str]:
    problems = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef) or _decorated(fn, "app.function"):
            continue
        where = f"{label}:{fn.name}"
        if _decorated(fn, "local_entrypoint"):
            calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
            if "launch" not in calls or _modal_call_lines(fn):
                problems.append(f"{where} must only delegate to the gated launch()")
            continue
        modal_lines = _modal_call_lines(fn)
        gate = _approval_gate_line(fn)
        if modal_lines and (gate is None or gate > min(modal_lines)):
            problems.append(f"{where} calls Modal before `if not require_approval(...): return`")
        elif (
            modal_lines and gate is not None and any(line < gate for line in _state_write_lines(fn))
        ):
            problems.append(f"{where} writes state before `if not require_approval(...): return`")
    return problems


def test_every_paid_launch_asks_for_approval_before_modal() -> None:
    problems = [
        problem
        for path in LAUNCH_FILES
        for problem in _ungated_launches(
            path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))
        )
    ]

    assert not problems, problems


UNGATED = """
def launch_early(args):
    run.remote()
    if not require_approval(args.approved, run="x", command="x"):
        return 2

def launch_writes_first(args):
    args.output_dir.mkdir(parents=True)
    if not require_approval(args.approved, run="x", command="x"):
        return 2
    run.spawn()

@app.local_entrypoint()
def main():
    run.remote()

def launch_ok(args):
    if not require_approval(args.approved, run="x", command="x"):
        return 2
    args.output_dir.mkdir(parents=True)
    run.spawn()
"""


def test_the_scan_finds_each_ungated_launch() -> None:
    assert _ungated_launches(UNGATED, "bad.py") == [
        "bad.py:launch_early calls Modal before `if not require_approval(...): return`",
        "bad.py:launch_writes_first writes state before `if not require_approval(...): return`",
        "bad.py:main must only delegate to the gated launch()",
    ]


def test_refusal_tells_the_agent_to_stop_and_ask(capsys: pytest.CaptureFixture[str]) -> None:
    assert not require_approval(False, run="probe", command="uv run python x.py --approved")

    error = capsys.readouterr().err
    assert "spends money" in error
    assert "ask Adam" in error
    assert "uv run python x.py --approved" in error
