from __future__ import annotations

import ast
from pathlib import Path

import pytest

from esme_pretrain.launch.budget import require_approval

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILES = sorted([*ROOT.glob("src/esme_pretrain/launch/*.py"), *ROOT.glob("scripts/*.py")])
INVOKE_ATTRS = {
    *("remote", "remote_gen", "spawn", "spawn_map", "map", "starmap", "for_each", "aio"),
    *("listdir", "read_file", "put_file", "batch_upload", "remove_file", "copy_files", "reload"),
}
Function = ast.FunctionDef | ast.AsyncFunctionDef
STATE_WRITES = {
    *("open", "write_text", "write_bytes", "mkdir", "makedirs", "touch", "mkdtemp"),
    *("save", "commit", "init", "login"),
}


def _decorated(fn: Function, name: str) -> bool:
    return any(name in ast.unparse(decorator) for decorator in fn.decorator_list)


def _modal_call_lines(fn: Function) -> list[int]:
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


def _state_write_lines(fn: Function) -> list[int]:
    lines = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in STATE_WRITES or name.startswith("write_"):
            lines.append(node.lineno)
    return lines


def _approval_gate_line(fn: Function) -> int | None:
    # Only a top-level gate holds on every path; one inside a branch can be skipped.
    for node in fn.body:
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


def _module_level_nodes(tree: ast.Module) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        nodes.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return nodes


def _import_time_modal_calls(tree: ast.Module, label: str) -> list[str]:
    # Module code runs on import, before any gate. Only lazy constructors belong there.
    problems = []
    for node in _module_level_nodes(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        root = node.func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        starts_app = node.func.attr in {"run", "deploy"} and getattr(root, "id", "") == "app"
        if node.func.attr in INVOKE_ATTRS or starts_app:
            problems.append(f"{label}:{node.lineno} calls Modal at import, before any gate")
    return problems


def _ungated_launches(source: str, label: str) -> list[str]:
    if "modal" not in source.lower():
        return []
    tree = ast.parse(source)
    problems = _import_time_modal_calls(tree, label)
    for fn in ast.walk(tree):
        if not isinstance(fn, Function) or _decorated(fn, "app.function"):
            continue
        where = f"{label}:{fn.name}"
        if _decorated(fn, "local_entrypoint"):
            # `modal run` starts the app (image build, uploads, Volumes) before the
            # entrypoint runs, so no gate inside it comes first.
            problems.append(f"{where}: launch with python and `app.run()` after the gate")
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

def launch_branch_gated(args):
    if args.check_only:
        if not require_approval(args.approved, run="x", command="x"):
            return 2
    run.spawn()

async def launch_async(args):
    await run.remote.aio()

def launch_generator(args):
    for _ in run.remote_gen():
        pass

def launch_volume(args):
    study_volume.listdir("/")

if modal is not None:
    study_volume.listdir("/")
    with app.run():
        pass

def launch_ok(args):
    if not require_approval(args.approved, run="x", command="x"):
        return 2
    args.output_dir.mkdir(parents=True)
    run.spawn()
"""


def test_the_scan_finds_each_ungated_launch() -> None:
    assert _ungated_launches(UNGATED, "bad.py") == [
        "bad.py:35 calls Modal at import, before any gate",
        "bad.py:34 calls Modal at import, before any gate",
        "bad.py:launch_early calls Modal before `if not require_approval(...): return`",
        "bad.py:launch_writes_first writes state before `if not require_approval(...): return`",
        "bad.py:main: launch with python and `app.run()` after the gate",
        "bad.py:launch_branch_gated calls Modal before `if not require_approval(...): return`",
        "bad.py:launch_async calls Modal before `if not require_approval(...): return`",
        "bad.py:launch_generator calls Modal before `if not require_approval(...): return`",
        "bad.py:launch_volume calls Modal before `if not require_approval(...): return`",
    ]


def test_refusal_tells_the_agent_to_stop_and_ask(capsys: pytest.CaptureFixture[str]) -> None:
    assert not require_approval(False, run="probe", command="uv run python x.py --approved")

    error = capsys.readouterr().err
    assert "spends money" in error
    assert "ask Adam" in error
    assert "uv run python x.py --approved" in error
