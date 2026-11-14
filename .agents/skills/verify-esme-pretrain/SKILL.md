---
name: verify-esme-pretrain
description: "Drives the real esme-pretrain training and Modal launch entrypoints on CPU, offline, with no spend. Use to run, launch, drive, or verify esme-pretrain, or to prove a change to it works."
---

# verify-esme-pretrain

The fast lane (`make check`) proves the code is consistent. This skill proves behavior. It runs the real training entrypoint on CPU, then resolves every paid launch plan and checks that each one refuses without `--approved`.

## Run

```bash
make verify                    # drives that cover files changed against main; the train drive when none match
make verify ARGS=all           # every drive, about 6 to 14 seconds
make verify ARGS="train tokenizer"
make verify ARGS=list          # each drive, what it proves, and the paths it covers
make verify ARGS=control       # re-runs train with a loss threshold it cannot reach; exits 0 only when FAIL is observed
```

The helper is `.agents/skills/verify-esme-pretrain/verify`. It runs children with `.venv/bin/python`, so run `make install` first.

## Reading Results

- Each drive prints `PASS|FAIL <drive> <seconds>s <detail>`. A FAIL detail names what the drive expected and what it observed. A drive that crashes is a FAIL.
- The run fails when it changes the checkout: a tracked, untracked, or ignored file, the index, HEAD, or `.git/config`. Only `.verify/` and `__pycache__/` may change.
- Evidence is in `.verify/runs/<utc>-<sha7>[-dirty]/`. Each drive has `<step>.cmd`, `.stdout`, `.stderr`, and `.exit` files, plus the rehearsal run directory under `artifacts/` (metrics, checkpoints, tokenizer). `summary.json` holds the overall `result`, `checkout_changed`, the drive results, and the git sha.
- `tripwire fired` means a child tried a non-loopback network connection or called `modal`, `wandb`, or `hf`. Treat it as a spend or data-leak risk, not a flake.

## Isolation

Each run gets a temp HOME and temp XDG, HF, W&B, and inductor dirs. It sets `HF_HUB_OFFLINE=1` and `WANDB_MODE=disabled`, removes `MODAL_*`, `WANDB_*`, `HF_*`, and secret-named variables, and points `MODAL_SERVER_URL` at a closed loopback port. A `sitecustomize.py` refuses non-loopback sockets, and PATH stubs refuse `modal`, `wandb`, and `hf`.

## Limits

- The drives never reach a GPU, Modal, FineWeb-Edu, or W&B. A real GPU run is proven only by an approved launch from its run card.
- The training drive uses synthetic token ids and a 2-layer model. It proves the loop, checkpoint, and resume contract, not throughput or model quality.
- The launch-plans drive proves the plans resolve and the gate refuses. It does not prove Modal image builds or volume layout.

## Gotchas

- The config validator only accepts `artifacts.output_dir` under `runs/pretrain-214m-b200/`. The drives write to a per-run `runs/pretrain-214m-b200/verify-*` directory and move it into the evidence dir.
- The throughput-bench plan resolves its pinned `--before-commit`, so it needs full git history. A shallow clone fails that drive.
- `make slow` is the slow pytest lane, not this skill.
