# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics MNIST (AGPL-3.0) is a PyTorch sandbox for trying small convolutional networks, MLPs, and pretrained ResNets on the MNIST digit database and on a few related Ultralytics classification experiments. It is scratch research code — standalone scripts, no package, no tests.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
pip3 install -U -r requirements.txt # numpy, torch, torchvision, opencv-python, h5py, scipy, tqdm
python3 train.py                    # trains ConvNetb for 20 epochs on the bundled data/MNIST*.mat
```

Only `train.py` runs on a clean checkout. `train_resnet.py`, `train_sandd.py`, `train_xview_classes.py`, and `detect.py` need local datasets or checkpoints that are not in the repository, plus `pretrainedmodels` for the ResNet paths. There is no test suite; CI is `.github/workflows/format.yml` (Ruff, docformatter, Prettier, codespell auto-applied to PR branches) and `cla.yml`.

## Architecture

- `models.py` holds every network: `MLP`, `ConvNeta`, `ConvNetb` (the MNIST default), plus `SANDD` and `WAVE2` reused from the [sandd](https://github.com/ultralytics/sandd) and [wave](https://github.com/ultralytics/wave) detector experiments.
- `train.py` is the reference loop: `scipy.io.loadmat` on the bundled `data/MNISTtrain.mat` / `data/MNISTtest.mat`, `create_batches` for batching, Adam, and `patienceStopper` for early stopping. The other `train_*.py` scripts are copies of this loop pointed at different data and models.
- `utils/utils.py` holds the shared helpers (`create_batches`, `split_data`, `normalize`, `patienceStopper`); `utils/torch_utils.py` holds `init_seeds`, `select_device`, `model_info`, `fuse_conv_and_bn`, and `load_classifier` (lazily imports `pretrainedmodels`); `utils/google_utils.py` wraps Google Drive and GCS transfers.
- `Dockerfile` builds on `nvcr.io/nvidia/pytorch` and is used for GPU runs of `train_resnet.py`; the comment block at the bottom is the collection of build/run one-liners.

## Conventions

- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds headers automatically; don't add or revert them manually.
- Hyperparameters are module-level constants inside `main()`, not CLI flags; there is no argparse in `train.py`. Keep that style rather than introducing a config layer.
- The training scripts are deliberately independent copies; put anything shared in `utils/` instead of importing one script from another.
