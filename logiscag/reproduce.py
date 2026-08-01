"""
One-command reproduction entry point.

    python -m logiscag.reproduce --verify
        No-SDV verification path. Runs the constraint-ladder test (asserts the
        cumulative strictness gradient none < temporal < moderate < strict)
        and a smoke sweep on proxy data with the mock synthesiser. Takes
        seconds; proves the wiring is correct without needing SDV installed.

    python -m logiscag.reproduce --real path/to/DataCoSupplyChainDataset.csv
        Real path (requires `pip install sdv`). Adapts the DataCo CSV to the
        canonical schema and runs the paper's Table 1 headline sweep
        (CopulaGAN, levels none/temporal/strict+reject) end to end, writing
        sweep_summary.csv, sweep_significance.csv, trtr_baseline.json, and the
        frontier figure to --out.

        Defaults to a fast, reduced-fidelity demo (1 seed, 20 epochs, 2000
        synthetic rows) so the command completes in minutes. Pass --full to
        match the paper's exact full-fidelity protocol (5 seeds, 100 epochs,
        10000 synthetic rows, ~2-3 hours on CPU) -- this is the configuration
        that produced Table 1.

These two paths are intentionally separate commands: --verify never imports
SDV and never touches real data; --real always requires both.
"""

import argparse
import os
import subprocess
import sys


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _subprocess_env():
    """Environment for child processes, with the repo root on PYTHONPATH.

    The helper scripts invoked below (`tests/test_constraint_ladder.py`,
    `privacy_utility_sweep.py`) import `pipeline` and `dataco_adapter`, which
    live at the repo root. When the package is installed (`pip install -e .`)
    those resolve normally, but a reviewer running this straight from a fresh
    clone has no such install -- and a bare `subprocess.run([sys.executable,
    script])` does not inherit the parent's sys.path. Exporting PYTHONPATH
    makes the verification path work in both situations.
    """
    env = os.environ.copy()
    root = _repo_root()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    return env


def _run_verify():
    print("=== LogiSCAG verification path (no SDV) ===\n")
    print("[1/2] Constraint-ladder test (asserts none < temporal < moderate < strict)...")
    repo_root = _repo_root()
    r1 = subprocess.run(
        [sys.executable, os.path.join(repo_root, "tests", "test_constraint_ladder.py")],
        env=_subprocess_env(),
    )
    if r1.returncode != 0:
        print("\n[FAIL] constraint-ladder test failed.")
        return r1.returncode

    print("\n[2/2] Smoke sweep on proxy data (mock synthesiser if SDV is absent)...")
    r2 = subprocess.run(
        [sys.executable, os.path.join(repo_root, "privacy_utility_sweep.py"),
         "--smoke", "--mia", "--out", "outputs/reproduce_verify"],
        env=_subprocess_env(),
    )
    if r2.returncode != 0:
        print("\n[FAIL] smoke sweep failed.")
        return r2.returncode

    print("\n[OK] Verification path passed. Wiring is correct.")
    print("Outputs written to outputs/reproduce_verify/")
    return 0


def _run_real(csv_path, out_dir, full):
    print("=== LogiSCAG real path (requires SDV) ===\n")
    try:
        import sdv  # noqa: F401
    except ImportError:
        print("[FAIL] SDV is not installed. Run: pip install sdv")
        return 1

    from logiscag.adapters.dataco import load_dataco, dataco_to_canonical

    print(f"[1/2] Adapting {csv_path} to the canonical schema...")
    dc = load_dataco(csv_path)
    canon = dataco_to_canonical(dc)
    print(f"  {len(dc):,} raw rows -> {len(canon):,} canonical rows")
    canon_path = os.path.join(os.path.dirname(out_dir) or ".", "_reproduce_canonical.csv")
    os.makedirs(out_dir, exist_ok=True)
    canon.to_csv(canon_path, index=False)

    seeds, epochs, n_synth = (5, 100, 10000) if full else (1, 20, 2000)
    print(f"\n[2/2] Running the Table 1 headline sweep (CopulaGAN, "
          f"none/temporal/strict+reject, {seeds} seed(s), {epochs} epochs)...")
    if full:
        print("  --full requested: this matches the paper's exact protocol and takes "
              "roughly 2-3 hours on CPU.")
    repo_root = _repo_root()
    cmd = [sys.executable, os.path.join(repo_root, "privacy_utility_sweep.py"),
           "--real", canon_path, "--architectures", "CopulaGAN",
           "--levels", "none", "temporal", "strict+reject",
           "--seeds", str(seeds), "--n-synth", str(n_synth), "--epochs", str(epochs),
           "--out", out_dir]
    r = subprocess.run(cmd, env=_subprocess_env())
    if r.returncode != 0:
        print("\n[FAIL] sweep failed.")
        return r.returncode

    print(f"\n[OK] Real path complete. Outputs written to {out_dir}/")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true", help="no-SDV verification path")
    group.add_argument("--real", type=str, metavar="CSV_PATH",
                        help="real path: path to DataCoSupplyChainDataset.csv")
    ap.add_argument("--out", type=str, default="outputs/reproduce_real",
                     help="output directory for the real path (default: outputs/reproduce_real)")
    ap.add_argument("--full", action="store_true",
                     help="match the paper's exact full-fidelity protocol (real path only)")
    args = ap.parse_args()

    if args.verify:
        sys.exit(_run_verify())
    else:
        sys.exit(_run_real(args.real, args.out, args.full))


if __name__ == "__main__":
    main()
