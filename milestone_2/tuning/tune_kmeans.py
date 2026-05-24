import subprocess
import os
import re

PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "bin", "python"))

K_values = list(range(1, 18))
N_INIT = 50


def run(K, test=False):
    cmd = [
        PY, "main.py",
        "--method", "kmeans",
        "--task", "classification",
        "--K", str(K),
        "--n_init", str(N_INIT),
        "--data_path", "../milestone_1/data/features.npz",
    ]
    if test:
        cmd.append("--test")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def parse(out):
    split = "Test" if "Test accuracy" in out else "Validation"
    acc = re.search(rf"{split} accuracy:\s*([\d.]+)%", out)
    f1 = re.search(rf"{split} F1:\s*([\d.]+)", out)
    return (float(acc.group(1)) if acc else None,
            float(f1.group(1)) if f1 else None)


results = []
for K in K_values:
    config_str = f"K={K} | n_init={N_INIT}"
    out = run(K)
    acc, f1 = parse(out)
    print(f"-> {config_str} | Val Acc={acc}% | Val F1={f1}")
    if f1 is not None:
        results.append((f1, acc, K, config_str))

results.sort(key=lambda x: x[0], reverse=True)

if results:
    _, _, best_K, best_str = results[0]
    out = run(best_K, test=True)
    acc_t, f1_t = parse(out)
    print(f"\nBest K-Means hyperparameters were ({best_str}) "
          f"and yielded a TEST accuracy of {acc_t}% and a TEST F1 of {f1_t}")
