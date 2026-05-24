import subprocess
import os
import re

PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "bin", "python"))

tasks = ["classification", "regression"]
learning_rates = [1e-2, 5e-3]
architectures = ["32", "128,64", "256,128"]
activations = ["relu"]
max_iters_list = [100, 200]


def run(task, lr, arch, act, iters, test=False):
    cmd = [
        PY, "main.py",
        "--method", "mlp",
        "--task", task,
        "--lr", str(lr),
        "--mlp_dim", arch,
        "--activation", act,
        "--max_iters", str(iters),
        "--data_path", "../milestone_1/data/features.npz",
    ]
    if test:
        cmd.append("--test")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def parse_classif(out):
    split = "Test" if "Test accuracy" in out else "Validation"
    acc = re.search(rf"{split} accuracy:\s*([\d.]+)%", out)
    f1 = re.search(rf"{split} F1:\s*([\d.]+)", out)
    return (float(acc.group(1)) if acc else None,
            float(f1.group(1)) if f1 else None)


def parse_reg(out):
    split = "Test" if "Test MSE" in out else "Validation"
    mse = re.search(rf"{split} MSE:\s*([\d.]+)", out)
    return float(mse.group(1)) if mse else None


classif_results = []
reg_results = []

for task in tasks:
    for lr in learning_rates:
        for arch in architectures:
            for act in activations:
                for iters in max_iters_list:
                    config = {"task": task, "lr": lr, "arch": arch, "act": act, "iters": iters}
                    config_str = f"Task={task} | LR={lr} | Dims={arch} | Act={act} | Iters={iters}"
                    out = run(task, lr, arch, act, iters)
                    if task == "classification":
                        acc, f1 = parse_classif(out)
                        print(f"-> {config_str} | Val Acc={acc}% | Val F1={f1}")
                        if f1 is not None:
                            classif_results.append((f1, acc, config, config_str))
                    else:
                        mse = parse_reg(out)
                        print(f"-> {config_str} | Val MSE={mse}")
                        if mse is not None:
                            reg_results.append((mse, config, config_str))

classif_results.sort(key=lambda x: x[0], reverse=True)

if classif_results:
    _, _, best_c, best_c_str = classif_results[0]
    out = run(best_c["task"], best_c["lr"], best_c["arch"], best_c["act"], best_c["iters"], test=True)
    acc_t, f1_t = parse_classif(out)
    print(f"Best classification hyperparameters were ({best_c_str}) "
          f"and yielded a TEST accuracy of {acc_t}% and a TEST F1 of {f1_t}")

if reg_results:
    _, best_r, best_r_str = reg_results[0]
    out = run(best_r["task"], best_r["lr"], best_r["arch"], best_r["act"], best_r["iters"], test=True)
    mse_t = parse_reg(out)
    print(f"Best regression hyperparameters were ({best_r_str}) "
          f"and yielded a TEST MSE of {mse_t}")
