import subprocess

# Hyperparameter grids to test
tasks = ["classification", "regression"]
learning_rates = [1e-2, 1e-3, 1e-4]
architectures = ["64,32", "128,64", "32"]
activations = ["sigmoid", "relu"]

print("Starting automatic grid search for best hyperparameters")

with open("tuning_results.txt", "w", encoding="utf-8") as f:
    f.write("MLP Tuning Results\n\n")

for task in tasks:
    for lr in learning_rates:
        for arch in architectures:
            for act in activations:
                print(f"-> Test : Task={task} | LR={lr} | Dims={arch} | Act={act}")
                
                cmd = [
                    "python", "main.py",
                    "--method", "mlp",
                    "--task", task,
                    "--lr", str(lr),
                    "--mlp_dim", arch,
                    "--activation", act,
                    "--max_iters", "30",
                    "--data_path", "../milestone_1/data/features.npz"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
                
                with open("tuning_results.txt", "a", encoding="utf-8") as f:
                    f.write(f"\nConfiguration: Task={task} | LR={lr} | Dims={arch} | Act={act}\n")
                    f.write(result.stdout)
                    if result.stderr:
                        f.write(f"Error")
                    f.write("\n ================ \n")

print("\nFinished !!")