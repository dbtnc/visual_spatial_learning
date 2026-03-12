#!/usr/bin/env python3

import subprocess
import sys

GRID_SIZE = 32
RUN_SIZE = 100
COV_MODELS = ["exponential"]

run_size = RUN_SIZE + 1
cov_m = COV_MODELS[0]
distribution = sys.argv[1]
phi_p = float(sys.argv[2])
n = float(sys.argv[3])

# How to run:
#parallel --jobs 16 'python run_par.py gaussian-nugget-03_v2 {} {}' ::: 0.{1..8} ::: 0.{2..8}

print(f"Phi: {phi_p} --- Percentage of points: {n}")

folder_distribution_name = f"a_run_{RUN_SIZE}_trials_{cov_m}_model_{distribution}_distribution_{int(phi_p * 100)}_phi"
folder_experiment_name = f"{folder_distribution_name}/a_run_{int(n * 100)}"
subprocess.run(["mkdir", "-p", folder_experiment_name])
subprocess.run(["touch", f"{folder_experiment_name}/krige_ok_failed.txt"])
subprocess.run(["truncate", "-s", "0", f"{folder_experiment_name}/krige_ok_failed.txt"])

seed_i = 1
c_wcl = 0
converged = False
for i in range(1, run_size):
    save_path = f"{folder_experiment_name}/true_grid_{i:03d}.csv"
    train_path = f"{folder_experiment_name}/true_grid_{i:03d}_train.csv"
    test_path = f"{folder_experiment_name}/true_grid_{i:03d}_test.csv"
    krige_res_path = f"{folder_experiment_name}/krige_ok_grid_{i:03d}.csv"

    while not converged:
        if distribution.startswith("ns-"):
            subprocess.run(["Rscript", "gen_nonstationary.R", str(GRID_SIZE), save_path, str(seed_i), distribution, cov_m, str(phi_p)])
        else:
            subprocess.run(["Rscript", "gen_field.R", str(GRID_SIZE), save_path, str(seed_i), distribution, cov_m, str(phi_p)])
        
        krige_ok_failed_path = f"{folder_experiment_name}/krige_ok_failed.txt"
        subprocess.run(["Rscript", "split_krige.R", save_path, train_path, test_path, krige_res_path, str(n), krige_ok_failed_path, str(seed_i)])
        wcl = int(subprocess.check_output(["wc", "-l", krige_ok_failed_path]).split()[0])
        if c_wcl == wcl:
            converged = True
        else:
            c_wcl += 1
            seed_i += 1

    seed_i += 1
    converged = False

print(f"Experiments: {run_size - 1} - Number of iterations: {seed_i - 1}")
