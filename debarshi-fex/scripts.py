import os
import time
import random
import argparse

def run_conservative():
    print("=== Conservative Controller Training ===")
    os.makedirs("logs", exist_ok=True)

    idx = 0
    for i in range(5):
        for dim in [42, 48, 54]:
            idx += 1
            session_name = f"train_dim{dim}_trial{i}"
            log_file = f"logs/train_dim{dim}_trial{i}.log"
            
            command = (
                f'screen -dmS {session_name} bash -c '
                f'"python controller_conservative.py '
                f'--epoch 2000 --bs 10 --greedy 0.1 '
                f'--ckpt t_range_0_1_2ksearch_int20k_bd4kDim{dim} '
                f'--tree depth2_sub --random_step 3 --lr 0.002 '
                f'--dim {dim} --base 200000 --left -1 --right 1 '
                f'--domainbs 20000 --bdbs 4000 > {log_file} 2>&1"'
            )
            
            print(f"Starting training: dim={dim}, trial={i}")
            print(f"Session name: {session_name}")
            print(f"Log file: {log_file}")
            
            os.system(command)
            time.sleep(500)

def run_poisson():
    print("=== Poisson Controller Training ===")
    os.makedirs("logs_poisson", exist_ok=True)

    for i in range(6):
        for dim in [10, 20, 30, 40, 50]:
            gpu = random.randint(0, 4)
            session_name = f"poisson_dim{dim}_trial{i}_gpu{gpu}"
            log_file = f"logs_poisson/poisson_dim{dim}_trial{i}_gpu{gpu}.log"
            
            command = (
                f'screen -dmS {session_name} bash -c '
                f'"python controller_poisson.py '
                f'--epoch 1000 --bs 10 --greedy 0.1 '
                f'--ckpt Dim{dim} --tree depth2_sub --random_step 3 '
                f'--lr 0.002 --dim {dim} --base 200000 --left -1 --right 1 '
                f'--domainbs 5000 --bdbs 1000 > {log_file} 2>&1"'
            )
            
            print(f"Starting: dim={dim}, trial={i}, gpu={gpu}")
            print(f"Session: {session_name}")
            print(f"Log: {log_file}")
            
            os.system(command)
            time.sleep(100)
    
def run_schrodinger():
    print("=== Schrodinger Controller Training ===")
    os.makedirs("logs_schrodinger", exist_ok=True)

    for i in range(2):
        for dim in [6, 12, 18, 24, 30]:
            gpu = random.randint(1, 9)
            session_name = f"schrodinger_dim{dim}_trial{i}_gpu{gpu}"
            log_file = f"logs_schrodinger/schrodinger_dim{dim}_trial{i}_gpu{gpu}.log"
            
            command = (
                f'screen -dmS {session_name} bash -c '
                f'"python controller_cubic_sh_firstdeflation_thenintegral.py '
                f'--epoch 1000 --bs 10 --greedy 0.1 '
                f'--tree depth2_sub --random_step 3 --lr 0.002 '
                f'--dim {dim} --base 200000 --left -1 --right 1 '
                f'--domainbs 2000 --intbs 10000 > {log_file} 2>&1"'
            )
            
            print(f"Starting Schrodinger training: dim={dim}, trial={i}, gpu={gpu}")
            print(f"Session name: {session_name}")
            print(f"Log file: {log_file}")
            
            os.system(command)
            time.sleep(100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training scripts")
    parser.add_argument("script", choices=["conservative", "poisson"], 
                        help="Which script to run")
    args = parser.parse_args()

    if args.script == "conservative":
        run_conservative()
    elif args.script == "poisson":
        run_poisson()
    elif args.script == "schrodinger":
        run_schrodinger()