import os
import time
import random

# Script 1: Poisson Controller
print("=== Poisson Controller Training ===")

# Create logs directory
os.makedirs("logs_poisson", exist_ok=True)

for i in range(6):
    for dim in [10, 20, 30, 40, 50]:
        gpu = random.randint(0,4)
        
        # Create unique session and log names
        session_name = f"poisson_dim{dim}_trial{i}_gpu{gpu}"
        log_file = f"logs_poisson/poisson_dim{dim}_trial{i}_gpu{gpu}.log"
        
        # Fixed screen command with proper syntax and logging
        command = f'screen -dmS {session_name} bash -c "python controller_poisson.py --epoch 1000 --bs 10 --greedy 0.1 --ckpt Dim{dim} --tree depth2_sub --random_step 3 --lr 0.002 --dim {dim} --base 200000 --left -1 --right 1 --domainbs 5000 --bdbs 1000 > {log_file} 2>&1"'
        
        print(f"Starting: dim={dim}, trial={i}, gpu={gpu}")
        print(f"Session: {session_name}")
        print(f"Log: {log_file}")
        
        os.system(command)
        time.sleep(100)

print("\n" + "="*50)