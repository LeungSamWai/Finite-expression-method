import os
import time
import random

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

idx = 0

for i in range(5):
    for dim in [42, 48, 54]:
        idx += 1
        
        # Create unique session name and log file (removed GPU reference)
        session_name = f"train_dim{dim}_trial{i}"
        log_file = f"logs/train_dim{dim}_trial{i}.log"
        
        # Removed --gpu argument
        command = f'screen -dmS {session_name} bash -c "python controller_conservative.py --epoch 2000 --bs 10 --greedy 0.1 --ckpt t_range_0_1_2ksearch_int20k_bd4kDim{dim} --tree depth2_sub --random_step 3 --lr 0.002 --dim {dim} --base 200000 --left -1 --right 1 --domainbs 20000 --bdbs 4000 > {log_file} 2>&1"'
        
        print(f"Starting training: dim={dim}, trial={i}")
        print(f"Session name: {session_name}")
        print(f"Log file: {log_file}")
        
        os.system(command)
        time.sleep(500)