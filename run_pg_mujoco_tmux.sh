# /bin/zsh

# Create 4 panes in a new window
tmux new-window
tmux split-window -h
tmux select-pane -t 1
tmux split-window
tmux select-pane -t 0
tmux split-window

# Run command in specific pane
tmux send-keys -t 0 \
    "workon py37 && " \
    "cd /data/hanjl/rl-exercise/ && " \
    "CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --exp_name PPO2 --allow_eval" \
    Enter
sleep 2
tmux send-keys -t 1 \
    "workon py37 && " \
    "cd /data/hanjl/rl-exercise/ && " \
    "CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --exp_name PPO2 --allow_eval" \
    Enter
sleep 2
tmux send-keys -t 2 \
    "workon py37 && " \
    "cd /data/hanjl/rl-exercise/ && " \
    "CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --exp_name PPO2 --allow_eval" \
    Enter
sleep 2

# Create 4 panes in a new window
tmux new-window
tmux split-window -h
tmux select-pane -t 1
tmux split-window
tmux select-pane -t 0
tmux split-window

# Run command in specific pane
tmux send-keys -t 0 \
    "workon py37 && " \
    "cd /data/hanjl/rl-exercise/ && " \
    "CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --exp_name PPO2 --allow_eval" \
    Enter
sleep 2
tmux send-keys -t 1 \
    "workon py37 && " \
    "cd /data/hanjl/rl-exercise/ && " \
    "CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --exp_name PPO2 --allow_eval" \
    Enter
sleep 2
tmux send-keys -t 2 \
    "workon py37 && " \
    "cd /data/hanjl/rl-exercise/ && " \
    "CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --exp_name PPO2 --allow_eval" \
    Enter
sleep 2