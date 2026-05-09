cd ~/visual_wholebody/low-level/legged_gym/scripts
python train.py --headless --exptid 1 --proj_name b2piper-low --task b2piper --sim_device cuda:0 --rl_device cuda:0 --observe_gait_commands


python play.py --exptid 1 --task b2piper --proj_name b2piper-low --checkpoint 20400 --observe_gait_commands