import os
import doodad
from doodad.wrappers.sweeper import launcher
from d4rl import infos
import gym

#env_name = 'maze2d-eval-umaze-v1'
env_name = 'maze2d-umaze-v1'
_, dataset = os.path.split(infos.DATASET_URLS[env_name])
dirname, _ = os.path.splitext(dataset)

mounts = []
mounts.append(doodad.MountLocal(local_dir='~/code/awr',
                              mount_point='/code/awr', pythonpath=True, filter_dir=('data', '.git', 'awr_env')))
mounts.append(doodad.MountLocal(local_dir='~/code/d4rl',
                              mount_point='/code/d4rl', pythonpath=True, filter_dir=('data', '.git', 'scripts')))
mounts.append(doodad.MountLocal(local_dir='~/.d4rl/rlkit/%s' % dirname,
                              mount_point='/datasets'))
mounts.append(doodad.MountLocal(local_dir='/data/doodad/awr',
                                mount_point='/data', output=True))

gcp_launcher = doodad.GCPMode(
    gcp_bucket='justin-doodad',
    gcp_log_path='doodad/logs/bear',
    gcp_project='qlearning000',
    instance_type='n1-standard-1',
    zone='us-west1-a',
    gcp_image='ubuntu-1804-docker-gpu',
    gcp_image_project='qlearning000'
)
local_launcher = doodad.LocalMode()


doodad.run_python(
    target='scripts/run_script.py',
    mode=local_launcher,
    mounts=mounts,
    docker_image='justinfu/awr:0.1',
    verbose=True,
    cli_args='--output_dir=/data --env=' + env_name
)

