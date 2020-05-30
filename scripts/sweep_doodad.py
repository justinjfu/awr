import os
import doodad
from doodad.wrappers.sweeper import launcher
import d4rl
import d4rl import infos
import gym

ENVS = [
    #'kitchen-complete-v0',
    #'kitchen-partial-v0',
    #'kitchen-undirected-v0'
    #'maze2d-umaze-v1',
    #'maze2d-medium-v1',
    #'maze2d-large-v1',
    #'maze2d-eval-umaze-v1',
    #'maze2d-eval-medium-v1',
    #'maze2d-eval-large-v1',
    #'pen-demos-v0',
    #'hammer-demos-v0',
    #'relocate-demos-v0',
    #'door-demos-v0',
]

FLOW_ENVS = [
    'flow-ring-random-v0',
    'flow-ring-controller-v0',
    'flow-merge-random-v0',
    'flow-merge-controller-v0',
]

mounts = []

mounts.append(doodad.MountLocal(local_dir='~/code/awr',
                                mount_point='/code/awr', pythonpath=True))
mounts.append(doodad.MountLocal(local_dir='~/code/d4rl',
                              mount_point='/code/d4rl', pythonpath=True))


sweeper = launcher.DoodadSweeper(
    mounts=mounts,
    docker_img='justinfu/rlkit:0.4',
    gcp_bucket_name='justin-doodad',
    gcp_image='ubuntu-1804-docker-gpu',
    gcp_project='qlearning000'
)

flow_sweeper = launcher.DoodadSweeper(
    mounts=mounts,
    docker_img='justinfu/rlkit_sumo:0.3',
    gcp_bucket_name='justin-doodad',
    gcp_image='ubuntu-1804-docker-gpu',
    gcp_project='qlearning000'
)


for env_name in ENVS:
    env = gym.make(env_name)
    _, dataset = os.path.split(env.dataset_filepath)
    dirname, _ = os.path.splitext(dataset)

    params = {
        'env_name': [env_name],
        'seed': range(3),
    }

    data_mount = doodad.MountLocal(local_dir='~/.d4rl/rlkit/%s' % dirname,
                              mount_point='/datasets')

    _swp = sweeper
    if env_name.startswith('flow'):
        _swp = flow_sweeper

    _swp.run_sweep_gcp(
        target='scripts/run_script.py',
        params=params,
        extra_mounts=[data_mount],
        instance_type='n1-standard-4',
        log_prefix='awr_d4rl'
    )

