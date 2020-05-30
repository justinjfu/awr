import os
import doodad
from doodad.wrappers.sweeper import launcher
import d4rl
import d4rl import infos
import gym

ENVS = [
    'maze2d-umaze-v1',
    'maze2d-medium-v1',
    'maze2d-large-v1',
    'pen-human-v0',
    'pen-cloned-v0',
    'pen-expert-v0',
    'hammer-human-v0',
    'hammer-cloned-v0',
    'hammer-expert-v0',
    'relocate-human-v0',
    'relocate-cloned-v0',
    'relocate-expert-v0',
    'door-human-v0',
    'door-cloned-v0',
    'door-expert-v0',
    'halfcheetah-random-v0',
    'halfcheetah-medium-v0',
    'halfcheetah-expert-v0',
    'halfcheetah-medium-replay-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-random-v0',
    'walker2d-medium-v0',
    'walker2d-expert-v0',
    'walker2d-medium-replay-v0',
    'walker2d-medium-expert-v0',
    'hopper-random-v0',
    'hopper-medium-v0',
    'hopper-expert-v0',
    'hopper-medium-replay-v0',
    'hopper-medium-expert-v0',
    'antmaze-umaze-v0',
    'antmaze-umaze-diverse-v0',
    'antmaze-medium-play-v0',
    'antmaze-medium-diverse-v0',
    'antmaze-large-play-v0',
    'antmaze-large-diverse-v0',
    'kitchen-complete-v0',
    'kitchen-partial-v0',
    'kitchen-mixed-v0',
]

FLOW_ENVS = [
    'flow-ring-random-v0',
    'flow-ring-controller-v0',
    'flow-merge-random-v0',
    'flow-merge-controller-v0',
]
ENVS.extend(FLOW_ENVS)

mounts = []

mounts.append(doodad.MountLocal(local_dir='~/code/awr',
                                mount_point='/code/awr', pythonpath=True))
mounts.append(doodad.MountLocal(local_dir='~/code/d4rl',
                              mount_point='/code/d4rl', pythonpath=True))


sweeper = launcher.DoodadSweeper(
    mounts=mounts,
    docker_img='justinfu/awr:0.1',
    gcp_bucket_name='justin-doodad',
    gcp_image='ubuntu-1804-docker-gpu',
    gcp_project='qlearning000'
)

flow_sweeper = launcher.DoodadSweeper(
    mounts=mounts,
    docker_img='justinfu/awr_flow:0.1',
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
    _script = 'scripts/run_script.py'
    if env_name.startswith('flow'):
        _swp = flow_sweeper
        _script = 'scripts/run_flow.py'

    _swp.run_sweep_gcp(
        target='scripts/run_script.py',
        params=params,
        extra_mounts=[data_mount],
        instance_type='n1-standard-4',
        log_prefix='awr_d4rl'
    )

