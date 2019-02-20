from printind.printind_function import printi, printiv
from env import resume_env, nb_actuations

import os
import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner

"""
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../Simulation/")

from Env2DCylinder import Env2DCylinder
"""

printi("resume env")

environment = resume_env(plot=500, step=50, dump=100)

printi("define network specs")

network_spec = [
    dict(type='dense', size=512),
    dict(type='dense', size=512),
]

printi("define agent")

printiv(environment.states)
printiv(environment.actions)
printiv(network_spec)

agent = PPOAgent(
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        # 10 episodes per update
        batch_size=20,
        # Every 10 episodes
        frequency=20
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=10000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    subsampling_fraction=0.2,
    optimization_steps=25,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

if(os.path.exists('saved_models/checkpoint')):
    restore_path = './saved_models'
else:
    restore_path = None

if restore_path is not None:
    printi("restore the model")
    agent.restore_model(restore_path)

printi("define runner")

runner = Runner(agent=agent, environment=environment)


def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))

    printi("save the mode")

    name_save = "./saved_models/ppo_model"
    # NOTE: need to check if should create the dir
    r.agent.save_model(name_save, append_timestep=False)

    # show for plotting
    # r.environment.show_control()
    # r.environment.show_drag()

    # print(sess.run(tf.global_variables()))

    return True

# Start learning
printi("start learning")
# good parameters that gave learning
# runner.run(episodes=10000, max_episode_timesteps=80, episode_finished=episode_finished)
runner.run(episodes=2000, max_episode_timesteps=nb_actuations, episode_finished=episode_finished)
# just for test, too few timesteps
# runner.run(episodes=10000, max_episode_timesteps=20, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
