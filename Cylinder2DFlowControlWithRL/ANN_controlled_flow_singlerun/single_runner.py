from printind.printind_function import printi, printiv
from env import resume_env

import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
import env
import os
import csv

"""
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../Simulation/")

from Env2DCylinder import Env2DCylinder
"""

printi("resume env")

environment = resume_env(plot=500, dump=10, single_run=True)
deterministic=True

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

restore_path = None
if(os.path.exists("saved_models/checkpoint")):
    restore_path = './saved_models'
restore_path = './best_model'

if restore_path is not None:
    printi("restore the model")
    agent.restore_model(restore_path)
else :
    print('Trained Network not found...')

if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")

def one_run():

    printi("start simulation")
    state = environment.reset()
    environment.render = True

    for k in range(3 * env.nb_actuations):
        #environment.print_state()
        action = agent.act(state, deterministic=deterministic)
        state, terminal, reward = environment.execute(action)
    # just for test, too few timesteps
    # runner.run(episodes=10000, max_episode_timesteps=20, episode_finished=episode_finished)

    data = np.genfromtxt("saved_models/test_strategy.csv", delimiter=";")
    data = data[1:,1:]
    m_data = np.average(data[len(data)//2:], axis=0)
    nb_jets = len(m_data)-4
    # Print statistics
    print("Single Run finished. AvgDrag : {}, AvgRecircArea : {}".format(m_data[1], m_data[2]))

    name = "test_strategy_avg.csv"
    if(not os.path.exists("saved_models")):
        os.mkdir("saved_models")
    if(not os.path.exists("saved_models/"+name)):
        with open("saved_models/"+name, "w") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["Name", "Drag", "Lift", "RecircArea"] + ["Jet" + str(v) for v in range(nb_jets)])
            spam_writer.writerow([environment.simu_name] + m_data[1:].tolist())
    else:
        with open("saved_models/"+name, "a") as csv_file:
            spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow([environment.simu_name] + m_data[1:].tolist())



if not deterministic:
    for _ in range(10):
        one_run()

else:
    one_run()
