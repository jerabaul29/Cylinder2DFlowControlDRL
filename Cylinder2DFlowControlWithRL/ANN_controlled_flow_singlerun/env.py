"""Resume and use the environment in the configuration version number 2.
"""

import sys
import os
import shutil
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from Env2DCylinder import Env2DCylinder
import numpy as np
from dolfin import Expression
from printind.printind_function import printi, printiv
import math

import os
cwd = os.getcwd()

nb_actuations = 80 #Nombre d'actuations du reseau de neurones par episode

def resume_env(plot=False,
               step=50,
               dump=100,
               remesh=False,
               random_start=False,
               single_run=False):
    # ---------------------------------------------------------------------------------
    # the configuration version number 1

    simulation_duration = 2.0 #duree en secondes de la simulation
    dt=0.0005

    root = 'mesh/turek_2d'
    if(not os.path.exists('mesh')):
        os.mkdir('mesh')

    jet_angle = 0

    geometry_params = {'output': '.'.join([root, 'geo']),
                    'length': 2.2,
                    'front_distance': 0.05 + 0.15,
                    'bottom_distance': 0.05 + 0.15,
                    'jet_radius': 0.05,
                    'width': 0.41,
                    'cylinder_size': 0.01,
                    'coarse_size': 0.1,
                    'coarse_distance': 0.5,
                    'box_size': 0.05,
                    'jet_positions': [90+jet_angle, 270-jet_angle],
                    'jet_width': 10,
                    'clscale': 0.25,
                    'template': '../geometry_2d.template_geo',
                    'remesh': remesh}

    def profile(mesh, degree):
        bot = mesh.coordinates().min(axis=0)[1]
        top = mesh.coordinates().max(axis=0)[1]
        print bot, top
        H = top - bot

        Um = 1.5

        return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                        '0'), bot=bot, top=top, H=H, Um=Um, degree=degree)

    flow_params = {'mu': 1E-3,
                  'rho': 1,
                  'inflow_profile': profile}

    solver_params = {'dt': dt}

    list_position_probes = []

    positions_probes_for_grid_x = [0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    positions_probes_for_grid_y = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))

    positions_probes_for_grid_x = [-0.025, 0.0, 0.025, 0.05]
    positions_probes_for_grid_y = [-0.15, -0.1, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))

    list_radius_around = [geometry_params['jet_radius'] + 0.02, geometry_params['jet_radius'] + 0.05]
    list_angles_around = np.arange(0, 360, 10)

    for crrt_radius in list_radius_around:
        for crrt_angle in list_angles_around:
            angle_rad = np.pi * crrt_angle / 180.0
            list_position_probes.append(np.array([crrt_radius * math.cos(angle_rad), crrt_radius * math.sin(angle_rad)]))

    output_params = {'locations': list_position_probes,
                     'probe_type': 'pressure'
                     }

    optimization_params = {"num_steps_in_pressure_history": 1,
                        "min_value_jet_MFR": -1e-2,
                        "max_value_jet_MFR": 1e-2,
                        "smooth_control": (nb_actuations/dt)*(0.1*0.0005/80),
                        "zero_net_Qs": True,
                        "random_start": random_start}

    inspection_params = {"plot": plot,
                        "step": step,
                        "dump": dump,
                        "range_pressure_plot": [-2.0, 1],
                        "range_drag_plot": [-0.175, -0.13],
                        "range_lift_plot": [-0.2, +0.2],
                        "line_drag": -0.1595,
                        "line_lift": 0,
                        "show_all_at_reset": False,
                        "single_run":single_run
                        }

    reward_function = 'drag_plain_lift'

    verbose = 0

    number_steps_execution = int((simulation_duration/dt)/nb_actuations)

    # ---------------------------------------------------------------------------------
    # do the initialization

    #On fait varier la valeur de n-iter en fonction de si on remesh
    if(remesh):
        n_iter = int(2.5*(simulation_duration/dt))
        n_iter = 500
        #Si le dossier mesh existe, on l'efface
        if(os.path.exists('mesh')):
            shutil.rmtree('mesh')
        os.mkdir('mesh')
    else:
        n_iter = None


    #Processing the name of the simulation

    simu_name = 'Simu'

    if (geometry_params["jet_positions"][0] - 90) != 0:
        next_param = 'A' + str(geometry_params["jet_positions"][0] - 90)
        simu_name = '_'.join([simu_name, next_param])
    if geometry_params["cylinder_size"] != 0.01:
        next_param = 'M' + str(geometry_params["cylinder_size"])[2:]
        simu_name = '_'.join([simu_name, next_param])
    if optimization_params["max_value_jet_MFR"] != 0.01:
        next_param = 'maxF' + str(optimization_params["max_value_jet_MFR"])[2:]
        simu_name = '_'.join([simu_name, next_param])
    if nb_actuations != 80:
        next_param = 'NbAct' + str(nb_actuations)
        simu_name = '_'.join([simu_name, next_param])
    next_param = 'drag'
    if reward_function == 'recirculation_area':
        next_param = 'area'
    if reward_function == 'max_recirculation_area':
        next_param = 'max_area'
    elif reward_function == 'drag':
        next_param = 'last_drag'
    elif reward_function == 'max_plain_drag':
        next_param = 'max_plain_drag'
    elif reward_function == 'drag_plain_lift':
        next_param = 'lift'
    elif reward_function == 'drag_avg_abs_lift':
        next_param = 'avgAbsLift'
    simu_name = '_'.join([simu_name, next_param])

    env_2d_cylinder = Env2DCylinder(path_root=root,
                                    geometry_params=geometry_params,
                                    flow_params=flow_params,
                                    solver_params=solver_params,
                                    output_params=output_params,
                                    optimization_params=optimization_params,
                                    inspection_params=inspection_params,
                                    n_iter_make_ready=n_iter,  # On recalcule si besoin
                                    verbose=verbose,
                                    reward_function=reward_function,
                                    number_steps_execution=number_steps_execution,
                                    simu_name = simu_name)

    return(env_2d_cylinder)
