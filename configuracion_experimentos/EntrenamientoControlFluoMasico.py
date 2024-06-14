"""# DEEP REINFORCEMENT LEARNING FOR ENERGYPLUS MODELS

This script test an EnergyPlus model based on Prototype 3 by IPV Mendoza.
The EnergyPlus model is a multi-agent environment based on the Gym library.
The agents are the building houses. The goal is to control the heating and cooling trought
the on-off switches. The agents are controlled by a DQN algorithm.

The script is divided in 3 parts:
1. Define the environment. (prot_3_ceiling_OnOffHVAC)
2. Define the algorithm to train the policy.
3. Define the experiment controls.
"""

"""## DEFINE ENVIRONMENT VARIABLES

This is not always required, but here is an example of how to define 
a environmet variable: (NOTE: This must be implemented before import ray.)

>>> import os
>>> os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
import os
os.environ['RAY_DEDUP_LOGS'] = '0'

"""## IMPORT THE NECESARY LIBRARIES
"""
import time
from tempfile import TemporaryDirectory
# This library generate a tmp folder to save the the EnergyPlus output
import gymnasium as gym
# Used to configurate the action and observation spaces
import ray
# To init ray
from ray import air, tune
# To configurate the execution of the experiment
from ray.tune import register_env
# To register the custom environment. RLlib is not compatible with conventional Gym register of 
# custom environments.
from ray.rllib.algorithms.ppo.ppo import PPOConfig
# To config the PPO algorithm.
# The ray ASHA Schedule is imported
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.policy.policy import PolicySpec
from eprllib.env.multiagent.marl_ep_gym_env import EnergyPlusEnv_v0
from eprllib.tools.rewards import normalize_reward_function
from eprllib.tools.action_transformers import thermostat_dual_mass_flow_rate
from numpy.random import choice
from typing import Dict, Any

"""## DEFINE THE EXPERIMENT CONTROLS
"""
algorithm = 'PPO'
# Define the algorithm to use to train the policy. Options are: PPO, SAC, DQN.
tune_runner  = False
# Define if the experiment tuning the variables or execute a unique configuration.
restore = False
# To define if is necesary to restore or not a previous experiment. Is necesary to stablish a 'restore_path'.
restore_path = ''
# Path to the folder where the experiment is located.
name = '_'+algorithm+'_beta05_'

env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/prot_3_ceiling_SetPointHVAC_PowerLimit.epJSON",
    "epw_training": choice(["C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H1.epw",
                            "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H2.epw",
                            "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H3.epw"]),
    "epw": "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H4.epw",
    # Configure the output directory for the EnergyPlus simulation.
    'output': TemporaryDirectory("output","",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'ep_terminal_output': False,
    
    # === EXPERIMENT OPTIONS === #
    # For evaluation process 'is_test=True' and for trainig False.
    'is_test': False,
    
    # === ENVIRONMENT OPTIONS === #
    # action space for simple agent case
    'action_space': gym.spaces.Discrete(6),
    "ep_variables":{
        "To": ("Site Outdoor Air Drybulb Temperature", "Environment"),
        "Ti": ("Zone Mean Air Temperature", "Thermal Zone: Living"),
        "v": ("Site Wind Speed", "Environment"),
        "d": ("Site Wind Direction", "Environment"),
        "RHo": ("Site Outdoor Air Relative Humidity", "Environment"),
        "RHi": ("Zone Air Relative Humidity", "Thermal Zone: Living"),
        "pres": ("Site Outdoor Air Barometric Pressure", "Environment"),
        "occupancy": ("Zone People Occupant Count", "Thermal Zone: Living"),
        "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Living Occupancy")
    },
    "ep_meters": {
        "electricity": "Electricity:Zone:THERMAL ZONE: LIVING",
        "gas": "NaturalGas:Zone:THERMAL ZONE: LIVING",
        "heating": "Heating:DistrictHeatingWater",
        "cooling": "Cooling:DistrictCooling",
    },
    "ep_actuators": {
        "heating_setpoint": ("Schedule:Compact", "Schedule Value", "HVACTemplate-Always 19"),
        "cooling_setpoint": ("Schedule:Compact", "Schedule Value", "HVACTemplate-Always 25"),
        "AirMassFlowRate": ("Ideal Loads Air System", "Air Mass Flow Rate", "Thermal Zone: Living Ideal Loads Air System"),
    },
    'time_variables': [
        'hour',
        'day_of_year',
        'day_of_week',
        ],
    'weather_variables': [
        'is_raining',
        'sun_is_up',
        "today_weather_beam_solar_at_time",
        ],
    "infos_variables": ["ppd", 'heating', 'cooling', 'occupancy','Ti'],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 10,
    'cut_episode_len': None, # longitud en días del periodo cada el cual se trunca un episodio
    "weather_prob_days": 2,
    # Action transformer
    'action_transformer': thermostat_dual_mass_flow_rate,
    # Reward function config
    'reward_function': normalize_reward_function,
    'reward_function_config': {
        # cut_reward_len_timesteps: Este parámetro permite que el agente no reciba una recompensa 
        # en cada paso de tiempo, en cambio las variables para el cálculo de la recompensa son 
        # almacenadas en una lista para luego utilizar una recompensa promedio cuando se alcanza 
        # la cantidad de pasos de tiempo indicados por 'cut_reward_len_timesteps'.
        'cut_reward_len_timesteps': 144,
        # Parámetros para la exclusión de términos de la recompensa
        'comfort_reward': True,
        'energy_reward': True,              
        # beta_reward: Parámetros de ponderación para la energía y el confort.
        'beta_reward': 0.5,               
        # energy_ref: El valor de referencia depende del entorno. Este puede corresponder a la energía máxima que puede demandar el entorno en un paso de tiempo, un valor de energía promedio u otro.
        'energy_ref': 1500000,
        # co2_ref: Este parámtero indica el valor de referencia de consentración de CO2 que se espera tener en un ambiente con una calidad de aire óptima.
        'co2_ref': 870,
        # Nombres de las variables utilizadas en su configuración del entorno.
        'occupancy_name': 'occupancy',
        'ppd_name': 'ppd',
        'T_interior_name': 'Ti',
        'cooling_name': 'cooling',
        'heating_name': 'heating'
    }
}

"""## INIT RAY AND REGISTER THE ENVIRONMENT
"""
ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')
# Inicialiced Ray Server
register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))
# Register the environment.

"""## CONFIGURATION OF THE SELECTED ALGORITHM

Different algorithms are configurated in the following lines. It is possible to add other
algorithm configuration here or modify the presents.
"""

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"


if algorithm == 'PPO': # PPO Configuration
    algo = PPOConfig().training(
        # General Algo Configs
        gamma = 0.8 if not tune_runner else tune.choice([0.7, 0.8, 0.9, 0.99]),
        # Float specifying the discount factor of the Markov Decision process.
        lr = 0.0001 if not tune_runner else tune.choice([0.0001, 0.001, 0.01]),
        # The learning rate (float) or learning rate schedule
        model = {
            "fcnet_hiddens": [128,128,128,128],
            "fcnet_activation": "relu",# if not tune_runner else tune.choice(['tanh', 'relu', 'swish', 'linear']),
            
            # == LSTM ==
            # Whether to wrap the model with an LSTM.
            "use_lstm": True,
            # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 48,
            # Size of the LSTM cell.
            "lstm_cell_size": 128,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": False,
            # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": False,
            # Initializer function or class descriptor for LSTM weights.
            # Supported values are the initializer names (str), classes or functions listed
            # by the frameworks (`tf2``, `torch`). See
            # https://pytorch.org/docs/stable/nn.init.html for `torch` and
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
            # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
            "lstm_weights_initializer": None,
            # Initializer configuration for LSTM weights.
            # This configuration is passed to the initializer defined in
            # `lstm_weights_initializer`.
            "lstm_weights_initializer_config": None,
            # Initializer function or class descriptor for LSTM bias.
            # Supported values are the initializer names (str), classes or functions listed
            # by the frameworks (`tf2``, `torch`). See
            # https://pytorch.org/docs/stable/nn.init.html for `torch` and
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
            # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
            "lstm_bias_initializer": None,
            # Initializer configuration for LSTM bias.
            # This configuration is passed to the initializer defined in
            # `lstm_bias_initializer`.
            "lstm_bias_initializer_config": None,
            # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            "_time_major": False,
            
            },
        # Arguments passed into the policy model. See models/catalog.py for a full list of the 
        # available model options.
        train_batch_size = 8000,
        # PPO Configs
        lr_schedule = None, # List[List[int | float]] | None = NotProvided,
        # Learning rate schedule. In the format of [[timestep, lr-value], [timestep, lr-value], …] 
        # Intermediary timesteps will be assigned to interpolated learning rate values. A schedule 
        # should normally start from timestep 0.
        use_critic = True, # bool | None = NotProvided,
        # Should use a critic as a baseline (otherwise don’t use value baseline; required for using GAE).
        use_gae = True, # bool | None = NotProvided,
        # If true, use the Generalized Advantage Estimator (GAE) with a value function, 
        # see https://arxiv.org/pdf/1506.02438.pdf.
        lambda_ = 0.2 if not tune_runner else tune.choice([0, 0.2, 0.5, 0.7, 1.0]), # float | None = NotProvided,
        # The GAE (lambda) parameter.  The generalized advantage estimator for 0 < λ < 1 makes a 
        # compromise between bias and variance, controlled by parameter λ.
        use_kl_loss = True, # bool | None = NotProvided,
        # Whether to use the KL-term in the loss function.
        kl_coeff = 9 if not tune_runner else tune.uniform(0.3, 10.0), # float | None = NotProvided,
        # Initial coefficient for KL divergence.
        kl_target = 0.05 if not tune_runner else tune.uniform(0.001, 0.1), # float | None = NotProvided,
        # Target value for KL divergence.
        sgd_minibatch_size = 800, # if not tune_runner else tune.choice([48, 128]), # int | None = NotProvided,
        # Total SGD batch size across all devices for SGD. This defines the minibatch size 
        # within each epoch.
        num_sgd_iter = 40, # if not tune_runner else tune.randint(30, 60), # int | None = NotProvided,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
        shuffle_sequences = True, # bool | None = NotProvided,
        # Whether to shuffle sequences in the batch when training (recommended).
        vf_loss_coeff = 0.4 if not tune_runner else tune.uniform(0.1, 1.0), # Tune this! float | None = NotProvided,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if you set 
        # vf_share_layers=True inside your model’s config.
        entropy_coeff = 10 if not tune_runner else tune.uniform(0.95, 15.0), # float | None = NotProvided,
        # Coefficient of the entropy regularizer.
        entropy_coeff_schedule = None, # List[List[int | float]] | None = NotProvided,
        # Decay schedule for the entropy regularizer.
        clip_param = 0.1, # if not tune_runner else tune.uniform(0.1, 0.4), # float | None = NotProvided,
        # The PPO clip parameter.
        vf_clip_param = 10, # if not tune_runner else tune.uniform(0, 50), # float | None = NotProvided,
        # Clip param for the value function. Note that this is sensitive to the scale of the 
        # rewards. If your expected V is large, increase this.
        grad_clip = None, # float | None = NotProvided,
        # If specified, clip the global norm of gradients by this amount.
    ).environment(
        env = "EPEnv",
        env_config = env_config,
    ).framework(
        framework = 'torch',
    ).fault_tolerance(
        recreate_failed_env_runners = True,
    ).env_runners(
        num_env_runners = 7,
        create_env_on_local_worker = True,
        rollout_fragment_length = 'auto',
        enable_connectors = True,
        num_envs_per_env_runner = 1,
        explore = True,
        exploration_config = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.,
            "final_epsilon": 0.,
            "epsilon_timesteps": 6*24*365*8*6,
        },
    ).multi_agent(
        policies = {
            'shared_policy': PolicySpec(),
        },
        policy_mapping_fn = policy_mapping_fn,
    ).reporting( # multi_agent config va aquí
        min_sample_timesteps_per_iteration = 1000,
    ).checkpointing(
        export_native_model_files = True,
    ).debugging(
        log_level = "ERROR",
    ).resources(
        num_gpus = 0,
    )

"""## START EXPERIMENT
"""
def trial_str_creator(trial):
    """This method create a description for the folder where the outputs and checkpoints 
    will be save.

    Args:
        trial: A trial type of RLlib.

    Returns:
        str: Return a unique string for the folder of the trial.
    """
    return "DualSetPointOnOffHVAC_beta05_{}_{}".format(trial.trainable_name, trial.trial_id)

if not restore:
    tune.Tuner(
        algorithm,
        tune_config=tune.TuneConfig(
            mode = "max",
            metric = "episode_reward_mean",
            num_samples = 1,
            # This is necesary to iterative execute the search_alg to improve the hyperparameters
            reuse_actors = False,
            trial_name_creator = trial_str_creator,
            trial_dirname_creator = trial_str_creator,
            
            # == Search algorithm configuration ==
            #search_alg = Repeater(BayesOptSearch(),repeat=10),
            #search_alg = BayesOptSearch(),
            
            # == Scheduler algorithm configuration ==
            # scheduler = ASHAScheduler(
            #     time_attr = 'timesteps_total',
            #     max_t= 6*24*365*11,
            #     grace_period=6*24*365*5,
            # ),
            
            
        ),
        run_config=air.RunConfig(
            name = "{date}_DualSetPointHVAC_marl_{algorithm}".format(
                date = time.strftime("%Y%m%d%H%M%S"),
                algorithm = algorithm,
            ),
            storage_path = 'C:/Users/grhen/ray_results',
            stop = {"episodes_total": 8*12},
            log_to_file = True,
            
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end = True,
                checkpoint_frequency = 20,
                #num_to_keep = 20,
            ),
            failure_config=air.FailureConfig(
                max_failures = 100,
                # Tries to recover a run up to this many times.
            ),
        ),
        param_space = algo.to_dict(),
    ).fit()

else:
    tune.Tuner.restore(
        path = restore_path,
        trainable = algorithm,
        resume_unfinished = True,
        resume_errored = True,
    )

"""## END EXPERIMENT AND SHUTDOWN RAY SERVE
"""
ray.shutdown()