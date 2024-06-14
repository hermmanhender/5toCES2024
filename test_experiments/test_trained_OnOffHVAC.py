from eprllib.postprocess.marl_init_evaluation import init_drl_evaluation
from eprllib.postprocess.marl_init_conventional import init_rb_HVAC_OnOff_evaluation
from eprllib.tools.rewards import normalize_reward_function
from numpy.random import choice
import gymnasium as gym

name = "CES2024_LSTM_PPO_OnOffHVAC"

env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/prot_3_ceiling_SetPointHVAC_PowerLimit.epJSON",
    "epw_training": choice(["C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H1.epw",
                            "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H2.epw",
                            "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H3.epw"]),
    "epw": "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H4.epw",
    # Configure the output directory for the EnergyPlus simulation.
    'output': 'C:/Users/grhen/Documents/Resultados_RLforEP/'+name,
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'ep_terminal_output': True,
    
    # === EXPERIMENT OPTIONS === #
    # For evaluation process 'is_test=True' and for trainig False.
    'is_test': True,
    
    # === ENVIRONMENT OPTIONS === #
    # action space for simple agent case
    'action_space': gym.spaces.Discrete(2),
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
        "HVAC_OnOff": ("Schedule:Constant", "Schedule Value", "HVAC_avialability"),
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
    "infos_variables": ["ppd", 'heating', 'cooling', 'occupancy', 'Ti', 'HVAC_OnOff'],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 10,
    'cut_episode_len': None, # longitud en días del periodo cada el cual se trunca un episodio
    "weather_prob_days": 2,
    # Reward function config
    'reward_function': normalize_reward_function,
    'cut_reward_len': 1, # longitud en días del periodo cada el cual se registra una recompensa
    'beta_reward': 0.5,
    'energy_ref': 1500000, # maxima energía (en joules) requerida por el control convencional
}
# beta 0.5
checkpoint_path = 'C:/Users/grhen/ray_results/20240604095101_OnOffHVAC_marl_PPO/OnOffHVAC_beta05_PPO_2fcad_00000/checkpoint_000027'

policy_config = { # configuracion del control convencional
        'SP_temp': 22, #es el valor de temperatura de confort
        'dT_up': 3, #es el límite superior para el rango de confort
        'dT_dn': 3, #es el límite inferior para el rango de confort
    }

add_name = 'drl_beta05'
init_drl_evaluation(
    env_config=env_config,
    checkpoint_path=checkpoint_path,
    name=name+'_'+add_name
)

add_name = 'rb_beta05'
init_rb_HVAC_OnOff_evaluation(
    env_config=env_config,
    policy_config=policy_config,
    name=name+'_'+add_name
)