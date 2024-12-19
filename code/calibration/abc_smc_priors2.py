from typing import Dict, List, Callable
from datetime import datetime, timedelta
import numpy as np
import pyabc
import math
import uuid
import os
import pickle as pkl
from pyabc import RV, Distribution
from functions2wave_data2 import import_country

def wmape_pyabc(sim_data : dict,
                actual_data : dict) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE) to use for pyabc calibration
    Parameters
    ----------
        @param actual_data (dict): dictionary of actual data
        @param sim_data (dict): dictionary of simulated data
    Return
    ------
        @return: returns wmape between actual and simulated data
    """
    return np.sum(np.abs(actual_data['data'] - sim_data['data'])) / np.sum(np.abs(actual_data['data']))


def create_folder(country):
    if os.path.exists(f"./calibration_runs/{country}") == False:
        os.system(f"mkdir ./calibration_runs/{country}")
        os.system(f"mkdir ./calibration_runs/{country}/abc_history/")
        os.system(f"mkdir ./calibration_runs/{country}/dbs/")


def get_seir_function(is_london: bool, model_id: str):
    """
    根据地区和模型类型返回对应的SEIR函数
    Args:
        is_london: True or False
        model_id:  ('baseline', 'model1', 'model2', 'model3', 'model4')
    """
    if is_london:
        if model_id == 'baseline':
            from baseline_model_single_strain2_data2_deltadays import SEIR
        elif model_id in ['model1', 'model2']:
            from behaviour_model_single_strain2_data2_deltadays import SEIR
        elif model_id in ['model3', 'model4']:
            from variation_model_single_strain2_data2_deltadays import SEIR
        else:
            raise ValueError(f"Unknown model_id: {model_id}")
    else:
        if model_id == 'baseline':
            from baseline_model_two_strains2_data2_deltadays import SEIR
        elif model_id in ['model1', 'model2']:
            from behaviour_model_two_strains2_data2_deltadays import SEIR
        elif model_id in ['model3', 'model4']:
            from variation_model_two_strains2_data2_deltadays import SEIR
        else:
            print('model_id', model_id)
            raise ValueError(f"Unknown model_id: {model_id}")

    return SEIR


def run_fullmodel(R0, seed, Delta, start_week_delta, i0_q, r0_q, **fixed_params):
    is_london = fixed_params['basin'] == 'London'
    model_id = fixed_params.get('model_id', '')

    SEIR = get_seir_function(is_london, model_id)
    all_params = {
        'R0': R0,
        'seed': seed,
        'Delta': Delta,
        'start_week_delta': start_week_delta,
        'i0_q': i0_q,
        'r0_q': r0_q,
    }
    if is_london:
        if model_id == 'baseline':
            required_params = [
                'start_date_org', 'end_date', 'vaxstart_date',
                'eps', 'mu', 'ifr', 'VE', 'VES', 'Nk',
                'google_mobility', 'vaccine', 'basin'
            ]
        else:
            required_params = [
                'start_date_org', 'end_date', 'vaxstart_date',
                'alpha', 'gamma', 'r', 'eps', 'mu', 'ifr',
                'VE', 'VES', 'Nk', 'behaviour', 'behaviour_bool', 'basin'
            ]
    else:
        if model_id == 'baseline':
            required_params = [
                'start_date_org', 'end_date', 'vaxstart_date',
                't_alpha_org', 't_alpha_delta', 'eps', 'mu', 'ifr',
                'VE', 'VES', 'VE2', 'VES2', 'Nk', 'Alpha_increase',
                'google_mobility', 'vaccine', 'basin'
            ]
        else:
            required_params = [
                'start_date_org', 'end_date', 'vaxstart_date',
                't_alpha_org', 't_alpha_delta', 'eps', 'mu',
                'alpha', 'gamma', 'r', 'ifr', 'VE', 'VES', 'VE2', 'VES2',
                'Nk', 'Alpha_increase', 'behaviour', 'behaviour_bool', 'basin'
            ]

    for param in required_params:
        if param not in fixed_params:
            raise ValueError(f"Missing required parameter: {param}")
        all_params[param] = fixed_params[param]
    results = SEIR(**all_params)['weekly_deaths']

    return results


def calibration(epimodel: Callable,
                prior: pyabc.Distribution,
                params: dict,
                distance: Callable,
                observations: List[float],
                basin_name: str,
                model_id: str,
                #start_yw: str,
                transition: pyabc.AggregatedTransition,
                max_walltime: timedelta = None,
                population_size: int = 1000,
                minimum_epsilon: float = 0.15,
                max_nr_populations: int = 20,
                filename: str = '',
                run_id=None,
                db=None):
    """
    Run ABC calibration on given model and prior
    Parameters
    ----------
        @param epimodel (Callable): epidemic model
        @param prior (pyabc.Distribution): prior distribution
        @param params (dict): dictionary of fixed parameters value
        @param distance (Callable): distance function to use
        @param observations (List[float]): real observations
        @param model_name (str): model name
        @param basin_name (str): name of the basin
        @param transition (pyabc.AggregatedTransition): next gen. perturbation transitions
        @param max_walltime (timedelta): maximum simulation time
        @param population_size (int): size of the population of a given generation
        @param minimum_epsilon (float): minimum tolerance (if reached calibration stops)
        @param max_nr_population (int): maximum number of generations
        @param filename (str): name of the files used to store ABC results
        @param runid: Id of previous run (needed to resume it)
        @param db: path to dd of previous run (needed to resume it)

    Returns
    -------
        @return: returns ABC history
    """

    def model(p):
        return {'data': epimodel(**p, **params)}


    if filename == '':
        filename = str(uuid.uuid4())
    print(filename)


    abc = pyabc.ABCSMC(model, prior, distance, transitions=transition, population_size=population_size, sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=47))
    if db == None:
        db_path = os.path.join(f'./calibration_runs/{basin_name}/dbs/', f"priors2_{model_id}_bas_{filename}.db")
        abc.new("sqlite:///" + db_path, {"data": observations})

    else:
        abc.load(db, run_id)

    history = abc.run(minimum_epsilon=minimum_epsilon,
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)

    with open(os.path.join(f'./calibration_runs/{basin_name}/abc_history/', f"priors2_{model_id}_{filename}.pkl"), 'wb') as file:
        pkl.dump(history, file)

    history.get_distribution()[0].to_csv(f"./posteriors/{basin_name}_{model_id}_{filename}.csv")
    np.savez_compressed(f"./posteriors/{basin_name}_{model_id}_{filename}.npz",
                        np.array([d["data"] for d in history.get_weighted_sum_stats()[1]]))

    return history


# default params
DEFAULT_PARAMS = {
    'eps': 1.0 / 3.7,
    'mu': 1.0 / 2.5,
    'population_size': 1000,
    'minimum_epsilon': 0.3,
    'max_nr_populations': 10,
    'max_walltime': timedelta(hours=240)
}

# vaccine efficacy
VACCINE_PARAMS = {
    'London': {'VE': 0.85, 'VES': 0.75, 'VE2': None, 'VES2': None},
    'British Columbia': {'VE': 0.9, 'VES': 0.85, 'VE2': 0.85, 'VES2': 0.75},
    'Lombardy': {'VE': 0.9, 'VES': 0.85, 'VE2': 0.85, 'VES2': 0.75},
    'Sao Paulo': {'VE': 0.8, 'VES': 0.65, 'VE2': 0.9, 'VES2': 0.6}
}

# IFR
IFR = [0.00161 / 100,  # 0-4
       0.00161 / 100,  # 5-9
       0.00695 / 100,  # 10-14
       0.00695 / 100,  # 15-19
       0.0309 / 100,  # 20-24
       0.0309 / 100,  # 25-29
       0.0844 / 100,  # 30-34
       0.0844 / 100,  # 35-39
       0.161 / 100,  # 40-44
       0.161 / 100,  # 45-49
       0.595 / 100,  # 50-54
       0.595 / 100,  # 55-59
       1.93 / 100,  # 60-64
       1.93 / 100,  # 65-69
       4.28 / 100,  # 70-74
       6.04 / 100]  # 75+

# # Delta和start_week_delta的统一配置
# DELTA_CONFIG = {
#     'delta_range': (7, 35),
#     'delta_length': 28,  # 35 - 7
#     'start_week_range': (0, 30),
#     'start_week_length': 30  # 30 - 0
# }


def get_base_prior() -> Dict:
    """获取基础先验分布参数"""
    prior = {
        'R0': RV("uniform", 1, 3 - 1),
        'seed': RV("uniform", 1, 2 ** 32 - 1),
        'i0_q': RV("uniform", 5 * 10 ** (-4), 2 * 10 ** (-2) - 5 * 10 ** (-4)),
        'r0_q': RV("uniform", 0.1, 0.4 - 0.1),
        'Delta': RV('rv_discrete', values=(np.arange(3, 64), [1. / 61.] * 61)),
        'start_week_delta': RV('rv_discrete', values=(np.arange(0, 8), [1. / 8.] * 8))
    }
    return prior


def get_behaviour_prior():
    """获取行为模型额外的先验分布参数"""
    return {
        'alpha': RV("uniform", math.log(0.0001), math.log(10) - math.log(0.0001)),
        'gamma': RV("uniform", math.log(0.0001), math.log(10) - math.log(0.0001)),
        'r': RV("uniform", 1., 1.5 - 1.)
    }


def get_prior_distribution(basin: str, model_id: str) -> pyabc.Distribution:
    """根据地区和模型类型获取先验分布"""
    prior = get_base_prior()

    if basin != 'London':
        prior['t_alpha_delta'] = RV('rv_discrete', values=(np.arange(0, 43), [1. / 43.] * 43))
        if basin == 'Sao Paulo':
            prior['Alpha_increase'] = RV("uniform", 1.0, 2.5 - 1.0)

    if model_id != 'baseline':
        prior.update(get_behaviour_prior())

    return Distribution(**prior)


def get_transition_mapping(basin: str, model_id: str) -> Dict:
    """获取转换映射"""
    mapping = {
        'Delta': pyabc.DiscreteJumpTransition(domain=np.arange(3, 64),
                                              p_stay=0.7),
        'R0': pyabc.MultivariateNormalTransition(),
        'seed': pyabc.MultivariateNormalTransition(),
        'start_week_delta': pyabc.DiscreteJumpTransition(domain=np.arange(0, 8),
                                                         p_stay=0.7),
        'i0_q': pyabc.MultivariateNormalTransition(),
        'r0_q': pyabc.MultivariateNormalTransition(),
    }

    if basin != 'London':
        mapping['t_alpha_delta'] = pyabc.DiscreteJumpTransition(domain=np.arange(0, 43), p_stay=0.7)

        if basin == 'Sao Paulo':
            mapping['Alpha_increase'] = pyabc.MultivariateNormalTransition()

    if model_id != 'baseline':
        mapping.update({
            'alpha': pyabc.MultivariateNormalTransition(),
            'gamma': pyabc.MultivariateNormalTransition(),
            'r': pyabc.MultivariateNormalTransition()
        })

    return mapping


def get_fixed_params(basin: str, model_id: str, dates: Dict[str, datetime], Nk: List[int]) -> Dict:
    """获取固定参数"""
    params = {
        'end_date': dates['end_date'],
        'start_date_org': dates['start_date'],
        'vaxstart_date': dates['vaccine_date'],
        'eps': DEFAULT_PARAMS['eps'],
        'mu': DEFAULT_PARAMS['mu'],
        'ifr': IFR,
        'Nk': Nk,
        'basin': basin,
        'model_id': model_id  # 添加 model_id 到参数字典中
    }

    # 添加疫苗相关参数
    params.update(VACCINE_PARAMS[basin])

    if basin != 'London':
        params['t_alpha_org'] = dates['t_alpha_org']
        if basin in ['British Columbia', 'Lombardy']:
            params['Alpha_increase'] = 1.5

    if model_id == 'baseline':
        params.update({
            'google_mobility': True,
            'vaccine': True
        })
    else:
        params.update({
            'behaviour': 'constant_rate' if model_id in ['model1', 'model3'] else 'vaccine_rate',
            'behaviour_bool': True
        })

    return params


def run_calibration(basin: str, model_id: str,
                    dates_dict: Dict[str, datetime],
                    start_yw: str,
                    end_yw: str) -> pyabc.History:

    country_dict = import_country(basin, path_to_data='../data2')

    # 获取真实死亡数据
    deaths_real = country_dict["deaths"].loc[
        (country_dict["deaths"]["year_week"] >= start_yw) &
        (country_dict["deaths"]["year_week"] <= end_yw)
        ]["deaths"].values

    # 获取人口数据
    Nk = country_dict["Nk"]

    # 获取固定参数
    fixed_params = get_fixed_params(basin, model_id, dates_dict, Nk)

    # 获取先验分布
    prior = get_prior_distribution(basin, model_id)

    # 获取转换映射
    transition = pyabc.AggregatedTransition(
        mapping=get_transition_mapping(basin, model_id)
    )

    # 调用calibration函数
    history = calibration(
        epimodel=run_fullmodel,
        prior=prior,
        params=fixed_params,
        transition=transition,
        distance=wmape_pyabc,
        observations=deaths_real,
        basin_name=basin,
        model_id=model_id,
        max_walltime=DEFAULT_PARAMS['max_walltime'],
        population_size=DEFAULT_PARAMS['population_size'],
        minimum_epsilon=DEFAULT_PARAMS['minimum_epsilon'],
        max_nr_populations=DEFAULT_PARAMS['max_nr_populations']
    )

    return history
