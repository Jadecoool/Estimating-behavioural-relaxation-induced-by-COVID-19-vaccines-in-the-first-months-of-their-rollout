import numpy as np
import pandas as pd
from typing import List
from behaviour_model_two_strains import SEIR_behaviour
import pyabc
import os
import uuid
import pickle as pkl
from pyabc import RV, Distribution
from datetime import timedelta
from typing import Callable, List
import argparse
from datetime import datetime, timedelta
from functions2wave_ import import_country

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

def calibration(epimodel: Callable,
                prior: pyabc.Distribution,
                params: dict,
                distance: Callable,
                observations: List[float],
                basin_name: str,
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


    abc = pyabc.ABCSMC(model, prior, distance, transitions=transition, population_size=population_size, sampler=pyabc.sampler.SingleCoreSampler())
    if db == None:
        db_path = os.path.join(f'./calibration_runs/{basin_name}/dbs/', f"new_07_bas_{filename}.db")
        abc.new("sqlite:///" + db_path, {"data": observations})

    else:
        abc.load(db, run_id)

    history = abc.run(minimum_epsilon=minimum_epsilon,
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)

    with open(os.path.join(f'./calibration_runs/{basin_name}/abc_history/', f"new_07_bas_{filename}.pkl"), 'wb') as file:
        pkl.dump(history, file)

    history.get_distribution()[0].to_csv(f"./posteriors/{basin_name}_new_07_bas_{filename}.csv")
    np.savez_compressed(f"./posteriors/{basin_name}_new_07_bas_{filename}.npz",
                        np.array([d["data"] for d in history.get_weighted_sum_stats()[1]]))

    return history




def run_calibration(basin, start_month, start_day, end_month, end_day, start_vax_month, start_vax_day):
    '''
    mode_option: baseline, constant, varying
    '''
    ################
    #prepare for the
    # datetimes
    start_date_org = datetime(2020, start_month, start_day)
    end_date = datetime(2021, end_month, end_day)
    vaxstart_date = datetime(2020, start_vax_month, start_vax_day)
    t_alpha_org = datetime(2020, 9, 14)
    # error metric
    #def wmape_(arr1, arr2):
        # weigthed mape
     #   return np.sum(np.abs(arr1 - arr2)) / np.sum(np.abs(arr1))

    # epi params
    eps = 1.0 / 3.7
    mu = 1.0 / 2.5

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
    #behaviour paramters
    mu_B, beta_B = 0, 0

    Alpha_increase=1.5

    # vaccine efficacy
    VE = 0.9
    VES = 0.85

    VE2 = 0.85
    VES2 = 0.75
    # number of compartment and age groups
    #ncomp = 20
    #nage = 16

    # parse basin name
    # parser = argparse.ArgumentParser(description='Optional app description')
    # parser.add_argument('basin', type=str, help='name of the basin')
    # args = parser.parse_args()
    # basin = args.basin
    #basin = "London"

    # import country
    country_dict = import_country(basin)

    #y1, w1 = start_date.isocalendar()[0], start_date.isocalendar()[1]
    #if w < 9:
    #    start_year_week=str(y1) + "-0" + str(w1)
    #else:
    #    start_year_week=str(y1) + "-" + str(w1)

    # deaths real
    deaths_real = country_dict["deaths"].loc[(country_dict["deaths"]["year_week"] >= "2020-38") & \
                                             (country_dict["deaths"]["year_week"] <= "2021-26")]["deaths"].values

    Nk=country_dict["Nk"]

    # create Basin object
    #basin = Basin(country, "../basins/")
    #Cs, dates = compute_contacts(basin, start_date, end_date)


    # get real deaths (first month excluded for the delay Delta)
    #real_deaths = basin.epi_data_deaths.loc[(basin.epi_data_deaths["date"] >= start_date) &
                                            #(basin.epi_data_deaths["date"] < end_date)].iloc[60:]

    #real_deaths.index = real_deaths.date
    #real_deaths = real_deaths.resample("W").sum()

    #在这里调用calibration主过程 ABC.SMC
    history = calibration(SEIR_behaviour,
                        prior=Distribution(
                                    R0=RV("uniform", 1.0, 2.0 - 1.0),
                                    Delta=RV('rv_discrete', values=(np.arange(3, 64), [1. / 61.] * 61)),
                                    start_week_delta=RV('rv_discrete', values=(np.arange(0, 6), [1. / 6.] * 6)),
                                    i0_q = RV("uniform", 10, 25 - 10),  # (20*new_pos-new_pos)
                                    r0_q = RV("uniform", 10, 25 - 10),
                                    t_alpha_delta = RV('rv_discrete', values=(np.arange(0, 45), [1. / 45.] * 45)),
                                    Alpha_increase = RV("uniform", 1.4, 1.5 - 1.4),
                                    alpha = RV("uniform", 0.01, 1 - 0.01),
                                    gamma = RV("uniform", 0.01, 1 - 0.01),
                                    r = RV("uniform", 1.01, 1.5 - 1.01),),
                       params={
                                'end_date': end_date,
                                'start_date_org': start_date_org,
                                'vaxstart_date': vaxstart_date,
                                't_alpha_org': t_alpha_org,
                                'eps': eps,
                                'mu': mu,
                                'ifr': IFR,
                                'VES': VES,
                                'VES2': VES2,
                                'VE': VE,
                                'VE2': VE2,
                                #'Alpha_increase': Alpha_increase,
                                'Nk': Nk,
                                'behaviour': 'constant_rate',
                                'google_mobility':True,
                                'basin':basin,},
                        distance=wmape_pyabc,
                        basin_name=basin,
                        #start_yw=start_year_week,
                        observations=deaths_real,
                        transition = pyabc.AggregatedTransition(
                                            mapping={
                                                'Delta': pyabc.DiscreteJumpTransition(domain=np.arange(3, 64), p_stay=0.7),
                                                'R0': pyabc.MultivariateNormalTransition(),
                                                'start_week_delta': pyabc.DiscreteJumpTransition(domain=np.arange(0, 6),
                                                                                                 p_stay=0.7),
                                                'i0_q': pyabc.MultivariateNormalTransition(),
                                                'r0_q': pyabc.MultivariateNormalTransition(),
                                                'alpha': pyabc.MultivariateNormalTransition(),
                                                'gamma': pyabc.MultivariateNormalTransition(),
                                                'r': pyabc.MultivariateNormalTransition(),
                                                'Alpha_increase': pyabc.MultivariateNormalTransition(),
                                                't_alpha_delta': pyabc.DiscreteJumpTransition(domain=np.arange(0, 45),
                                                                                      p_stay=0.7),
                                            }
                                        ),
                            max_nr_populations=10,
                            population_size=1000,
                            max_walltime=timedelta(hours=96),
                            minimum_epsilon=0.3)




if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--country")

    #args = parser.parse_args()
    #create_folder(str(args.country))
    #run_calibration(str(args.country))
    #startdates=[(11, 2), (11, 9), (11, 16), (11, 23), (11, 30), (12, 7)]

    #for i in range(len(startdates)):
     #   run_calibration("London", startdates[i][0], startdates[i][1], 7, 4, 12, 8)
    #startdates = (11, 1)
    run_calibration("Lombardy", 8, 10, 7, 4, 12, 27)