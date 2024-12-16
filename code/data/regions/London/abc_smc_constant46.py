import numpy as np
from typing import List
from vaccine_sandiego import integrate_BV
import pyabc
import os
import uuid
import pickle as pkl
from pyabc import RV, Distribution
from datetime import timedelta
from typing import Callable, List
import argparse
from datetime import datetime, timedelta
from functions2wave import import_country, get_totR, update_contacts

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


    abc = pyabc.ABCSMC(model, prior, distance, transitions=transition, population_size=population_size, sampler = pyabc.sampler.MulticoreParticleParallelSampler(n_procs = 1))
    if db == None:
        db_path = os.path.join(f'./calibration_runs/{basin_name}/dbs/', f"variation_45_const_rate_{filename}.db")
        abc.new("sqlite:///" + db_path, {"data": observations})

    else:
        abc.load(db, run_id)

    history = abc.run(minimum_epsilon=minimum_epsilon,
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)

    with open(os.path.join(f'./calibration_runs/{basin_name}/abc_history/', f"variation_45_const_rate_{filename}.pkl"), 'wb') as file:
        pkl.dump(history, file)

    history.get_distribution()[0].to_csv(f"./posteriors/variation_posterior_45_{basin_name}_const_rate_{filename}.csv")
    np.savez_compressed(f"./posteriors/variation_posterior_samples_45_{basin_name}_const_rate_{filename}.npz",
                        np.array([d["data"] for d in history.get_weighted_sum_stats()[1]]))

    return history




def run_calibration(basin):
    '''
    mode_option: baseline, constant, varying
    '''
    ################
    #prepare for the
    # datetimes
    start_date = datetime(2020, 11, 2)
    end_date = datetime(2021, 7, 4)
    vaxstart_date = datetime(2020, 12, 8)

    # error metric
    #def wmape_(arr1, arr2):
        # weigthed mape
     #   return np.sum(np.abs(arr1 - arr2)) / np.sum(np.abs(arr1))

    # epi params
    eps = 1.0 / 3.7
    mu = 1.0 / 2.5
    omega = 1.0 / 1.5
    chi = 0.55
    f = 0.35
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

    # behaviour params turn on (to be calabrated) see later code
    #r = 1.0
    #alpha, gamma, rV,
    VES, VEM = 0.8, 0.5

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

    # pre-compute contacts matrices
    Cs = {}
    date, dates = start_date, [start_date]
    for i in range((end_date - start_date).days):
        Cs[date] = update_contacts(country_dict, date)
        date += timedelta(days=1)
        dates.append(date)

    # add week of year
    dates = [datetime(2020, 11, 2) + timedelta(days=d) for d in range(244)]  # 209是相差days   之前数据是差244 days
    year_week = []
    for date in dates:  #####date.isocalendar 获得year, 周数, 星期数
        y, w = date.isocalendar()[0], date.isocalendar()[1]
        if w < 9:
            year_week.append(str(y) + "-0" + str(w))
        else:
            year_week.append(str(y) + "-" + str(w))

    # deaths real
    deaths_real = country_dict["deaths"].loc[(country_dict["deaths"]["year_week"] >= "2020-45") & \
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
    history = calibration(integrate_BV,
                        prior=Distribution(
                                    R0=RV("uniform", 0.6, 3.3 - 0.6),
                                    Delta=RV('rv_discrete', values=(np.arange(10, 36), [1. / 26.] * 26)),
                                    i0=RV("uniform", 1.0, 1000.0 - 1.0),
                                    r0=RV("uniform",1.0, 1000.0 - 1.0),
                                    r=RV("uniform",1.01, 1.61 - 1.01),
                                    alpha = RV("uniform", 0.1, 100.0 - 0.1),
                                    gamma=RV("uniform", 0.1, 100.0 - 0.1)),
                       params={
                                'T': (end_date - start_date).days,
                                'T_nonvax': (vaxstart_date-start_date).days,
                                'eps': eps,
                                'mu': mu,
                                'omega': omega,
                                'chi': chi,
                                'f': f,
                                'IFR': IFR,
                                'VES': VES,
                                'VEM': VEM,
                                'Cs': Cs,
                                'Nk': Nk,
                                'model': 'constant_rate',
                                'dates': dates,
                                'basin':basin},
                        distance=wmape_pyabc,
                        basin_name=basin,
                        observations=deaths_real,
                        transition = pyabc.AggregatedTransition(
                                            mapping={
                                                'Delta': pyabc.DiscreteJumpTransition(domain=np.arange(10, 36), p_stay=0.7),
                                                'R0': pyabc.MultivariateNormalTransition(),
                                                'i0': pyabc.MultivariateNormalTransition(),
                                                'r0': pyabc.MultivariateNormalTransition(),
                                                'r': pyabc.MultivariateNormalTransition(),
                                                'alpha': pyabc.MultivariateNormalTransition(),
                                                'gamma': pyabc.MultivariateNormalTransition(),
                                            }
                                        ),
                            max_nr_populations=10,
                            population_size=1000,
                            max_walltime=timedelta(hours=12),
                            minimum_epsilon=0.15)




if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--country")

    #args = parser.parse_args()
    #create_folder(str(args.country))
    #run_calibration(str(args.country))
    run_calibration("London")
