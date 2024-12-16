import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functions2wave_data2 import import_country
import random
import os


def run_cf(basin):

    if basin == 'British Columbia':
        start_date_org = datetime(2020, 8, 17)
        end_date = datetime(2021, 7, 4)
        t_alpha_org = datetime(2020, 12, 21) - timedelta(days=42)
        vaxstart_date = datetime(2020, 12, 19)
        VE = 0.9
        VES = 0.85
        VE2 = 0.85
        VES2 = 0.75

    elif basin == 'Lombardy':
        start_date_org = datetime(2020, 8, 10)
        end_date = datetime(2021, 7, 4)
        t_alpha_org = datetime(2020, 9, 28) - timedelta(days=42)
        vaxstart_date = datetime(2020, 12, 27)
        VE = 0.9
        VES = 0.85
        VE2 = 0.85
        VES2 = 0.75

    elif basin == 'London':
        start_date_org = datetime(2020, 9, 21)
        end_date = datetime(2021, 7, 4)
        vaxstart_date = datetime(2020, 12, 8)
        VE = 0.85
        VES = 0.75

    elif basin == 'Sao Paulo':
        start_date_org = datetime(2020, 11, 23)
        end_date = datetime(2021, 10, 3)
        t_alpha_org = datetime(2021, 3, 29) - timedelta(days=42)
        vaxstart_date = datetime(2021, 1, 18)
        VE = 0.8
        VES = 0.65
        VE2 = 0.9
        VES2 = 0.6

    if basin == 'London':
        from baseline_model_single_strain2_data2 import SEIR
    else:
        from baseline_model_two_strains2_data2 import SEIR


    country_dict = import_country(basin)
    Nk=country_dict['Nk']
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

    directory = './posteriors/new6/'
    file = [f for f in os.listdir(directory) if basin in f and '_06_' in f and f.endswith('.csv')]
    print(file)
    pos = pd.read_csv(os.path.join(directory, file[0]))

    death_samples_novax = []
    infection_samples_novax = []
    death_samples_vax = []
    infection_samples_vax = []

    for idx, data in pos.iterrows():
        print(idx)
        if basin == 'London':
            Delta = data[1]
            R0 = data[2]
            i0_q = data[3]
            r0_q = data[4]
            t0 = data[5]
            np.random.seed(0)
            results1 = SEIR(start_date_org=start_date_org, end_date=end_date, start_week_delta=int(t0),
                                           vaxstart_date=vaxstart_date,
                                          i0_q=i0_q, r0_q=r0_q, R0=R0,
                                          eps=eps, mu=mu, ifr=IFR, Delta=Delta,
                                          VE=VE, VES=VES,
                                          Nk=Nk,
                                          google_mobility=True,
                                          vaccine=True,
                                          basin=basin)
            np.random.seed(0)
            results2 = SEIR(start_date_org=start_date_org, end_date=end_date, start_week_delta=int(t0),
                                          vaxstart_date=vaxstart_date,
                                          i0_q=i0_q, r0_q=r0_q, R0=R0,
                                          eps=eps, mu=mu, ifr=IFR, Delta=Delta,
                                          VE=VE, VES=VES,
                                          Nk=Nk,
                                          google_mobility=True,
                                          vaccine=False,
                                          basin=basin)
            death_sim_vax = results1['deaths_not_cut']
            infection_sim_vax = results1['infections_not_cut']
            death_sim_novax = results2['deaths_not_cut']
            infection_sim_novax = results2['infections_not_cut']

        elif basin == 'Sao Paulo':
            Delta = data[2]
            R0 = data[3]
            i0_q = data[4]
            r0_q = data[5]
            t0 = data[6]

            Alpha_increase = data[1]
            t_alpha_delta = data[7]
            np.random.seed(0)
            results1 = SEIR(start_date_org=start_date_org, end_date=end_date,
                                                    start_week_delta=int(t0),
                                                    t_alpha_org=t_alpha_org, vaxstart_date=vaxstart_date,
                                                    i0_q=i0_q, r0_q=r0_q, R0=R0,
                                                    eps=eps, mu=mu, ifr=IFR, Delta=Delta,
                                                    VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                                    Nk=Nk,
                                                    Alpha_increase=Alpha_increase, t_alpha_delta=t_alpha_delta,
                                                    google_mobility=True,
                                                    vaccine=True,
                                                    basin=basin)
            np.random.seed(0)
            results2 = SEIR(start_date_org=start_date_org,
                                                                    end_date=end_date, start_week_delta=int(t0),
                                                                    t_alpha_org=t_alpha_org,
                                                                    vaxstart_date=vaxstart_date,
                                                                    i0_q=i0_q, r0_q=r0_q, R0=R0,
                                                                    eps=eps, mu=mu, ifr=IFR, Delta=Delta,
                                                                    VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                                                    Nk=Nk,
                                                                    Alpha_increase=Alpha_increase, t_alpha_delta=t_alpha_delta,
                                                                    google_mobility=True,
                                                                    vaccine=False,
                                                                    basin=basin)
            death_sim_vax = results1['deaths_not_cut']
            infection_sim_vax = results1['infections_not_cut']
            death_sim_novax = results2['deaths_not_cut']
            infection_sim_novax = results2['infections_not_cut']

        else:
            Delta = data[1]
            R0 = data[2]
            i0_q = data[3]
            r0_q = data[4]
            t0 = data[5]
            t_alpha_delta = data[6]
            np.random.seed(0)
            results1 = SEIR(start_date_org=start_date_org, end_date=end_date,
                                                    start_week_delta=int(t0),
                                                    t_alpha_org=t_alpha_org, vaxstart_date=vaxstart_date,
                                                    i0_q=i0_q, r0_q=r0_q, R0=R0,
                                                    eps=eps, mu=mu, ifr=IFR, Delta=Delta,
                                                    VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                                    Nk=Nk,
                                                    Alpha_increase=1.5, t_alpha_delta=t_alpha_delta,
                                                    google_mobility=True,
                                                    vaccine=True,
                                                    basin=basin)
            np.random.seed(0)
            results2 = SEIR(start_date_org=start_date_org,
                                                                    end_date=end_date, start_week_delta=int(t0),
                                                                    t_alpha_org=t_alpha_org,
                                                                    vaxstart_date=vaxstart_date,
                                                                    i0_q=i0_q, r0_q=r0_q, R0=R0,
                                                                    eps=eps, mu=mu, ifr=IFR, Delta=Delta,
                                                                    VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                                                    Nk=Nk,
                                                                    Alpha_increase=1.5,
                                                                    t_alpha_delta=t_alpha_delta,
                                                                    google_mobility=True,
                                                                    vaccine=False,
                                                                    basin=basin)
            death_sim_vax = results1['deaths_not_cut']
            infection_sim_vax = results1['infections_not_cut']
            death_sim_novax = results2['deaths_not_cut']
            infection_sim_novax = results2['infections_not_cut']


        death_samples_vax.append(death_sim_vax)
        infection_samples_vax.append(infection_sim_vax)

        death_samples_novax.append(death_sim_novax)
        infection_samples_novax.append(infection_sim_novax)

    np.savez_compressed(f"./posteriors/counterfactual_remove_vaccine/death_new(different lengths of trajectories)/{basin}_06_death0_vax_dailyvax.npz", death_samples_vax)
    np.savez_compressed(f"./posteriors/counterfactual_remove_vaccine/infection_new(different lengths of trajectories)/{basin}_06_infection0_vax_dailyvax.npz", infection_samples_vax)

    np.savez_compressed(f"./posteriors/counterfactual_remove_vaccine/death_new(different lengths of trajectories)/{basin}_06_death0_novax_dailyvax.npz", death_samples_novax)
    np.savez_compressed(f"./posteriors/counterfactual_remove_vaccine/infection_new(different lengths of trajectories)/{basin}_06_infection0_novax_dailyvax.npz", infection_samples_novax)