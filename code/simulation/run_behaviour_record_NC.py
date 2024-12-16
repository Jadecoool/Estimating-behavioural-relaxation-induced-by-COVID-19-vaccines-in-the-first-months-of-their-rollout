import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functions2wave_data2 import import_country
import random
import os
random.seed(0)

def get_NCandC(basin, model):
    country_dict = import_country(basin, path_to_data='../data2')
    Nk=country_dict['Nk']
    eps = 1.0 / 3.7
    mu = 1.0 / 2.5

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
        from behaviour_model_single_strain2_data2 import SEIR_behaviour
        from variation_model_single_strain2_data2 import SEIR_variation
    else:
        from behaviour_model_two_strains2_data2 import SEIR_behaviour
        from variation_model_two_strains2_data2 import SEIR_variation

    if model == '_07_' or model == '_09_':
        behaviour_rate = 'constant_rate'
    elif model == '_08_' or model == '_10_':
        behaviour_rate = 'vaccine_rate'


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
    file = [f for f in os.listdir(directory) if basin in f and model in f and f.endswith('.csv')]
    print(file)
    pos = pd.read_csv(os.path.join(directory, file[0]))

    noncompliant_samples=[]
    compliant_samples=[]

    if basin == 'London':
        for idx, data in pos.iterrows():
            print(idx)
            Delta = data[1]
            R0 = data[2]

            alpha = data[3]
            gamma = data[4]
            i0_q = data[5]
            r = data[6]
            r0_q = data[7]
            t0 = data[8]
            if model == '_07_' or model == '_08_':
                results = SEIR_behaviour(start_date_org=start_date_org, end_date=end_date,
                                                        vaxstart_date=vaxstart_date,
                                                        start_week_delta=int(t0), i0_q=i0_q, r0_q=r0_q, R0=R0, Delta=Delta,
                                                        eps=eps, mu=mu, ifr=IFR,
                                                        VE=VE, VES=VES,
                                                        Nk=Nk,
                                                        alpha=alpha, gamma=gamma, r=r,
                                                        behaviour=behaviour_rate,
                                                        behaviour_bool=True,
                                                        basin=basin)
            elif model == '_09_' or model == '_10_':
                results = SEIR_variation(start_date_org=start_date_org, end_date=end_date,
                                                        vaxstart_date=vaxstart_date,
                                                        start_week_delta=int(t0), i0_q=i0_q, r0_q=r0_q, R0=R0, Delta=Delta,
                                                        eps=eps, mu=mu, ifr=IFR,
                                                        VE=VE, VES=VES,
                                                        Nk=Nk,
                                                        alpha=alpha, gamma=gamma, r=r,
                                                        behaviour=behaviour_rate,
                                                        behaviour_bool = True,
                                                        basin=basin)
            NC = results['daily_noncompliant']
            C = results['daily_compliant']
            noncompliant_samples.append(NC)
            compliant_samples.append(C)
    elif basin == 'Sao Paulo':
        for idx, data in pos.iterrows():
            print(idx)
            Alpha_increase = data[1]
            Delta = data[2]
            R0 = data[3]

            alpha = data[4]
            gamma = data[5]
            i0_q = data[6]
            r = data[7]
            r0_q = data[8]
            t0 = data[9]
            t_alpha_delta = data[10]
            if model == '_07_' or model == '_08_':
                results = SEIR_behaviour(start_date_org=start_date_org, end_date=end_date,
                                         vaxstart_date=vaxstart_date, t_alpha_org=t_alpha_org,
                                         start_week_delta=int(t0), i0_q=i0_q, r0_q=r0_q, R0=R0, Delta=Delta,
                                         Alpha_increase=Alpha_increase, t_alpha_delta=t_alpha_delta,
                                         eps=eps, mu=mu, ifr=IFR,
                                         VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                         Nk=Nk,
                                         alpha=alpha, gamma=gamma, r=r,
                                         behaviour=behaviour_rate,
                                         behaviour_bool=True,
                                         basin=basin)
            elif model == '_09_' or model == '_10_':
                results = SEIR_variation(start_date_org=start_date_org, end_date=end_date,
                                         vaxstart_date=vaxstart_date, t_alpha_org=t_alpha_org,
                                         start_week_delta=int(t0), i0_q=i0_q, r0_q=r0_q, R0=R0, Delta=Delta,
                                         Alpha_increase=Alpha_increase, t_alpha_delta=t_alpha_delta,
                                         eps=eps, mu=mu, ifr=IFR,
                                         VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                         Nk=Nk,
                                         alpha=alpha, gamma=gamma, r=r,
                                         behaviour=behaviour_rate,
                                         behaviour_bool=True,
                                         basin=basin)
            NC = results['daily_noncompliant']
            C = results['daily_compliant']
            noncompliant_samples.append(NC)
            compliant_samples.append(C)

    elif basin == 'British Columbia' or basin == 'Lombardy':
        for idx, data in pos.iterrows():
            print(idx)
            Delta = data[1]
            R0 = data[2]

            alpha = data[3]
            gamma = data[4]
            i0_q = data[5]
            r = data[6]
            r0_q = data[7]
            t0 = data[8]
            t_alpha_delta = data[9]
            if model == '_07_' or model == '_08_':
                results = SEIR_behaviour(start_date_org=start_date_org, end_date=end_date,
                                         vaxstart_date=vaxstart_date, t_alpha_org=t_alpha_org,
                                         start_week_delta=int(t0), i0_q=i0_q, r0_q=r0_q, R0=R0, Delta=Delta,
                                         t_alpha_delta=t_alpha_delta,
                                         Alpha_increase=1.5,
                                         eps=eps, mu=mu, ifr=IFR,
                                         VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                         Nk=Nk,
                                         alpha=alpha, gamma=gamma, r=r,
                                         behaviour=behaviour_rate,
                                         behaviour_bool=True,
                                         basin=basin)
            elif model == '_09_' or model == '_10_':
                results = SEIR_variation(start_date_org=start_date_org, end_date=end_date,
                                         vaxstart_date=vaxstart_date, t_alpha_org=t_alpha_org,
                                         start_week_delta=int(t0), i0_q=i0_q, r0_q=r0_q, R0=R0, Delta=Delta,
                                         t_alpha_delta=t_alpha_delta,
                                         Alpha_increase=1.5,
                                         eps=eps, mu=mu, ifr=IFR,
                                         VE=VE, VES=VES, VE2=VE2, VES2=VES2,
                                         Nk=Nk,
                                         alpha=alpha, gamma=gamma, r=r,
                                         behaviour=behaviour_rate,
                                         behaviour_bool=True,
                                         basin=basin)
            NC = results['daily_noncompliant']
            C = results['daily_compliant']
            # print(NC)
            # print(C)
            noncompliant_samples.append(NC)
            compliant_samples.append(C)

    np.savez_compressed(f"./posteriors/NC_C_samples/{basin}{model}noncompliant_samples.npz", noncompliant_samples)
    np.savez_compressed(f"./posteriors/NC_C_samples/{basin}{model}compliant_samples.npz", compliant_samples)
