import numpy as np
from typing import List
from numba import jit
import pandas as pd
from datetime import datetime
from utils import get_beta
from datetime import datetime, timedelta
from numba import jit
from functions2wave_ import import_country, get_totR, update_contacts, get_contacts

ncomp = 14
nage = 16


def SEIR_cf_mobility(start_date_org,
                  end_date,
                  start_week_delta,
                  vaxstart_date,
                  t_alpha_org,
                  t_alpha_delta,
                  i0_q: float,
                  r0_q: float,
                  R0: float,
                  eps: float,
                  mu: float,
                  ifr: List[float],
                  Delta: int,
                  VES: float,
                  VE: float,
                  VES2: float,
                  VE2: float,
                  # Delta_std: float,
                  Nk: List[float],
                  Alpha_increase: float,
                  vaccine: bool,
                  # C : List[List[float]],
                  # detection_rate : float,
                  basin: str):
    """
    SEIR model with mobility data used to modulate the force of infection.
    Parameters
    ----------
        @param inf_t0 (int): initial number of infected
        @param rec_t0 (int): initial number of recovered
        @param Nk (List[float]): number of individuals in different age groups
        @param r (List[float]): contact reduction parameters
        @param T (int): simulation steps
        @param R0 (float): basic reproductive number
        @param eps (float): inverse of latent period
        @param mu (float): inverse of infectious period
        @param ifr (List[float]): infection fatality rate by age groups
        @param Delta (float): delay in deaths (mean)
        @param Delta_std (float): delay in deaths (std)
        @param C (List[List[float]]): contact matrix
        @param detection_rate (float): fraction of deaths that are reported
        @param dates (List[datetime]): list of simulation dates
        @param hemisphere (int): hemisphere (0: north, 1: tropical, 2: south)
        @param seasonality_min (float): seasonality parameter
        @param deaths_delay (str): method to calculate deaths delay. Defaults to "fixed" (alternative is "gamma")

    Return
    ------
        @return: dictionary of compartments and deaths

            S: 0, E: 1, I: 2, R: 3,
            SV: 4, EV: 5, IV: 6, RV: 7
            E_a: 8, I_a: 9, R_a:10
            EV_a: 11, IV_a: 12, R_a: 13
    """

    VEM = 1- ((1-VE)/(1-VES))
    VEM2 = 1 - ((1 - VE2) / (1 - VES2))

    start_date = start_date_org + timedelta(weeks=int((start_week_delta)))
    T = (end_date - start_date).days

    T_nonvax = (vaxstart_date - start_date).days

    t_alpha = t_alpha_org + timedelta(days=int((t_alpha_delta)))
    t_alpha_i = (t_alpha - start_date).days

    # number of age groups
    n_age = len(Nk)

    country_dict = import_country(basin)

    deltadays = end_date - start_date
    dates = [start_date + timedelta(days=d) for d in range(deltadays.days)]

    # contact matrix
    # for no mobility
    C = get_contacts(country_dict)

    # to get beta, we use the original C
    Cs = {}
    for i in dates:
        Cs[i] = update_contacts(country_dict, i)


    #vaccine preparation
    if basin=='British Columbia':
        # get daily rate of vaccine for age group: rV_age
        file = pd.read_csv("../data/regions/" + basin + "/epidemic/vac_1st_age_correct_ver(agegroup_adjust).csv",
                           index_col=0)
        date_uni = np.unique(file['week_end'])
        rV_age = {}
        for d in date_uni:
            rV_age[d] = list(file.loc[file['week_end'] == d, 'rV'])
    else:
        # get daily rate of vaccine for age group: rV_age
        file = pd.read_csv("../data/regions/" + basin + "/epidemic/vac_1st_age_correct_ver(agegroup_adjust).csv",
                           index_col=0)
        date_uni = np.unique(file['date'])
        rV_age = []
        for d in date_uni:
            rV_age.append(list(file.loc[file['date'] == d, 'rV']))

    # initialize compartments and set initial conditions (S: 0, SB: 1, E: 2, I: 3, R: 4)
    compartments, deaths = np.zeros((ncomp, nage, T)), np.zeros((nage, T))
    infections = np.zeros((nage, T))

    # initial i0 and r0
    if basin == 'Sao Paulo':
        # I0
        df_inf = pd.read_csv(r"../data/epidemic_raw/Brazil_SP_death_case.csv")
        df_inf.date = pd.to_datetime(df_inf.date)
        new_pos = df_inf.loc[df_inf['date'] == start_date]['new_confirmed'].values[0]
        # for i in range(0,7):
        #    new_pos += df_inf.loc[df_inf['date'] == start_date-timedelta(days=i)]['new_confirmed'].values[0]
        i0 = new_pos * i0_q
        # R(t=0)
        recovered_accu = df_inf.loc[df_inf['date'] < start_date]['new_confirmed'].sum()
        r0 = recovered_accu * r0_q
    if basin == 'British Columbia':
        # I0
        df_inf = pd.read_csv(r"../data/regions/British Columbia/epidemic/cases.csv")
        df_inf.date = pd.to_datetime(df_inf.date)
        new_pos = df_inf.loc[df_inf['date'] == start_date]['value_daily'].values[0]
        i0 = new_pos * i0_q
        # R(t=0)
        recovered_accu = df_inf.loc[df_inf['date'] == start_date - timedelta(days=1)]['value'].values[0]
        r0 = recovered_accu * r0_q

    if basin == 'Lombardy':
        # I0
        df_inf = pd.read_csv(r"../data/regions/Lombardy/epidemic/existing_positive_italy.csv")
        df_inf.date = pd.to_datetime(df_inf.date)
        new_pos = df_inf.loc[df_inf['date'] == start_date]['daily_positive'].values[0]
        i0 = new_pos * i0_q
        df_inf2 = pd.read_csv(r"../data/epidemic_raw/daily_positive_italy.csv")
        df_inf2.date = pd.to_datetime(df_inf2.date)
        recovered_accu = df_inf2.loc[df_inf2['date'] < start_date]['daily_positive'].sum()
        r0 = recovered_accu * r0_q

    # distribute intial infected and recovered among age groups
    for age in range(n_age):
        # I
        inf_t0_age = int(i0 * Nk[age] / np.sum(Nk))
        compartments[2, age, 0] = int(int(inf_t0_age) * (1 / mu) / ((1 / mu) + (1 / eps)))
        # E
        compartments[1, age, 0] = inf_t0_age - compartments[2, age, 0]
        # R
        compartments[3, age, 0] = int(r0 * Nk[age] / np.sum(Nk))
        # S
        compartments[0, age, 0] = Nk[age] - (
                    compartments[1, age, 0] + compartments[2, age, 0] + compartments[3, age, 0])

    # simulate
    for t in np.arange(1, T, 1):
        # compute beta
        beta = get_beta(R0, mu, Nk, Cs[dates[t]])
        beta_Alpha = get_beta(Alpha_increase * R0, mu, Nk, Cs[dates[t]])
        force_inf = np.sum(beta * C * (compartments[2, :, t - 1] + compartments[6, :, t - 1]) / Nk,
                           axis=1)
        force_inf_Alpha = np.sum(beta_Alpha * C * (compartments[9, :, t - 1] + compartments[12, :, t - 1]) / Nk, axis=1)

        if vaccine==True:
            # get vaccine
            if basin == 'Lombardy' or basin == 'Sao Paulo':
                if t >= T_nonvax + 1:
                    V_S = get_vaccinated(basin, rV_age, compartments[:, :, t - 1].ravel('F'), t - 1 - T_nonvax,
                                         start_date,
                                         Nk)  # next time i.e. i step we have new vaccinated individuals VS VS_NC
                    for age in range(nage):
                        if compartments[0, age, t - 1] < V_S[age]:
                            compartments[4, age, t - 1] += compartments[0, age, t - 1]
                            compartments[0, age, t - 1] = 0
                        else:
                            compartments[0, age, t - 1] -= V_S[age]  # S -
                            compartments[4, age, t - 1] += V_S[age]  # SV +
                else:
                    V_S = np.zeros(nage)
            else:
                if t >= T_nonvax + 1:
                    # only have vax data every weekend not everyday, so get vaccined if t in rV dates
                    # t in rV dates
                    if (start_date + timedelta(days=int(t))).strftime('%Y-%m-%d') in date_uni:
                        V_S = get_vaccinated(basin, rV_age, compartments[:, :, t - 1].ravel('F'), t, start_date,
                                             Nk)  # next time i.e. i step we have new vaccinated individuals VS VS_NC
                        for age in range(nage):
                            if compartments[0, age, t - 1] < V_S[age]:
                                compartments[4, age, t - 1] += compartments[0, age, t - 1]
                                compartments[0, age, t - 1] = 0
                            else:
                                compartments[0, age, t - 1] -= V_S[age]  # S -
                                compartments[4, age, t - 1] += V_S[age]  # SV +
                    # t not in rV dates
                    else:
                        for age in range(nage):
                            compartments[0, age, t - 1] -= 0
                            compartments[4, age, t - 1] += 0
                else:
                    V_S = np.zeros(nage)


        # compute transitions
        prob_S_to_E = force_inf  # not probability but rate as force_int might be > 1
        prob_S_to_E_alpha = force_inf_Alpha
        total_leaving_from_S = np.random.binomial(compartments[0, :, t - 1].astype(int),
                                                  1 - np.exp(-(prob_S_to_E + prob_S_to_E_alpha)))
        new_E = np.random.binomial(total_leaving_from_S, np.array(
            [a / b if b != 0 else 0 for a, b in zip(prob_S_to_E, prob_S_to_E + prob_S_to_E_alpha)]))
        new_E_Alpha = total_leaving_from_S - new_E

        prob_SV_to_EV = (1 - VES) * force_inf
        prob_SV_to_EV_alpha = (1 - VES2) * force_inf_Alpha
        total_leaving_from_SV = np.random.binomial(compartments[4, :, t - 1].astype(int),
                                                   1 - np.exp(-(prob_SV_to_EV + prob_SV_to_EV_alpha)))
        new_EV = np.random.binomial(total_leaving_from_SV, np.array(
            [a / b if b != 0 else 0 for a, b in zip(prob_SV_to_EV, prob_SV_to_EV + prob_SV_to_EV_alpha)]))
        new_EV_Alpha = total_leaving_from_SV - new_EV

        new_I = np.random.binomial(compartments[1, :, t - 1].astype(int), eps)
        new_R = np.random.binomial(compartments[2, :, t - 1].astype(int), mu)
        new_IV = np.random.binomial(compartments[5, :, t - 1].astype(int), eps)
        new_RV = np.random.binomial(compartments[6, :, t - 1].astype(int), mu)
        new_I_Alpha = np.random.binomial(compartments[8, :, t - 1].astype(int), eps)
        new_R_Alpha = np.random.binomial(compartments[9, :, t - 1].astype(int), mu)
        new_IV_Alpha = np.random.binomial(compartments[11, :, t - 1].astype(int), eps)
        new_RV_Alpha = np.random.binomial(compartments[12, :, t - 1].astype(int), mu)

        #  update next step solution
        # S
        compartments[0, :, t] = compartments[0, :, t - 1] - new_E - new_E_Alpha
        # E
        compartments[1, :, t] = compartments[1, :, t - 1] + new_E - new_I
        # I
        compartments[2, :, t] = compartments[2, :, t - 1] + new_I - new_R
        # R
        compartments[3, :, t] = compartments[3, :, t - 1] + new_R
        # SV
        compartments[4, :, t] = compartments[4, :, t - 1] - new_EV - new_EV_Alpha
        # EV
        compartments[5, :, t] = compartments[5, :, t - 1] + new_EV - new_IV
        # IV
        compartments[6, :, t] = compartments[6, :, t - 1] + new_IV - new_RV
        # RV
        compartments[7, :, t] = compartments[7, :, t - 1] + new_RV

        # E_Alpha
        compartments[8, :, t] = compartments[8, :, t - 1] + new_E_Alpha - new_I_Alpha
        # I_Alpha
        compartments[9, :, t] = compartments[9, :, t - 1] + new_I_Alpha - new_R_Alpha
        # R_Alpha
        compartments[10, :, t] = compartments[10, :, t - 1] + new_R_Alpha
        # E_Alpha_V
        compartments[11, :, t] = compartments[11, :, t - 1] + new_EV_Alpha - new_IV_Alpha
        # I_Alpha_V
        compartments[12, :, t] = compartments[12, :, t - 1] + new_IV_Alpha - new_RV_Alpha
        # R_Alpha_V
        compartments[13, :, t] = compartments[13, :, t - 1] + new_RV_Alpha

        #for i in range(ncomp):
        #    if compartments[i, :, t].any()<0:
        #        print(i, t)
        #        print(compartments[i, :, t])

        # compute deaths
        if (t - 1) + Delta < deaths.shape[1]:
            deaths[:, (t - 1) + int(Delta)] += np.random.binomial((new_R + new_R_Alpha), ifr)
            deaths[:, (t - 1) + int(Delta)] += np.random.binomial(new_RV, np.array(ifr) * (1 - VEM))
            deaths[:, (t - 1) + int(Delta)] += np.random.binomial(new_RV_Alpha, np.array(ifr) * (1 - VEM2))
        infections[:, (t - 1)] = compartments[2, :, t] + compartments[6, :, t]+ compartments[9, :, t]+ compartments[12, :, t]

        if t==t_alpha_i:
            # I(t_alpha) initial
            for age in range(nage):
                compartments[9, age, t] = (compartments[2, age, t]+compartments[6, age, t]) * 0.01
        if (basin == 'Sao Paulo') and (t>=t_alpha_i):
            eps = 1.0/3.0


    deaths_sum = deaths.sum(axis=0)
    df_deaths = pd.DataFrame(data={"deaths": deaths_sum}, index=pd.to_datetime(dates))
    deaths_week = df_deaths.resample("W").sum()

    infections_sum = infections.sum(axis=0)
    df_infections = pd.DataFrame(data={"infections": infections_sum}, index=pd.to_datetime(dates))
    infections_week = df_infections.resample("W").sum()

    weekly_deaths = list(deaths_week.deaths.values)
    weekly_infections = list(infections_week.infections.values)
    for i in range(start_week_delta):
        weekly_deaths.insert(0, 0)
        weekly_infections.insert(0, 0)

    return weekly_deaths, weekly_infections


def get_vaccinated(basin, rV_age, y, i, start_date, Nk):
    """
        This functions compute the n. of S individuals that will receive a vaccine in the next step
            :param rV (float): vaccination rate
            #:param Nk (array): number of individuals in different age groups
            :param y (array): compartment values at time t
            :param i (int): this time step from integrate_BV function
            :return: returns the two arrays of n. of vaccinated in different age groups for S and S_NC in the next step
        """
    # list of n. of vaccinated in each age group for S and S_NC in this step
    V_S = np.zeros(nage)
    if basin == 'Lombardy':
        # this time step in the integrate_BV function
        t_rv = i
        for age in range(nage):
            if y[(ncomp * age) + 0] <= 0:
                V_S[age] = 0
                continue
            elif age == 0:
                V_S[age] = 0  # 0 - 4 years
            elif age == 1:
                rV = rV_age[t_rv][0]
                V_S[age] = round(rV * Nk[age])  # 5 - 9 years
            elif age == 2 or age == 3:
                rV = rV_age[t_rv][1]
                V_S[age] = round(rV * Nk[age])  # 10 - 19 years
            elif age == 4 or age== 5:
                rV = rV_age[t_rv][2]
                V_S[age] = round(rV * Nk[age])  # 20 - 29 years
            elif age == 6 or age== 7:
                rV = rV_age[t_rv][3]
                V_S[age] = round(rV * Nk[age])  # 30 - 39 years
            elif age == 8 or age== 9:
                rV = rV_age[t_rv][4]
                V_S[age] = round(rV * Nk[age])  # 40 - 49 years
            elif age == 10 or age== 11:
                rV = rV_age[t_rv][5]
                V_S[age] = round(rV * Nk[age])  # 50 - 59 years
            elif age == 12 or age== 13:
                rV = rV_age[t_rv][6]
                V_S[age] = round(rV * Nk[age])  # 60 - 69 years
            elif age == 14 or age== 15:
                rV = (rV_age[t_rv][7]+rV_age[t_rv][8]+rV_age[t_rv][9])/3
                V_S[age] = round(rV * Nk[age])  # 70 - 74 years and 75+
        # print("V_S:",V_S)
        # print("V_S_NC:", V_S_NC)
        return V_S



    elif basin=='Sao Paulo':
        t_rv = i

        for age in range(nage):
            if y[(ncomp * age) + 0] <= 0:
                V_S[age] = 0
                continue
            if age % 2 == 1:
                rV = rV_age[t_rv][int((age - 1) / 2)]
                V_S[age] = round(rV * Nk[age])  # 5 - 9 years
            else:
                rV = rV_age[t_rv][int(age / 2)]
                V_S[age] = round(rV * Nk[age])  # 5 - 9 years
        # print("V_S:",V_S)
        # print("V_S_NC:", V_S_NC)
        return V_S

    elif basin=='British Columbia':
        # this time step in the integrate_BV function
        t_rv = (start_date + timedelta(days=int(i))).strftime('%Y-%m-%d')  # date str

        if t_rv < "2021-01-09":

            den_16_69 = 0
            for age in range(3, 14):
                den_16_69 += Nk[age]
            for age in range(nage):
                if y[(ncomp * age) + 0] <= 0:
                    V_S[age] = 0
                    continue
                elif age == 0 or age == 1 or age == 2:
                    rV = rV_age[t_rv][0]  # 0-15
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[0] + Nk[1] + Nk[2]))
                elif age == 14:
                    rV = rV_age[t_rv][2]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[14] + Nk[15]))  # 70 - 74 years
                elif age == 15:
                    rV1 = rV_age[t_rv][2]
                    rV2 = rV_age[t_rv][3]
                    V_S[age] = round(rV1 * Nk[age] * Nk[age] / (Nk[14] + Nk[15]) + rV2 * Nk[age])  # 75 + years
                else:
                    rV = rV_age[t_rv][1]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / den_16_69)  # 75 + years

        elif t_rv < '2021-04-17':
            den_18_69 = 0
            for age in range(4, 14):
                den_18_69 += Nk[age]
            for age in range(nage):
                if y[(ncomp * age) + 0] <= 0:
                    V_S[age] = 0
                    continue
                elif age == 0 or age == 1 or age == 2 or age == 3:
                    rV = rV_age[t_rv][0]  # 0-15
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[0] + Nk[1] + Nk[2] + Nk[3]))
                elif age == 14:
                    rV = rV_age[t_rv][2]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[14] + Nk[15]))  # 70 - 74 years
                elif age == 15:
                    rV1 = rV_age[t_rv][2]
                    rV2 = rV_age[t_rv][3]
                    V_S[age] = round(rV1 * Nk[age] * Nk[age] / (Nk[14] + Nk[15]) + rV2 * Nk[age])  # 75 + years
                else:
                    rV = rV_age[t_rv][1]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / den_18_69)  # 75 + years

        elif t_rv < '2021-06-05':
            den_18_69 = 0
            for age in range(4, 14):
                den_18_69 += Nk[age]

            for age in range(nage):
                if y[(ncomp * age) + 0] <= 0:
                    V_S[age] = 0
                    continue
                elif age == 0 or age == 1 or age == 2 or age == 3:
                    rV = rV_age[t_rv][0]  # 0-17
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[0] + Nk[1] + Nk[2] + Nk[3]))
                elif age == 4 or age == 5:
                    rV = rV_age[t_rv][1]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[4] + Nk[5]))  # 18- 29 years
                elif age == 6 or age == 7:
                    rV = rV_age[t_rv][2]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[6] + Nk[7]))  # 30 - 39 years
                elif age == 8 or age == 9:
                    rV = rV_age[t_rv][3]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[8] + Nk[9]))  # 40 - 49 years
                elif age == 10 or age == 11:
                    rV = rV_age[t_rv][4]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[10] + Nk[11]))  # 50 - 59 years
                elif age == 12 or age == 13:
                    rV = rV_age[t_rv][5]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[12] + Nk[13]))  # 60 - 69 years
                elif age == 14:
                    rV = rV_age[t_rv][6]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[14] + Nk[15]))  # 70 - 75 years
                else:
                    rV1 = rV_age[t_rv][6]
                    rV2 = rV_age[t_rv][7]
                    V_S[age] = round(rV1 * Nk[age] * Nk[age] / (Nk[14] + Nk[15]) + rV2 * Nk[age])  # 75 + years

        else:
            for age in range(nage):
                if y[(ncomp * age) + 0] <= 0:
                    V_S[age] = 0
                    continue
                elif age == 0:
                    rV = rV_age[t_rv][0]  # 0-4
                    V_S[age] = round(rV * Nk[age])
                elif age == 1 or age == 2:
                    rV = rV_age[t_rv][1]  # 5-11
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[1] + Nk[2]))  # 18- 29 years
                elif age == 3:
                    rV = rV_age[t_rv][2]
                    V_S[age] = round(rV * Nk[age])  # 15 - 19 years
                elif age == 4 or age == 5:
                    rV = rV_age[t_rv][3]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[8] + Nk[9]))  # 18 - 29 years
                elif age == 6 or age == 7:
                    rV = rV_age[t_rv][4]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[8] + Nk[9]))  # 30 - 39 years
                elif age == 8 or age == 9:
                    rV = rV_age[t_rv][5]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[8] + Nk[9]))  # 40 - 49 years
                elif age == 10 or age == 11:
                    rV = rV_age[t_rv][6]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[10] + Nk[11]))  # 50 - 59 years
                elif age == 12 or age == 13:
                    rV = rV_age[t_rv][7]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[12] + Nk[13]))  # 60 - 69 years
                elif age == 14:
                    rV = rV_age[t_rv][8]
                    V_S[age] = round(rV * Nk[age] * Nk[age] / (Nk[14] + Nk[15]))  # 70 - 75 years
                else:
                    rV1 = rV_age[t_rv][8]
                    rV2 = rV_age[t_rv][9]
                    V_S[age] = round(rV1 * Nk[age] * Nk[age] / (Nk[14] + Nk[15]) + rV2 * Nk[age])  # 75 + years
        return V_S
