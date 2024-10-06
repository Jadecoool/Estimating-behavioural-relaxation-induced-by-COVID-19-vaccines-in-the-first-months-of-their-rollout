import numpy as np
from typing import List
from numba import jit
import pandas as pd
from datetime import datetime
from utils import get_beta
from datetime import datetime, timedelta
from numba import jit
from functions2wave_ import import_country, get_totR, update_contacts, get_contacts

ncomp = 10
nage = 16


def SEIR_cf_mobility(start_date_org,
                  end_date,
                  start_week_delta,
                  vaxstart_date,
                  i0_q: float,
                  r0_q: float,
                  R0: float,
                  alpha: float,
                  gamma: float,
                  r: float,
                  eps: float,
                  mu: float,
                  ifr: List[float],
                  Delta: int,
                  VES: float,
                  VE: float,
                  Nk: List[float],
                  behaviour: str,
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
            S_NC: 8, SV_NC: 9
    """

    VEM = 1- ((1-VE)/(1-VES))

    start_date = start_date_org + timedelta(weeks=int((start_week_delta)))
    T = (end_date - start_date).days
    T_nonvax = (vaxstart_date - start_date).days

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

    # I0
    df_inf = pd.read_csv(r"../data/epidemic_raw/data_2023-Oct-26 London_cases.csv")
    df_inf.date = pd.to_datetime(df_inf.date)
    new_pos = df_inf.loc[df_inf['date'] == start_date]['newCases'].values[0]
    i0 = new_pos * i0_q

    # R(t=0)
    recovered_accu = df_inf.loc[df_inf['date'] == start_date - timedelta(days=1)]['cumCases'].values[0]
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

    V=0
    Vt = []
    # simulate
    for t in np.arange(1, T, 1):
        if t >= 2:
            vt = V / sum(Nk)
        else:
            vt =0

        if t >= 2:
            dt = 100000 * (sum(sum(deaths[:, 0 : t - 1])) - sum(sum(deaths[:, 0 : t - 2])))/ np.sum(Nk)
            #print(dt)
        else:
            dt = 0

        if behaviour == "constant_rate":
            if t>= T_nonvax + 1:
                nc_rate = [alpha for i in range(nage)]
                c_rate = [gamma for i in range(nage)]
            else:
                nc_rate = [0 for i in range(nage)]
                c_rate = [0 for i in range(nage)]
        elif behaviour == "vaccine_rate":
            nc_rate = [vt * alpha for i in range(nage)]
            c_rate = [dt * gamma for i in range(nage)]



        # compute force of infection
        beta = get_beta(R0, mu, Nk, Cs[dates[t]])
        force_inf = np.sum(beta * C * (compartments[2, :, t - 1] + compartments[6, :, t - 1]) / Nk, axis=1)

        if t >= T_nonvax + 1:
            V_S, V_S_NC = get_vaccinated(rV_age, compartments[:, :, t - 1].ravel('F'), t - 1 - T_nonvax,
                                 Nk)  # next time i.e. i step we have new vaccinated individuals VS VS_NC
            for age in range(nage):
                if compartments[0, age, t - 1] < V_S[age]:
                    compartments[4, age, t - 1] += compartments[0, age, t - 1]
                    compartments[0, age, t - 1] = 0
                else:
                    compartments[0, age, t - 1] -= V_S[age]  # S -
                    compartments[4, age, t - 1] += V_S[age]  # SV

                if compartments[8, age, t - 1] < V_S_NC[age]:
                    compartments[9, age, t - 1] += compartments[8, age, t - 1]
                    compartments[8, age, t - 1] = 0
                else:
                    compartments[8, age, t - 1] -= V_S_NC[age]  # V -
                    compartments[9, age, t - 1] += V_S_NC[age]  # V_NC
        else:
            V_S, V_S_NC = np.zeros(nage), np.zeros(nage)

        # compute transitions

        # from S to S_NC, E
        prob_S_to_E = force_inf  # not probability but rate as force_int might be > 1
        prob_S_to_NC = nc_rate

        total_leaving_from_S = np.random.binomial(compartments[0, :, t - 1].astype(int),
                                                  1 - np.exp(-(prob_S_to_E + prob_S_to_NC)))

        new_E = np.random.binomial(total_leaving_from_S, np.array(
            [a / b if b != 0 else 0 for a, b in zip(1 - np.exp(-(prob_S_to_E)), 1 - np.exp(
                -(prob_S_to_E + prob_S_to_NC)))]))
        new_S_NC = total_leaving_from_S - new_E

        # from SV to SV_NC, EV
        prob_SV_to_EV = (1 - VES) * force_inf
        prob_SV_to_NC = nc_rate

        total_leaving_from_SV = np.random.binomial(compartments[4, :, t - 1].astype(int),
                                                   1 - np.exp(-(prob_SV_to_EV + prob_SV_to_NC)))
        new_EV = np.random.binomial(total_leaving_from_SV, np.array(
            [a / b if b != 0 else 0 for a, b in zip(1 - np.exp(-prob_SV_to_EV),
                                                    1 - np.exp(
                                                        -(prob_SV_to_EV + prob_SV_to_NC)))]))
        new_SV_NC = total_leaving_from_SV - new_EV

        # from S_NC to S, E, E_alpha
        prob_NC_to_E = r * force_inf  # not probability but rate as force_int might be > 1
        prob_NC_to_S = c_rate

        # print('NC2E',prob_NC_to_E)
        # print('NC2E_ALPHA',prob_NC_to_E_alpha)
        # print('NC2S', prob_NC_to_S)
        # print('e-(x)', np.exp(-(prob_NC_to_E + prob_NC_to_E_alpha + prob_NC_to_S)))

        total_leaving_from_NC = np.random.binomial(compartments[8, :, t - 1].astype(int),
                                                   1 - np.exp(-(prob_NC_to_E + prob_NC_to_S)))

        new_E_ = np.random.binomial(total_leaving_from_NC, np.array(
            [a / b if b != 0 else 0 for a, b in zip(1 - np.exp(-(prob_NC_to_E)),
                                                    1 - np.exp(-(prob_NC_to_E + prob_NC_to_S)))]))
        new_NC_S = total_leaving_from_NC - new_E_


        # from S_V_NC to SV, EV, EV_alpha
        prob_V_NC_to_EV = r * (1 - VES) * force_inf  # not probability but rate as force_int might be > 1

        prob_V_NC_to_SV = c_rate

        total_leaving_from_V_NC = np.random.binomial(compartments[9, :, t - 1].astype(int), 1 - np.exp(
            -(prob_V_NC_to_EV + prob_V_NC_to_SV)))

        new_EV_ = np.random.binomial(total_leaving_from_V_NC, np.array(
            [a / b if b != 0 else 0 for a, b in zip(1 - np.exp(-(prob_V_NC_to_EV)), 1 - np.exp(
                -(prob_V_NC_to_EV + prob_V_NC_to_SV)))]))
        new_V_NC_S = total_leaving_from_V_NC - new_EV_


        new_I = np.random.binomial(compartments[1, :, t - 1].astype(int), eps)
        new_R = np.random.binomial(compartments[2, :, t - 1].astype(int), mu)
        new_IV = np.random.binomial(compartments[5, :, t - 1].astype(int), eps)
        new_RV = np.random.binomial(compartments[6, :, t - 1].astype(int), mu)


        #  update next step solution
        # S
        compartments[0, :, t] = compartments[0, :, t - 1] - new_E - new_S_NC + new_NC_S
        # E
        compartments[1, :, t] = compartments[1, :, t - 1] + new_E + new_E_ - new_I
        # I
        compartments[2, :, t] = compartments[2, :, t - 1] + new_I - new_R
        # R
        compartments[3, :, t] = compartments[3, :, t - 1] + new_R

        # SV
        compartments[4, :, t] = compartments[4, :, t - 1] - new_EV - new_SV_NC + new_V_NC_S
        # EV
        compartments[5, :, t] = compartments[5, :, t - 1] + new_EV + new_EV_ - new_IV
        # IV
        compartments[6, :, t] = compartments[6, :, t - 1] + new_IV - new_RV
        # RV
        compartments[7, :, t] = compartments[7, :, t - 1] + new_RV

        # S_NC
        compartments[8, :, t] = compartments[8, :, t - 1] + new_S_NC - new_NC_S - new_E_
        # S_NC_V
        compartments[9, :, t] = compartments[9, :, t - 1] + new_SV_NC - new_V_NC_S - new_EV_

        # compute deaths
        if (t - 1) + Delta < deaths.shape[1]:
            deaths[:, (t - 1) + int(Delta)] += np.random.binomial( new_R, ifr)
            deaths[:, (t - 1) + int(Delta)] += np.random.binomial( new_RV, np.array(ifr)*(1 - VEM))
        infections[:, (t - 1)] = compartments[6, :, t] + compartments[2, :, t]

        Vt.append(np.sum(V_S) + np.sum(V_S_NC) )
        V += Vt[-1]

    deaths_sum = deaths.sum(axis=0)
    df_deaths = pd.DataFrame(data={"deaths": deaths_sum}, index=pd.to_datetime(dates))
    deaths_week = df_deaths.resample("W").sum()

    infections_sum = infections.sum(axis=0)
    df_infections = pd.DataFrame(data={"infections": infections_sum}, index=pd.to_datetime(dates))
    infections_week = df_infections.resample("W").sum()

    return deaths_week.deaths.values[5 - start_week_delta:], infections_week.infections.values[5 - start_week_delta:]


def get_vaccinated(rV_age, y, i, Nk):
    """
        This functions compute the n. of S individuals that will receive a vaccine in the next step
            :param rV (float): vaccination rate
            #:param Nk (array): number of individuals in different age groups
            :param y (array): compartment values at time t
            :param i (int): this time step from integrate_BV function
            :return: returns the two arrays of n. of vaccinated in different age groups for S and S_NC in the next step
    """

    t_rv = i

    V_S, V_S_NC = np.zeros(nage), np.zeros(nage)

    for age in range(nage):
        den_age = y[(ncomp * age) + 0] + y[(ncomp * age) + 8]

        if y[(ncomp * age) + 0] <= 0:
            V_S[age] = 0
            V_S_NC[age] = 0
            continue
        if age == 0:
            V_S[age] = 0
            V_S_NC[age] = 0
        elif age == 1:
            rV = rV_age[t_rv][8]
            V_S[age] = round(rV * y[(ncomp * age) + 0] / den_age * Nk[age])  # 5 - 9 years
            V_S_NC[age] = round(rV * y[(ncomp * age) + 8] / den_age * Nk[age])
        elif age <= 9:
            rV = rV_age[t_rv][age - 2]
            V_S[age] = round(rV * y[(ncomp * age) + 0] / den_age * Nk[age])
            V_S_NC[age] = round(rV * y[(ncomp * age) + 8] / den_age * Nk[age])
        else:
            rV = rV_age[t_rv][age - 1]
            V_S[age] = round(rV * y[(ncomp * age) + 0] / den_age * Nk[age])
            V_S_NC[age] = round(rV * y[(ncomp * age) + 8] / den_age * Nk[age])
    # print("V_S:",V_S)
    # print("V_S_NC:", V_S_NC)
    return V_S, V_S_NC