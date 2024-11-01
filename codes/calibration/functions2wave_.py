#  libraries
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta


def get_totR(path, start_date, country):
    """
    This function import the total number or non-susceptible individuals for a given country up to start_date
        :param path (string): path to the data folder
        :param start_date (datetime): starting date
        :param country (string): country
        :return
    """

    # import projections df
    df_inf = pd.read_csv(path + "/epidemic_raw/data_2023-Oct-26 London_cases.csv")
    df_inf.date = pd.to_datetime(df_inf.date)

    # loc country and period
    #df_inf_country = df_inf[(df_inf.Entity == country.replace("_", " ")) & (df_inf.Date < start_date)].reset_index(
    #    drop=True)

    #cols = ['Daily new estimated infections of COVID-19 (ICL, mean)',
     #       'Daily new estimated infections of COVID-19 (IHME, mean)',
      #      'Daily new estimated infections of COVID-19 (YYG, mean)',
       #     'Daily new estimated infections of COVID-19 (LSHTM, median)']
       # df_inf_country[cols].sum().mean()
    return df_inf.loc[df_inf['date']==start_date-timedelta(days=1)]['cumCases'].values[0]





def import_country(country, path_to_data=r"../data"):
    """
    This function returns all data needed for a specific country
        :param country (string): name of the country
        :param path_to_data (string): path to the countries folder
        :return dict of country data (country_name, work, school, home, other_locations, Nk, epi_data)
    """

    # import contacts matrix
    work            = np.loadtxt(path_to_data + "/regions/" + country + "/contacts_matrix/contacts_work.csv", delimiter=",")
    school          = np.loadtxt(path_to_data + "/regions/" + country + "/contacts_matrix/contacts_school.csv", delimiter=",")
    home            = np.loadtxt(path_to_data + "/regions/" + country + "/contacts_matrix/contacts_home.csv", delimiter=",")
    other_locations = np.loadtxt(path_to_data + "/regions/" + country + "/contacts_matrix/contacts_other_locations.csv", delimiter=",")

    # import demographic
    Nk = pd.read_excel(path_to_data + "/regions/" + country + "/demographic/pop_5years.xlsx").total.values

    # import epidemiological data
    #cases  = pd.read_csv(path_to_data + "/epidemic_raw/data_2023-Oct-26 London_cases.csv")
    deaths = pd.read_csv(path_to_data + "/regions/" + country + "/epidemic/deaths.csv")

    # import restriction
    school_reductions = pd.read_csv(path_to_data + "/regions/" + country + "/restriction/school.csv")
    work_reductions = pd.read_csv(path_to_data + "/regions/" + country + "/restriction/work.csv")
    oth_reductions = pd.read_csv(path_to_data + "/regions/" + country + "/restriction/other_loc.csv")


    # create dict of data
    country_dict = {"country": country,
                    "contacts_work": work,
                    "contacts_school": school,
                    "contacts_home": home,
                    "contacts_other_locations": other_locations,
                    "school_red": school_reductions,
                    "work_red": work_reductions,
                    "oth_red": oth_reductions,
                    "Nk": Nk,
                    "deaths": deaths}
                    #"cases": cases}

    return country_dict


def get_beta(R0, mu, Nk, C):
    """
    Compute the transmission rate beta for a SEIR model with age groups
    Parameters
    ----------
        @param R0: basic reproductive number
        @param mu: recovery rate
        @param Nk: number of individuals in different age groups
        @param C: contact matrix
    Return
    ------
        @return: the transmission rate beta
    """
    # get seasonality adjustment
    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]
    return R0 * mu / (np.max([e.real for e in np.linalg.eig(C_hat)[0]]))


# def get_beta(R0, mu, chi, omega, f, C, Nk):
#     """
#     This functions return beta for a SEIR model with age structure
#         :param R0 (float): basic reproductive number
#         :param mu (float): recovery rate
#         :param chi (float): relative infectivity of P, A infectious
#         :param omega (float): inverse of the prodromal phase
#         :param f (float): probability of being asymptomatic
#         :param C (matrix): contacts matrix
#         :param Nk (array): n. of individuals in different age groups
#         :return: returns the rate of infection beta
#     """
#
#     C_hat = np.zeros((C.shape[0], C.shape[1]))
#     for i in range(C.shape[0]):
#         for j in range(C.shape[1]):
#             C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]
#
#     max_eV = np.max([e.real for e in np.linalg.eig(C_hat)[0]])
#     return R0 / (max_eV * (chi / omega + (1 - f) / mu + chi * f / mu))

def get_contacts(country_dict):
    #print("date", date)
    # get baseline contacts matrices
    home = country_dict["contacts_home"]
    work = country_dict["contacts_work"]
    school = country_dict["contacts_school"]
    oth_loc = country_dict["contacts_other_locations"]
    return home + school + work + oth_loc


def update_contacts(country_dict, date):
    #print("date", date)
    # get baseline contacts matrices
    home = country_dict["contacts_home"]
    work = country_dict["contacts_work"]
    school = country_dict["contacts_school"]
    oth_loc = country_dict["contacts_other_locations"]

    # get year-week
    if date.isocalendar()[1] < 10:
        year_week = str(date.isocalendar()[0]) + "-0" + str(date.isocalendar()[1])
    else:
        year_week = str(date.isocalendar()[0]) + "-" + str(date.isocalendar()[1])

    #print("year_week", year_week)
    # get work / other_loc reductions
    work_reductions = country_dict["work_red"]
    comm_reductions = country_dict["oth_red"]
    school_reductions = country_dict["school_red"]
    school_reductions["datetime"] = pd.to_datetime(school_reductions["datetime"])

    #if year_week <= "2021-30":
    omega_w = work_reductions.loc[work_reductions.year_week == year_week]["work_red"].values[0]
    omega_c = comm_reductions.loc[comm_reductions.year_week == year_week]["oth_red"].values[0]
    C1_school = school_reductions.loc[school_reductions.datetime == date]["C1M_School.closing"].values[0]

        # check we are not going below zero
    if C1_school < 0:
        C1_school = 0

    omega_s = (3 - C1_school) / 3

    # print(date)
    # print(omega_s)
    # print(omega_w)
    # print(omega_c)
    # contacts matrix with reductions
    return home + (omega_s * school) + (omega_w * work) + (omega_c * oth_loc)