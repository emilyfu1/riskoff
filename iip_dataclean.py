import pandas as pd
import numpy as np
from dotenv import dotenv_values, find_dotenv
import os

# this looks for your configuration file and then reads it as a dictionary
config = dotenv_values(find_dotenv())

# set path using the dictionary key for which one you want
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'

def iip_dataclean(data):
    # country codes
    imf_codes = pd.read_csv(path_rawdata + "Country_iso_codes.csv")
    
    # add iso3 codes in order to merge data
    iso3_map = imf_codes.set_index('IMF').to_dict()['iso3']
    data['iso3']= data['Country Code'].map(iso3_map)
    # fill euro value
    data.loc[data['Country Name'] == 'Euro Area', 'iso3'] = 'EUR'

    # remove unnecessary rows/columns
    data = data[data['Attribute'] == 'Value']
    data = data.drop(['Country Name', 'Country Code', 'Attribute', 'Unnamed: 139', 'Unnamed: 141'], axis=1, errors='ignore')

    # Specify the columns to keep as identifiers
    id_vars = ['Indicator Name', 'Indicator Code', 'iso3']

    # Use the melt function to convert the DataFrame to long format
    iip_long = pd.melt(data, id_vars=id_vars, var_name='date', value_name='value')

    # remove the weird strings by just changing them to nan
    iip_long.loc[iip_long['value'] == 'C', 'value'] = np.nan
    iip_long.loc[iip_long['value'] == '-', 'value'] = np.nan
    iip_long.loc[iip_long['value'] == 'K', 'value'] = np.nan
    # i'm considering zeros as nans as well just based on how the zeros are distributed
    iip_long.loc[iip_long['value'] == '0', 'value'] = np.nan

    # convert to datetime
    iip_long['date'] = pd.to_datetime(iip_long['date'])
    # end-of-month values
    iip_long['date'] = iip_long['date'] + pd.offsets.MonthEnd(0)

    # convert to numeric
    iip_long['value'] = pd.to_numeric(iip_long['value'])
    
    return iip_long