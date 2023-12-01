import pandas as pd
import numpy as np
from dotenv import dotenv_values, find_dotenv
import os

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

# this looks for your configuration file and then reads it as a dictionary
config = dotenv_values(find_dotenv())

# set path using the dictionary key for which one you want
path_rawdata = os.path.abspath(config["RAWDATA"]) + '\\'
path_figures = os.path.abspath(config["FIGURES"]) + '\\'

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

# my plan for dealing with dates:
# year values: December of that year xxxx-12-31
# mid-year values: June of that year xxxx-06-30
def convert_date(row):
    year = int(row['date'][:4])
    
    if 'S' not in row['date']:
        return f'{year}-12-31'
    else:
        return f'{year}-06-30'

def filter_data_heatmap(data, indicator, agg=None):
    if 'A' in indicator:
        indicatorname = 'Assets'
    else:
        indicatorname = 'Liabilities'

    currency = indicator[-3:]
    # countries that i want
    # just showing as good a picture as possible for data availability
    countries = ['USA', 'CAN', 'KOR', 'GBR', 'JPN', 'EUR', 'IRL', 'CHE', 'AUS', 'NZL', 'SWE', 
                'NOR', 'FIN', 'DNK', 'CZE', 'ZAF', 'DEU', 'AUT', 'BEL', 'BOL', 'BRA', 'CHL', 
                'CHN', 'COL', 'EST', 'FRA', 'IDN', 'IND', 'ISL', 'ISR', 'ITA', 'MEX', 'MYS', 
                'NLD', 'POL', 'QAT', 'ROU', 'RUS', 'SAU', 'SVK', 'THA', 'TUR', 'UKR', 'HKG',
                'UGA', 'VNM', 'TWN', 'RWA', 'QAT', 'PRT', 'SYR', 'PHL', 'PAK', 'DOM', 'ARG']
    
    forheatmap = data[data['Indicator Code'] == indicator][['iso3', 'date', 'value']]

    # insert countries based on my list by creating a dataframe to merge
    df_countries = pd.DataFrame(countries, columns=['iso3'])
    # add date range to country list
    df_countries['date'] = df_countries['iso3'].apply(lambda x: [f'{year}' for year in list(set(forheatmap['date']))])
    # explode list of dates
    df_countries = df_countries.explode('date', ignore_index=True)
    if not (indicator == 'I_L_T_T_T_BP6_USD' or indicator == 'I_A_T_T_T_BP6_USD'):
        df_countries['date'] = pd.to_datetime(df_countries['date'])
    # merge with existing data
    forheatmap = pd.merge(left=forheatmap, right=df_countries, how='outer', on=['date', 'iso3'])

    if agg == 'firstcountry':
        version = 'first country'
    else:
        agg = 'counterpart country'
    if indicator == 'I_L_T_T_T_BP6_USD' or indicator == 'I_A_T_T_T_BP6_USD':
        # fix dates
        # insert missing years by copying the 1997 data and replacing everyting with nans and then merging
        # honestly kinda shitty way to do this
        data_1998 = forheatmap[forheatmap['date'] == '1997'] 
        data_1998.loc[data_1998['date'] == '1997', 'date'] = '1998'
        data_1998.loc[data_1998['value'].notnull(), 'value'] = np.nan
        data_1999 = data_1998.copy()
        data_1999.loc[data_1999['date'] == '1998', 'date'] = '1999'
        data_2000 = data_1999.copy()
        data_2000.loc[data_2000['date'] == '1999', 'date'] = '2000'
        forheatmap = pd.concat([forheatmap, data_1998, data_1999, data_2000])

        # convert format
        forheatmap['date'] = forheatmap.apply(convert_date, axis=1)
        # insert missing half-year values
        for index, row in forheatmap.iterrows():
                year = int(row['date'][:4])
                middle_of_year_date = f'{year}-06-30'

                # add rows if the date doesn't appear in the data
                if not any(forheatmap['date'] == middle_of_year_date):
                    # Create a new row with the middle-of-year date
                    new_row = row.copy()
                    new_row['date'] = middle_of_year_date
                    new_row['value'] = np.nan
                    forheatmap.loc[len(forheatmap)] = new_row
        
        forheatmap['date'] = pd.to_datetime(forheatmap['date'])
        forheatmap = forheatmap.sort_values('date').reset_index(drop=True)

    # filter by country
    forheatmap = forheatmap[forheatmap['iso3'].isin(countries)]

    # Create a pivot table to reshape the data for the heatmap
    heatmap_data = forheatmap.pivot_table(index='iso3', columns='date', values='value', aggfunc='count', fill_value=0)

    # Create a binary matrix indicating the presence of data
    binary_matrix = heatmap_data > 0

    # Reorder the rows based on the number of True values in each row (descending order)
    # binary_matrix = binary_matrix.loc[binary_matrix.sum(axis=1).sort_values(ascending=False).index]

    # Heatmap
    colours = (colors.hex2color('#CED4DA'),colors.hex2color('#339AF0'))
    cmap = LinearSegmentedColormap.from_list('Custom', colours, len(colours))
    xint = [i.year for i in binary_matrix.columns.tolist()]
    sns.set(font_scale = 1.1)
    plt.subplots(figsize=(20,15))
    ax = sns.heatmap(binary_matrix, cmap=cmap, yticklabels=True, xticklabels=xint  ,linewidths=0.01, cbar=False)
    ax.set(xlabel="", ylabel="")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 0)
    ax.set_xticks(ax.get_xticks()[::8])
    if indicator == 'I_L_T_T_T_BP6_USD' or indicator == 'I_A_T_T_T_BP6_USD':
        plt.title('Availability of CPIS data (' + indicator + ': ' + indicatorname + currency + ') aggregated by ' + version + ', select countries')
    else:
        plt.title('Availability of IIP data (' + indicator + ': ' + indicatorname + ') when denominated in ' + currency + ', select countries')
    plt.show()

    # Export graph as pdf
    fig = ax.get_figure()
    if indicator == 'I_L_T_T_T_BP6_USD' or indicator == 'I_A_T_T_T_BP6_USD':
        fig.savefig(path_figures + 'availability_' + indicator + '_' + agg + '.pdf', bbox_inches = 'tight')
    else:
        fig.savefig(path_figures + 'availability_' + indicator + '.pdf', bbox_inches = 'tight')

    return forheatmap