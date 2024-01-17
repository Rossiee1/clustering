import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

def plot_correlation(plot_data):
    # Plot data for population correlation between
    # France, Germany Italy and UK 


    # Create lists for the plot
    countries = list(plot_data.keys())
    correlations = list(plot_data.values())

    # Define colors for each country
    colors = ['#022356', '#EF6F33', '#6A1588', '#2E1337']

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.bar(countries, correlations, color=colors)
    ax.set_xlabel('Country')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Correlation between Population and GDP', fontweight='bold', color='darkblue')
    ax.set_ylim(-0.5, 1)

    # Add the correlation values on top of the bars
    for i, v in enumerate(correlations):
        ax.text(i, v + 0.05 if v > 0 else v - 0.05, str(round(v, 4)),
                ha='center', va='bottom' if v > 0 else 'top')

    # Add legend with colors
    legend_labels = [f'{country} ({correlation:.4f})' for country, correlation in plot_data.items()]
    ax.legend(bars, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Show the plot
    return plt.show()

def fit(x, a, b):
    return a * x + b

# Define the error ranges function
def ranges(x, popt, pcov):
    perr = np.sqrt(np.diag(pcov))
    y = fit(x, *popt)
    y_upper = fit(x, *(popt + perr))
    y_lower = fit(x, *(popt - perr))
    return y, y_lower, y_upper


def plot_country_confidence(df, country_name, series_code, title):
    # Filter the data for India and the series code 'NY.GNP.MKTP.CD'
    selected_country = df[(df['Country Name'] == country_name) & (df['Series Code'] == series_code)]

    # Drop the non-year columns to focus on the yearly data
    sel_columns = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
    selected_country_years = selected_country.drop(columns=sel_columns)


    # Since there was an error with 'Unnamed: 0', we will exclude it from the drop
    selected_country_years = selected_country.drop(columns=sel_columns)

    # Transpose the data to have years as rows
    selected_country_transposed = selected_country_years.transpose()
    selected_country_transposed.columns = ['GNP']

    # Reset index to have years as a column
    selected_country_transposed.reset_index(inplace=True)
    selected_country_transposed.rename(columns={'index': 'Year'}, inplace=True)

    # Extract year from the column name and convert to integer
    selected_country_transposed['Year'] = selected_country_transposed['Year'].str.extract('(\d+)').astype(int)

    # Convert GNP column to numeric, coerce errors to NaN
    selected_country_transposed['GNP'] = pd.to_numeric(selected_country_transposed['GNP'], errors='coerce')

    # Drop rows with NaN values
    selected_country_cleaned = selected_country_transposed.dropna()

    # Fit the linear model to the data
    popt, pcov = curve_fit(fit, selected_country_cleaned['Year'], selected_country_cleaned['GNP'])

    # Generate the confidence range
    years = np.array(selected_country_cleaned['Year'])
    GNP, GNP_lower, GNP_upper = ranges(years, popt, pcov)

    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.plot(selected_country_cleaned['Year'], selected_country_cleaned['GNP'], 'o', label='Data')
    plt.plot(years, GNP, '-', label='Fit')
    plt.fill_between(years, GNP_lower, GNP_upper, color='gray', alpha=0.2, label='Confidence range')

    # Label the plot
    plt.xlabel('Year')
    plt.ylabel('GNP (current US$)')
    plt.title('GNP of United Kingdom with Confidence Range')
    plt.legend()

    # Show the plot
    return plt.show()


def cluster_plot(df, series_name_y, series_name_x, n_clusters):
    first_series = df[df['Series Code'] == series_name_y]
    second_series = df[df['Series Code'] == series_name_x]

    # Melt the dataframes
    first_series_melted = first_series.melt(id_vars=['Country Name', 'Country Code'], 
                                          value_vars=first_series.columns[4:], 
                                          var_name='Year', value_name='CO2 emissions (kt)')
    second_series_melted = second_series.melt(id_vars=['Country Name', 'Country Code'], 
                                                  value_vars=second_series.columns[4:], 
                                                  var_name='Year', value_name='Population, total')

    # Merge the two datasets on 'Country Code' and 'Year'
    joined = pd.merge(first_series_melted, second_series_melted, on=['Country Code', 'Year'])

    # Convert 'Year' to datetime and extract the year for plotting
    joined['Year'] = pd.to_datetime(joined['Year'].str.extract('(\d{4})')[0]).dt.year

    # Convert to numeric and drop NaNs
    joined['CO2 emissions (kt)'] = pd.to_numeric(joined['CO2 emissions (kt)'], errors='coerce')
    joined['Population, total'] = pd.to_numeric(joined['Population, total'], errors='coerce')
    joined.dropna(subset=['CO2 emissions (kt)', 'Population, total'], inplace=True)

    # Prepare the data for clustering
    X = joined[['CO2 emissions (kt)', 'Population, total']]

    # Use KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # Assign the clusters to the dataframe
    joined['Cluster'] = kmeans.labels_

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=joined, x='Population, total', y='CO2 emissions (kt)', hue='Cluster')
    plt.title('Clustering of CO2 emissions (kt) and Population, total')
    plt.xlabel('Population, total')
    plt.ylabel('CO2 emissions (kt)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Cluster')
    return plt.show()


def country_corr(df, country_name):
    # df = pd.read_csv('wb_data.csv', encoding='ascii')

    # Filter the dataframe for Country
    country_df = df[df['Country Name'] == country_name]

    # Replace '..' with NaN to ensure proper numeric conversion
    country_df = country_df.replace('..', pd.NA)

    # Convert all columns to numeric, except 'Series Code', coercing errors to NaN
    numeric_columns = country_df.columns.drop(['Series Name', 'Series Code', 'Country Name', 'Country Code'])
    country_df[numeric_columns] = country_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop non-numeric columns except 'Series Code' and columns with all NaN values
    country_df = country_df.drop(columns=['Series Name', 'Country Code'])
    country_df = country_df.dropna(axis=1, how='all')

    # Now we will create a correlation matrix for each unique 'Series Code'
    # First, we need to pivot the dataframe so that each 'Series Code' becomes a row
    china_pivoted = country_df.pivot(index='Series Code', columns='Country Name', values=numeric_columns)
    correlation_matrix_pivoted = china_pivoted.T.corr()

    # Plot the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix_pivoted, annot=True, fmt='.2f')
    plt.title('Heatmap Correlation Matrix for Germany')
    return plt.show()

def plot_country_labour(df):
    # Filter the dataset for the required countries and indicator
    filtered_data = df[(df['Country Name'].isin(['United Kingdom', 'France', 'Italy', 'Germany'])) & (df['Series Code'] == 'NY.GDP.MKTP.KN')]

    # Since the years are in separate columns, we need to melt the dataframe to have a single year column
    melted_data = pd.melt(filtered_data, id_vars=['Country Name'], value_vars=[str(year) + ' [YR' + str(year) + ']' for year in range(2000, 2021)],
                           var_name='Year', value_name='Total Labor Force')

    # Convert the 'Year' column to just the year number
    melted_data['Year'] = melted_data['Year'].str.extract('(\\d{4})').astype(int)

    # Pivot the data to have countries as columns and years as rows for plotting
    pivoted_data = melted_data.pivot(index='Year', columns='Country Name', values='Total Labor Force')

    # Convert non-numeric data to numeric, coerce errors to 
    pivoted_data = pivoted_data.apply(pd.to_numeric, errors='coerce')

    # Plot the grouped bar chart
    pivoted_data.plot(kind='bar', figsize=(15, 7))
    plt.title('Total Labor Force by Country')
    plt.xlabel('Year')
    plt.ylabel('Total Labor Force')
    plt.legend(title='Country')
    return plt.show()


if __name__ == "__main__":


    df = pd.read_csv('./wb_data.csv')


    # Reshape the data to have one row per country and year with population and GDP values
    df = pd.read_csv('./wb_data.csv')

    # Filter the dataset for population and GDP series
    population_data = df[df['Series Name'] == 'Population, total']
    gdp_data = df[df['Series Name'] == 'GDP (constant LCU)']

    # Drop unnecessary columns
    population_data = population_data.drop(columns=['Series Name', 'Series Code'])
    gdp_data = gdp_data.drop(columns=['Series Name', 'Series Code'])

    # Rename the yearly columns to a common format for merging
    columns_rename = {year: year[-9:] for year in population_data.columns if 'YR' in year}
    population_data.rename(columns=columns_rename, inplace=True)
    gdp_data.rename(columns=columns_rename, inplace=True)

    # Melt the dataframes to have one row per country-year
    population_melted = population_data.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='Population_Value')
    gdp_melted = gdp_data.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP_Value')

    # Merge the population and GDP data on country and year
    merged_data = pd.merge(population_melted, gdp_melted, on=['Country Name', 'Country Code', 'Year'])

    # Convert the 'Population_Value' and 'GDP_Value' columns to numeric, coercing non-numeric values to NaN
    merged_data['Population_Value'] = pd.to_numeric(merged_data['Population_Value'], errors='coerce')
    merged_data['GDP_Value'] = pd.to_numeric(merged_data['GDP_Value'], errors='coerce')

    # Drop rows with NaN values in 'Population_Value' or 'GDP_Value'
    merged_data_clean = merged_data.dropna(subset=['Population_Value', 'GDP_Value'])

    # Calculate the correlation between population and GDP for each country with cleaned data
    correlations = {}
    for country in merged_data_clean['Country Name'].unique():
        country_data = merged_data_clean[merged_data_clean['Country Name'] == country]
        correlation = country_data[['Population_Value', 'GDP_Value']].corr().iloc[0, 1]
        correlations[country] = correlation

    # Convert the correlations dictionary to a DataFrame for plotting
    correlation_df = pd.DataFrame(list(correlations.items()), columns=['Country', 'Correlation'])

    # Sort the DataFrame based on the absolute value of the correlation to get meaningful order
    sorted_correlation_df = correlation_df.reindex(correlation_df.Correlation.abs().sort_values(ascending=False).index)

    # Display the sorted correlation DataFrame
    print(sorted_correlation_df.head())

    data = {
        'France': 0.9577,
        'Germany': 0.0629,
        'Italy': -0.2849,
        'United Kingdom': 0.9266
    }
    plot_correlation(data)

    country_name = 'United Kingdom'
    series_code = 'SP.POP.TOTL'
    title = ''
    plot_country_confidence(df, country_name, series_code, title)

    cluster_plot(df, 'EN.ATM.CO2E.KT', 'SP.POP.TOTL', 3)

    country_corr(df, 'Germany')

    plot_country_labour(df)

