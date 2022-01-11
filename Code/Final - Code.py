import PyPDF2
import os
import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import country_converter as coco
from pandas_datareader import wb
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
import statsmodels.api as sm

# We have split up the code into chunks by data source: the EIU Democracy Index; the UN data;
# the EPI data from Yale; and the World Bank data. We then standardize country names for each dataframe, 
# and merge the dataframes together. The last step involves merging the combined dataframe with geospatial data.

PATH = r'C:\Users\monic\Documents\GitHub\final-project-final-project-carrie-monica-yu\Data'
PATH2 = r'C:\Users\monic\Documents\GitHub\final-project-final-project-carrie-monica-yu\Final dataframes'
PATH3 = r'C:\Users\monic\Documents\GitHub\final-project-final-project-carrie-monica-yu\Static plots'

### Economist Intelligence Unit Democracy Index: PDF parsing
FNAME = 'democracy-index-2020.pdf'
get_pdf = PyPDF2.PdfFileReader(os.path.join(PATH, FNAME))

# Functions for parsing the PDF to retrieve dataframe:
def table_onepage(n):
    page = get_pdf.getPage(n)
    text = page.extractText()
    # Using regular expressions to retrieve country names:
    country = re.findall('([A-Za-z]+[- (]*[A-Za-z]+[A-Za-z )]*)[\\n]*[0-9]+\.+[0-9]+',text)
    # Using regular expressions to retrieve the Democracy Index:
    score = re.findall('([0-9]\.[0-9][0-9])',text)
    colname = ['Country','Overall','I: Electoral process and pluralism', 
               'II: Functioning of government' , 'III: Political participation', 
               'IV: Political culture' , 'V: Civil liberties']    
    df_dem = pd.DataFrame(columns=colname)
    df_dem['Country'] = country
    # Converting the list into a dataframe:
    for i in range(len(colname)):
        if i == 0:
            df_dem[colname[i]] = country
        else:
            df_dem[colname[i]] = [score[n] for n in range(len(score)) if (n+1)%6==i%6]
    # Converting into float values:
    df_dem.iloc[:,1:] = df_dem.iloc[:,1:].applymap(float)
    # '10.00' values were parsed as '0.00', so we're correcting for it:
    for col in colname[1:]:
            df_dem.loc[(df_dem['Overall']>5)&(df_dem[col]==0), col]=10.00
    return df_dem

def table_onepage2(n):
    page = get_pdf.getPage(n)
    text = page.extractText()
    # Using regular expressions to retrieve country names:
    country = re.findall('([A-Za-z]+[- (]*[A-Za-z]+[A-Za-z )]*)[\\n]*[0-9]+\.+[0-9]+',text)
    # Using regular expressions to retrieve the Democracy Index:
    score = re.findall('([0-9]\.[0-9][0-9])',text)
    # Using regular expressions to retrieve the column names (different years):
    colname = re.findall('(20[0-9][0-9])',text)
    colname = colname[2:]
    colname.insert(0,'Country')
    df_dem = pd.DataFrame(columns=colname)
    df_dem['Country'] = country
    # Converting the list into dataframe:
    for i in range(len(colname)):
        if i == 0:
            df_dem[colname[i]] = country
        else:
            df_dem[colname[i]] = [score[n] for n in range(len(score)) if (n+1)%13==i%13]
    return df_dem

# Combining the two functions into one:
def dem_table(fromp, top, tabletype):
    df = pd.DataFrame()
    if tabletype == 'specific':
        for i in range(fromp,top+1):
            new = table_onepage(i)
            df = pd.concat([df, new])
    elif tabletype == 'year':
        for i in range(fromp,top+1):
            new = table_onepage2(i)
            df = pd.concat([df, new])
    return df

# Creating the two tables: The first table lists the 2020 Democracy Index score for each country, as well as a score
# for each of the 5 components (again, only for the year 2020). 
# The second table lists the Democracy Index score for each country from the years 2010 to 2020.
tab1 = dem_table(9,14,'specific')
tab2 = dem_table(22,26,'year')    

# Cleaning and reshaping data for the second table:
demo_index = tab2[(tab2['Country']!='average') & (tab2['Country']!='World average')]
demo_index = demo_index.drop(['2008','2006'], axis=1)
demo_index = demo_index.melt(id_vars='Country', var_name='Year', value_name='Democracy Index' )
demo_index = demo_index.replace({'UK':'United Kingdom','UAE':'United Arab Emirates'})
demo_index.loc[:, "Democracy Index"] = pd.to_numeric(demo_index["Democracy Index"])

### UN data:
FNAME2 = r'Total unemployment rate (female to male ratio).csv'
FNAME3 = r'Human Development Index (HDI).csv'

# Reading UN files: 
def load_un_file(path, fname, rows):
    un = pd.read_csv(os.path.join(path, fname),
                       na_values='..',
                       encoding='ISO-8859-1',
                       skiprows=rows,
                       skipfooter=18,
                       engine='python')
    un.dropna(how='all', axis=1, inplace=True)
    return un

# Reshaping UN data:
def reshape_un(un, value_name):
    cols = un.columns[2:]
    un_reshaped = un.melt(id_vars='Country',
                     value_vars=cols,
                     var_name='Year',
                     value_name=value_name)
    un_reshaped['Country']= un_reshaped['Country'].str.strip()
    return un_reshaped

# Creating UN dataframe:
un_unemp = load_un_file(path=PATH, fname=FNAME2, rows=6)
un_hdi = load_un_file(path=PATH, fname=FNAME3, rows=5)
un_unemp_melted = reshape_un(un=un_unemp, value_name='Unemployment rate (female to male ratio)')
un_hdi_melted = reshape_un(un=un_hdi, value_name='HDI')
un_melted = un_unemp_melted.merge(un_hdi_melted, on=['Country', 'Year'], how='left')

### EPI data:
def get_table(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table')
    return table

def get_rows():
    rows = []
    for row in table.find_all('tr'):
        td_tags = row.find_all('td')
        rows.append([val.text for val in td_tags])
    rows = [','.join(row).replace('\n', '').replace('\t', '') for row in rows[1:]]
    return rows
        
def create_file(header, content):
    content.insert(0, header)
    document = '\n'.join(content)
    return document

def write_file(path, fname, document):
    with open(os.path.join(path, fname), 'w') as f:
        f.write(document)
        
# Creating EPI dataframe:
table = get_table(url='https://epi.yale.edu/epi-results/2020/component/epi')
rows = get_rows()
document = create_file(header='Country,Rank,EPI,10-Year Change',
                       content=rows)
write_file(path=PATH,
           fname='epi.csv',
           document=document)

# Loading EPI data:
epi = pd.read_csv(os.path.join(PATH, 'epi.csv'),
                     na_values='-')

# Reshaping and cleaning EPI data:
epi_cleaned = epi[['Country', 'EPI']]
epi_cleaned.insert(1, 'Year', '2020') #We only have EPI data for the year 2020.

### World Bank data:
# A list of the indicators we need:
inds = ['NY.GDP.PCAP.PP.CD', 'SL.UEM.TOTL.ZS', 'SI.POV.GINI', 'BN.KLT.DINV.CD', 'SH.STA.MMRT',
        'SP.DYN.LE00.IN', 'SE.SCH.LIFE', 'SE.SCH.LIFE.FE', 'SE.SCH.LIFE.MA', 'SP.DYN.IMRT.IN',
        'EN.ATM.CO2E.PC', 'AG.LND.FRST.K2']

# Downloading data from the World Bank:
wb_data = pd.DataFrame(wb.download(indicator=inds,country='all', start=2010,end=2020))
wb_data = wb_data.reset_index()
# We noticed that the World Bank data included regional and global data (e.g., "European Union", "Middle East & North Africa"  
# as well as other types of aggregated data; this spanned rows from index 0 to 539. 
# We have accordingly dropped these rows from the dataframe.

# Cleaning World Bank data:
def drop_regions(df):
    df = df.drop(range(0, 539))
    df = df[df['country'] != 'Global Partnership for Education' ]
    df = df[df['country'] != 'Lending category not classified']
    return df

def fix_colnames(df):
    cols = {'NY.GDP.PCAP.PP.CD': 'GDP per capita, PPP',
            'SL.UEM.TOTL.ZS': 'Unemployment, total',
            'SI.POV.GINI': 'GINI Index',
            'BN.KLT.DINV.CD': 'Foreign Investment',
            'SH.STA.MMRT': 'Maternal Mortality ratio',
            'SP.DYN.LE00.IN': 'Life Expectancy',
            'SE.SCH.LIFE': 'Expected Years of Schooling',
            'SE.SCH.LIFE.FE': 'Expected Years of Schooling, Female',
            'SE.SCH.LIFE.MA': 'Expected Years of Schooling, Male',
            'SP.DYN.IMRT.IN': 'Mortality Rate, Infant',
            'EN.ATM.CO2E.PC': 'CO2 Emissions',
            'AG.LND.FRST.K2': 'Forest Area'}
    df = df.rename(columns=cols)
    df = df.reset_index()
    return df

df_wb = drop_regions(wb_data)
df_wb = fix_colnames(df_wb)
df_wb = df_wb.drop('index', axis=1)
df_wb = df_wb.rename(columns={'country': 'Country','year': 'Year'})
df_wb = df_wb.replace({"Korea, Dem. People's Rep.": 'North Korea', 'Korea, Rep.': 'South Korea'})

### Standardizing country names:
demo_index['Country'] = coco.convert(names=demo_index['Country'], to='name_short')
un_melted['Country'] = coco.convert(names=un_melted['Country'], to='name_short')
epi_cleaned['Country'] = coco.convert(names=epi_cleaned['Country'], to='name_short')
df_wb['Country'] = coco.convert(names=df_wb['Country'], to='name_short')
tab1['Country'] = coco.convert(names=tab1['Country'], to='name_short')

### Merging all the dataframes (except tab1) into a combined one:
combine1 = demo_index.merge(df_wb, on=['Country','Year'], how='left')
combine2 = epi_cleaned.merge(un_melted, on=['Country','Year'], how='outer')
combined = combine1.merge(combine2, on=['Country','Year'], how='left')
# The country name conversion process resulted in duplicated lines for North Korea, so we are correcting for that here:
combined = combined[~((combined['Country'] == 'North Korea') & (np.isnan(combined['Unemployment, total'])))]

### Merging the combined dataframe with regional and sub-regional identifiers:
def get_regions(url, df):
    # convert country names
    regions = pd.read_csv(url)
    regions['name'] = coco.convert(names=regions['name'], to='name_short')
    regions = regions[['name', 'region', 'sub-region']]
    merged = pd.merge(df, regions, how='left', left_on='Country', right_on='name')
    merged = merged.drop('name', axis=1)
    return merged

url = r'https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv'
combined = get_regions(url=url, df=combined)

### Writing the two dataframes out into .csv files:
combined.to_csv(os.path.join(PATH2, 'combined.csv'), index=False)
tab1.to_csv(os.path.join(PATH2, 'DI.csv'), index=False)

### Static plots:
# Plotting the top 10 most democratic and most authoritarian countries:
# Citation: https://www.kaggle.com/joshuaswords/does-hosting-the-olympics-improve-performance/notebook
def get_20_countries(df):
    temp = df[df['Year'] == '2020'].sort_values(by='Democracy Index')
    df_auth = temp[:10]
    df_demo = temp[-10:].sort_values(by='Democracy Index', ascending=False)
    temp = pd.concat([df_demo,df_auth])
    temp = temp.sort_values('Democracy Index', ascending=True)
    return temp

def plot_lines(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    my_range = range(1, len(df)+1)
    ax.set_title("\nTop 10 Most Democratic and Most Authoritarian Countries\n", fontsize=18, fontweight='bold' )
    ax.hlines(y=my_range, xmin=0, xmax=df['Democracy Index'], color='gray')
    ax.plot(df['Democracy Index'], my_range, "o", markersize=10, color='#244747')
    for s in ['top', 'bottom', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.set_xlabel('Democracy Index')    
    ax.set_yticks(my_range)
    ax.set_yticklabels(df['Country'])
    fig.savefig(PATH3 + "\Top and bottom 10 countries.png", dpi=300, bbox_inches="tight")

top_10 = get_20_countries(df=combined)
plot_lines(df=top_10)

# Plotting a heatmap of the correlation between the Democracy Index and indicators of interest:
def plot_heatmap(df):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=80)
    df_temp = df[df['Year'] == '2017'].drop(['Year','EPI'], axis=1)
    sns.heatmap(df_temp.corr(),annot=True)
    ax.set_title("\nCorrelation Between Democracy Index scores \nand Economic & Human Development Indicators\n", fontsize=18, fontweight='bold' )
    fig.savefig(PATH3 + "\Heatmap.png", dpi=300, bbox_inches="tight")

plot_heatmap(df=combined)

# Plotting a boxplot showing the distribution of Democracy Index scores for each sub-region:
# Citations: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
#            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
#            https://www.geeksforgeeks.org/how-to-add-text-to-matplotlib/ 
def plot_box(df):
    fig, ax = plt.subplots(1,1, figsize=(10, 12),dpi=100)
    df_temp = df[df['Year'] == '2020']
    sns.boxplot(data=df_temp, y='sub-region', x='Democracy Index')
    sns.swarmplot(data=df_temp, y='sub-region', x='Democracy Index', size=5, color='black')
    ax.text(1.2,-2.2, 'Distribution of Democracy Index scores for sub-regions',
            fontfamily='sans-serif', fontsize=14, fontweight='bold', color='black')
    ax.text(0,-1, 'The regions with the highest scores are Northern Europe, Australia and New Zealand\
            \nand Western European. Asia and Sub-Saharan Africa have the lowest scores.', 
            fontfamily='monospace', fontsize=12, fontweight='light', color='gray')
    ax.yaxis.label.set_visible(False)
    fig.savefig(PATH3 + "\DI score by sub-region.png", dpi=300, bbox_inches="tight")
    
plot_box(df=combined)

# Classifying each country by type of political regime based on their Democracy Index score (following the EIU's classification system):
# Citation: https://thispointer.com/python-how-to-use-if-else-elif-in-lambda-functions/
combined['Political Regime'] = combined['Democracy Index'].apply(lambda x: 'Full democracy' if x > 8 
                                                                                            else ('Flawed democracy' if (x > 6 and x < 8) 
                                                                                                                     else('Hybrid regime' if (x < 6 and x > 4) 
                                                                                                                                          else 'Authoritarian regime')))

# Plotting a swarmplot of GDP per capita (PPP) for each type of political regime:
# Citation: https://madhuramiah.medium.com/some-interesting-visualizations-with-seaborn-python-ad207f50b844
def plot_swarm(df):
    df_temp = df[df['Year'] == '2020']
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    # Removing border from plot
    for s in ['top', 'bottom', 'right']:
        ax.spines[s].set_visible(False)
    ax.axes.get_xaxis().set_ticks([])
    # Plotting the swarmplot
    sns.swarmplot(x='Political Regime',  y='GDP per capita, PPP', data=df_temp)
    average_gdp = df_temp['GDP per capita, PPP'].mean()
    ax.axhline(average_gdp, ls='--', c='black')
    # Formatting:
    plt.title('Correlation between political regimes and GDP per capita (PPP) (2020)',fontsize=14,fontweight='bold') 
    ax.text(0.15, 110000, 'Full democracies tend to have high GDPs, \
            \nwhile flawed democracies exhibit a fairly dispersed distribution. \
            \nMost hybrid and authoritarian regimes have a lower GDP than the mean.',\
            fontfamily='monospace',fontsize=14,fontweight='light',color='gray')
    fig.savefig(PATH3 + "\GDP by type of political regime.png", dpi=300)

plot_swarm(df=combined)

# Gender inequalities in expected years of schooling for each type of political regime:
# Citation: https://stackoverflow.com/questions/46794373/make-a-bar-graph-of-2-variables-based-on-a-dataframe
def get_edu_data(df, year):
    df_temp = df.loc[df['Year'] == year]
    df_temp = df_temp[['Expected Years of Schooling, Female', 'Expected Years of Schooling, Male', 'Political Regime']]
    data = df_temp.groupby('Political Regime').agg('mean')
    data = data.reset_index()
    data = data.sort_values('Expected Years of Schooling, Female', ascending=False)
    return data

def plot_bar(df):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    df.set_index('Political Regime').plot.bar(rot=0, ax=ax)
    ax.set_title('Correlation between political regimes and expected years of schooling (2018)',fontsize=14,fontweight='bold') 
    fig.savefig(PATH3 + "\Gender inequality in access to education by political regime.png", dpi=300)

edu_data = get_edu_data(df=combined, year='2018')
plot_bar(df=edu_data)

# Comparing how full democracies and authoritarian regimes score on each indicator:
# Citations: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8
def get_regime_data(df, year, regime):
    temp = df.loc[df['Year'] == year]
    data = temp.loc[temp['Political Regime'] == regime]
    return data

def plot_kde_hdi(df_demo, df_auth):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Full Democracies vs Authoritarian Regimes: 6 Human Development Indicators',
                 fontsize=20, fontweight='bold')
    sns.kdeplot(ax=axes[0, 0], data=df_demo, shade=True, color='lightgreen', x='Life Expectancy')
    sns.kdeplot(ax=axes[0, 0], data=df_auth, shade=True, color='lightcoral', x='Life Expectancy')
    sns.kdeplot(ax=axes[0, 1], data=df_demo, shade=True, color='lightgreen', x='Expected Years of Schooling')
    sns.kdeplot(ax=axes[0, 1], data=df_auth, shade=True, color='lightcoral', x='Expected Years of Schooling')
    sns.kdeplot(ax=axes[0, 2], data=df_demo, shade=True, color='lightgreen', x='Mortality Rate, Infant', label='Full Democracy')
    sns.kdeplot(ax=axes[0, 2], data=df_auth, shade=True, color='lightcoral', x='Mortality Rate, Infant', label='Authoritarian Regime')
    sns.kdeplot(ax=axes[1, 0], data=df_demo, shade=True, color='lightgreen', x='CO2 Emissions')
    sns.kdeplot(ax=axes[1, 0], data=df_auth, shade=True, color='lightcoral', x='CO2 Emissions')
    sns.kdeplot(ax=axes[1, 1], data=df_demo, shade=True, color='lightgreen', x='Forest Area')
    sns.kdeplot(ax=axes[1, 1], data=df_auth, shade=True, color='lightcoral', x='Forest Area')
    sns.kdeplot(ax=axes[1, 2], data=df_demo, shade=True, color='lightgreen', x='HDI')
    sns.kdeplot(ax=axes[1, 2], data=df_auth, shade=True, color='lightcoral', x='HDI')
    axes[0, 2].legend()
    fig.savefig(PATH3 + "\KDE for human development indicators.png", dpi=300)

def plot_kde_eco(df_demo, df_auth): 
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Full Democracies vs Authoritarian Regimes: 4 Economic Indicators',
                 fontsize=20, fontweight='bold')
    sns.kdeplot(ax=axes[0, 0], data=df_demo, shade=True, color='lightgreen', x='GDP per capita, PPP')
    sns.kdeplot(ax=axes[0, 0], data=df_auth, shade=True, color='lightcoral', x='GDP per capita, PPP')
    sns.kdeplot(ax=axes[0, 1], data=df_demo, shade=True, color='lightgreen', x='Unemployment, total', label='Full Democracy')
    sns.kdeplot(ax=axes[0, 1], data=df_auth, shade=True, color='lightcoral', x='Unemployment, total', label='Authoritarian Regime')
    sns.kdeplot(ax=axes[1, 0], data=df_demo, shade=True, color='lightgreen', x='GINI Index')
    sns.kdeplot(ax=axes[1, 0], data=df_auth, shade=True, color='lightcoral', x='GINI Index')
    sns.kdeplot(ax=axes[1, 1], data=df_demo, shade=True, color='lightgreen', x='Foreign Investment')
    sns.kdeplot(ax=axes[1, 1], data=df_auth, shade=True, color='lightcoral', x='Foreign Investment')
    axes[0, 1].legend()
    fig.savefig(PATH3 + "\KDE for economic indicators.png", dpi=300)

combined_demo = get_regime_data(df=combined, year='2018', regime='Full democracy')
combined_auth = get_regime_data(df=combined, year='2018', regime='Authoritarian regime')
plot_kde_hdi(df_demo=combined_demo, df_auth=combined_auth)
plot_kde_eco(df_demo=combined_demo, df_auth=combined_auth)

### Running an OLS regression of indicators of interest on Democracy Index score:
cols = list(combined.columns)[3:18]
# Linear regression results:
lr_result = {'Indicator':[], 'Coefficient':[], 'p-value':[]}
for variable in cols:
    tab = combined[['Democracy Index', variable]]
    tab = tab.dropna()
    tab = tab.replace(0,0.00001) # This lets us log the values. 
    if variable != 'Foreign Investment': # Foreign investments is the only indicator that contains negative values.
        tab[variable] = tab[variable].map(log)
    X = tab['Democracy Index']
    X = sm.add_constant(X)
    y = tab[variable]
    lr = sm.OLS(y, X)
    result = lr.fit()
    lr_result['Indicator'].append(variable)
    lr_result['Coefficient'].append(round(list(result.params)[1], 4))
    lr_result['p-value'].append(list(result.pvalues)[1])
lr_result_df = pd.DataFrame(lr_result)
# These are the results:
print(lr_result_df)

### Links to data sources:
# EIU Democracy Index: https://www.eiu.com/n/campaigns/democracy-index-2020/
# UN data: http://hdr.undp.org/en/indicators/137506; http://hdr.undp.org/en/indicators/169706
# EPI data: https://epi.yale.edu/epi-results/2020/component/epi