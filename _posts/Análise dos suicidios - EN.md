
# Suicide analysis


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

#### Reading the data


```python
df = pd.read_csv('master.csv')
```

Let's take a look at the data:


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>sex</th>
      <th>age</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>suicides/100k pop</th>
      <th>country-year</th>
      <th>HDI for year</th>
      <th>gdp_for_year ($)</th>
      <th>gdp_per_capita ($)</th>
      <th>generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12171</th>
      <td>Ireland</td>
      <td>1994</td>
      <td>female</td>
      <td>15-24 years</td>
      <td>13</td>
      <td>303000</td>
      <td>4.29</td>
      <td>Ireland1994</td>
      <td>NaN</td>
      <td>57,166,037,102</td>
      <td>17188</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <th>11049</th>
      <td>Guatemala</td>
      <td>2014</td>
      <td>female</td>
      <td>5-14 years</td>
      <td>13</td>
      <td>1906536</td>
      <td>0.68</td>
      <td>Guatemala2014</td>
      <td>0.627</td>
      <td>58,722,323,918</td>
      <td>4210</td>
      <td>Generation Z</td>
    </tr>
    <tr>
      <th>23290</th>
      <td>South Africa</td>
      <td>1996</td>
      <td>male</td>
      <td>35-54 years</td>
      <td>36</td>
      <td>4031071</td>
      <td>0.89</td>
      <td>South Africa1996</td>
      <td>NaN</td>
      <td>147,607,982,695</td>
      <td>3908</td>
      <td>Boomers</td>
    </tr>
    <tr>
      <th>7902</th>
      <td>Ecuador</td>
      <td>2002</td>
      <td>male</td>
      <td>55-74 years</td>
      <td>39</td>
      <td>525419</td>
      <td>7.42</td>
      <td>Ecuador2002</td>
      <td>NaN</td>
      <td>28,548,945,000</td>
      <td>2472</td>
      <td>Silent</td>
    </tr>
    <tr>
      <th>5936</th>
      <td>Colombia</td>
      <td>2010</td>
      <td>male</td>
      <td>75+ years</td>
      <td>65</td>
      <td>418296</td>
      <td>15.54</td>
      <td>Colombia2010</td>
      <td>0.706</td>
      <td>287,018,184,638</td>
      <td>6836</td>
      <td>Silent</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>suicides/100k pop</th>
      <th>HDI for year</th>
      <th>gdp_per_capita ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27820.000000</td>
      <td>27820.000000</td>
      <td>2.782000e+04</td>
      <td>27820.000000</td>
      <td>8364.000000</td>
      <td>27820.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2001.258375</td>
      <td>242.574407</td>
      <td>1.844794e+06</td>
      <td>12.816097</td>
      <td>0.776601</td>
      <td>16866.464414</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.469055</td>
      <td>902.047917</td>
      <td>3.911779e+06</td>
      <td>18.961511</td>
      <td>0.093367</td>
      <td>18887.576472</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1985.000000</td>
      <td>0.000000</td>
      <td>2.780000e+02</td>
      <td>0.000000</td>
      <td>0.483000</td>
      <td>251.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1995.000000</td>
      <td>3.000000</td>
      <td>9.749850e+04</td>
      <td>0.920000</td>
      <td>0.713000</td>
      <td>3447.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2002.000000</td>
      <td>25.000000</td>
      <td>4.301500e+05</td>
      <td>5.990000</td>
      <td>0.779000</td>
      <td>9372.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2008.000000</td>
      <td>131.000000</td>
      <td>1.486143e+06</td>
      <td>16.620000</td>
      <td>0.855000</td>
      <td>24874.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>22338.000000</td>
      <td>4.380521e+07</td>
      <td>224.970000</td>
      <td>0.944000</td>
      <td>126352.000000</td>
    </tr>
  </tbody>
</table>
</div>



The dataset has data from suicides from 1985 to 2016.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27820 entries, 0 to 27819
    Data columns (total 12 columns):
    country               27820 non-null object
    year                  27820 non-null int64
    sex                   27820 non-null object
    age                   27820 non-null object
    suicides_no           27820 non-null int64
    population            27820 non-null int64
    suicides/100k pop     27820 non-null float64
    country-year          27820 non-null object
    HDI for year          8364 non-null float64
     gdp_for_year ($)     27820 non-null object
    gdp_per_capita ($)    27820 non-null int64
    generation            27820 non-null object
    dtypes: float64(2), int64(4), object(6)
    memory usage: 2.5+ MB


is there null data?


```python
df.isnull().sum()
```




    country                   0
    year                      0
    sex                       0
    age                       0
    suicides_no               0
    population                0
    suicides/100k pop         0
    country-year              0
    HDI for year          19456
     gdp_for_year ($)         0
    gdp_per_capita ($)        0
    generation                0
    dtype: int64



## Understanding the data

The country-year field displays the country name and year of the record. In this way, it is a redundant field and will be discarded. Also due to most data from the 'HDI for year' field, it will be discarded.


```python
df.drop(['country-year', 'HDI for year'], inplace=True, axis = 1)
```

Let's rename some columns simply to make it easier to access them.


```python
df = df.rename(columns={'gdp_per_capita ($)': 'gdp_per_capita', ' gdp_for_year ($) ':'gdp_for_year'})
```

In this case, the 'gdp_for_year' field is as a string, so let's convert this to a number.


```python
for i, x in enumerate(df['gdp_for_year']):
    df['gdp_for_year'][i] = x.replace(',', '')
    
df['gdp_for_year'] = df['gdp_for_year'].astype('int64')
```

    /home/luis/anaconda3/envs/dl/lib/python3.6/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      


## Data Description

Each data in the data set represents a year, a country, a certain age range, and a gender. For example, in the country Brazil in the year 1985, over 75 years, committed suicide 129 men.

The data set has 10 attributes. These being:

- Country: country of record data;
- Year: year of record data;
- Sex: Sex (male or female);
- Age: Suicide age range, ages divided into six categories;
- Suicides_no: number of suicides;
- Population: population of this sex, in this age range, in this country and in this year;
- Suicides / 100k pop: Reason between the number of suicides and the population / 100k;
- GDP_for_year: GDP of the country in the year who issue;
- GDP_per_capita: ratio between the country's GDP and its population;
- Generation: Generation of the suicides in question, being possible 6 different categories.

Possible age categories and generations are:


```python
df['age'].unique()
```




    array(['15-24 years', '35-54 years', '75+ years', '25-34 years',
           '55-74 years', '5-14 years'], dtype=object)




```python
df['generation'].unique()
```




    array(['Generation X', 'Silent', 'G.I. Generation', 'Boomers',
           'Millenials', 'Generation Z'], dtype=object)



## Adding some things

As the HDI was discarded and it is very interesting to assess whether the development of the country has an influence on the suicide rate, I have separated a list of first and second world countries from the data of the site:

http://worldpopulationreview.com

Then I categorized each country in the data set into first, second and third world.


```python
Frist_world = ['United States', 'Germany', 'Japan', 'Turkey', 'United Kingdom', 'France', 'Italy', 'South Korea',
              'Spain', 'Canada', 'Australia', 'Netherlands', 'Belgium', 'Greece', 'Portugal', 
              'Sweden', 'Austria', 'Switzerland', 'Israel', 'Singapore', 'Denmark', 'Finland', 'Norway', 'Ireland',
              'New Zeland', 'Slovenia', 'Estonia', 'Cyprus', 'Luxembourg', 'Iceland']

Second_world = ['Russian Federation', 'Ukraine', 'Poland', 'Uzbekistan', 'Romania', 'Kazakhstan', 'Azerbaijan', 'Czech Republic',
               'Hungary', 'Belarus', 'Tajikistan', 'Serbia', 'Bulgaria', 'Slovakia', 'Croatia', 'Maldova', 'Georgia',
               'Bosnia And Herzegovina', 'Albania', 'Armenia', 'Lithuania', 'Latvia', 'Brazil', 'Chile', 'Argentina',
               'China', 'India', 'Bolivia', 'Romenia']
```


```python
country_world = []
for i in range(len(df)):
    
    if df['country'][i] in Frist_world:
        country_world.append('1')
    elif df['country'][i] in Second_world:
        country_world.append('2')
    else:
        country_world.append('3')

df['country_world'] = country_world
```

# Exploratory analysis

I will analyze the impact of some attributes on the amount of suicides. We start this year.

#### Year


```python
suicides_no_year = []

for y in df['year'].unique():
    suicides_no_year.append(sum(df[df['year'] == y]['suicides_no']))

n_suicides_year = pd.DataFrame(suicides_no_year, columns=['suicides_no_year'])
n_suicides_year['year'] = df['year'].unique()

top_year = n_suicides_year.sort_values('suicides_no_year', ascending=False)['year']
top_suicides = n_suicides_year.sort_values('suicides_no_year', ascending=False)['suicides_no_year']

plt.figure(figsize=(8,5))
plt.xticks(rotation=90)
sns.barplot(x = top_year, y = top_suicides)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff84c16d1d0>




![png](output_25_1.png)


#### Age


```python
suicides_no_age = []

for a in df['age'].unique():
    suicides_no_age.append(sum(df[df['age'] == a]['suicides_no']))

plt.xticks(rotation=30)
sns.barplot(x = df['age'].unique(), y = suicides_no_age)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff84c18ce48>




![png](output_27_1.png)


#### Sex


```python
suicides_no_sex = []

for s in df['sex'].unique():
    suicides_no_sex.append(sum(df[df['sex'] == s]['suicides_no']))

sns.barplot(x = df['sex'].unique(), y = suicides_no_sex)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff84a0f1b00>




![png](output_29_1.png)



```python
sns.catplot(x='sex', y='suicides_no',col='age', data=df, estimator=np.median,height=4, aspect=.7,kind='bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7ff87d66b2b0>




![png](output_30_1.png)


#### Country

Countries with larger populations should have more suicides.


```python
suicides_no_pais = []
for c in df['country'].unique():
    suicides_no_pais.append(sum(df[df['country'] == c]['suicides_no']))
    
n_suicides_pais = pd.DataFrame(suicides_no_pais, columns=['suicides_no_pais'])
n_suicides_pais['country'] = df['country'].unique()

quant = 15
top_paises = n_suicides_pais.sort_values('suicides_no_pais', ascending=False)['country'][:quant]
top_suicides = n_suicides_pais.sort_values('suicides_no_pais', ascending=False)['suicides_no_pais'][:quant]
sns.barplot(x = top_suicides, y = top_paises)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff849d02630>




![png](output_32_1.png)


By using the amount of suicides per 100k inhabitants, we remove the bias of overpopulated countries.


```python
suicides_no_pais = []
for c in df['country'].unique():
    suicides_no_pais.append(sum(df[df['country'] == c]['suicides/100k pop']))
    
n_suicides_pais = pd.DataFrame(suicides_no_pais, columns=['suicides_no_pais/100k'])
n_suicides_pais['country'] = df['country'].unique()

quant = 15
top_paises = n_suicides_pais.sort_values('suicides_no_pais/100k', ascending=False)['country'][:quant]
top_suicides = n_suicides_pais.sort_values('suicides_no_pais/100k', ascending=False)['suicides_no_pais/100k'][:quant]
sns.barplot(x = top_suicides, y = top_paises)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff849aaf278>




![png](output_34_1.png)


#### Generation


```python
suicides_no_gen = []
for g in df['generation'].unique():
    suicides_no_gen.append(sum(df[df['generation'] == g]['suicides_no']))

plt.figure(figsize=(8,5))
sns.barplot(x = df['generation'].unique(), y = suicides_no_gen)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff849a49898>




![png](output_36_1.png)


#### Country world


```python
suicides_no_world = []
for w in df['country_world'].unique():
    suicides_no_world.append(sum(df[df['country_world'] == w]['suicides_no']))
    
sns.barplot(x = df['country_world'].unique(), y = suicides_no_world)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff83c5b96d8>




![png](output_38_1.png)



```python
suicides_no_world = []
for w in df['country_world'].unique():
    suicides_no_world.append(sum(df[df['country_world'] == w]['suicides/100k pop']))
    
sns.barplot(x = df['country_world'].unique(), y = suicides_no_world)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff8468ef978>




![png](output_39_1.png)


#### GDP for year


```python
sns.scatterplot(x = 'gdp_for_year', y = 'suicides_no', data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff8498e2cc0>




![png](output_41_1.png)


#### GDP por capita


```python
sns.scatterplot(x = 'gdp_per_capita', y = 'suicides_no', data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff8498d43c8>




![png](output_43_1.png)


### Attribute Correlation


```python
plt.figure(figsize=(8,7))
sns.heatmap(df.corr(), cmap = 'coolwarm', annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff84982d7b8>




![png](output_45_1.png)


# Checking the suicidade/100k distribution of some countries


```python
countries = ['Russian Federation', 'Brazil', 'Poland', 'Italy', 'United States', 'Germany', 'Japan', 'Spain', 'France']
df_filtred = df[[df['country'][i] in countries for i in range(len(df))]]

plt.figure(figsize=(12,6))
sns.boxplot(x = 'suicides/100k pop', y = 'country', data = df_filtred)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff833580780>




![png](output_47_1.png)


### General Plot of the World


```python
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
```


```python
init_notebook_mode(connected=True) 
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>



```python
cod = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
```


```python
codes = []
for i in range(len(n_suicides_pais)):
    c = n_suicides_pais['country'][i]
    f = 0
    for j in range(len(cod)):
        if c == cod['COUNTRY'][j]:
            tmp = cod['CODE'][j]
            f = 1
            break
    if f == 0:
        if c == 'Bahamas':
            tmp  = 'BHM'
        elif c == 'Republic of Korea':
            tmp = 'KOR'
        elif c == 'Russian Federation':
            tmp = 'RUS'
        else:
            tmp = 'VC'
    codes.append(tmp)
```


```python
data = dict(
        type = 'choropleth',
        locations = codes,
        z = n_suicides_pais['suicides_no_pais/100k'],
        text = n_suicides_pais['country'],
        colorbar = {'title' : 'número de suicídios'},
      )
```


```python
layout = dict(
    title = 'Mapa de calor de suicídios 1985-2016',
    geo = dict(
        showframe = False,
        projection = {'type':'equirectangular'}
    )
)
```


```python
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
```


<div id="0190bb3c-9166-471e-9b5b-450853a0d06a" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
if (document.getElementById("0190bb3c-9166-471e-9b5b-450853a0d06a")) {
    Plotly.newPlot("0190bb3c-9166-471e-9b5b-450853a0d06a", [{"colorbar": {"title": {"text": "n\u00famero de suic\u00eddios"}}, "locations": ["ALB", "ATG", "ARG", "ARM", "ABW", "AUS", "AUT", "AZE", "BHM", "BHR", "BRB", "BLR", "BEL", "BLZ", "BIH", "BRA", "BGR", "CPV", "CAN", "CHL", "COL", "CRI", "HRV", "CUB", "CYP", "CZE", "DNK", "DMA", "ECU", "SLV", "EST", "FJI", "FIN", "FRA", "GEO", "DEU", "GRC", "GRD", "GTM", "GUY", "HUN", "ISL", "IRL", "ISR", "ITA", "JAM", "JPN", "KAZ", "KIR", "KWT", "KGZ", "LVA", "LTU", "LUX", "MAC", "MDV", "MLT", "MUS", "MEX", "MNG", "MNE", "NLD", "NZL", "NIC", "NOR", "OMN", "PAN", "PRY", "PHL", "POL", "PRT", "PRI", "QAT", "KOR", "ROU", "RUS", "KNA", "LCA", "VC", "SMR", "SRB", "SYC", "SGP", "SVK", "SVN", "ZAF", "ESP", "LKA", "SUR", "SWE", "CHE", "THA", "TTO", "TUR", "TKM", "UKR", "ARE", "GBR", "USA", "URY", "UZB"], "text": ["Albania", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Barbados", "Belarus", "Belgium", "Belize", "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Cabo Verde", "Canada", "Chile", "Colombia", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Dominica", "Ecuador", "El Salvador", "Estonia", "Fiji", "Finland", "France", "Georgia", "Germany", "Greece", "Grenada", "Guatemala", "Guyana", "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Kazakhstan", "Kiribati", "Kuwait", "Kyrgyzstan", "Latvia", "Lithuania", "Luxembourg", "Macau", "Maldives", "Malta", "Mauritius", "Mexico", "Mongolia", "Montenegro", "Netherlands", "New Zealand", "Nicaragua", "Norway", "Oman", "Panama", "Paraguay", "Philippines", "Poland", "Portugal", "Puerto Rico", "Qatar", "Republic of Korea", "Romania", "Russian Federation", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and Grenadines", "San Marino", "Serbia", "Seychelles", "Singapore", "Slovakia", "Slovenia", "South Africa", "Spain", "Sri Lanka", "Suriname", "Sweden", "Switzerland", "Thailand", "Trinidad and Tobago", "Turkey", "Turkmenistan", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan"], "z": [924.7599999999999, 179.14, 3894.589999999997, 976.2100000000002, 1596.5199999999998, 4677.410000000004, 9076.229999999998, 356.2400000000002, 344.28000000000003, 467.24000000000007, 891.1300000000002, 7831.129999999998, 7900.500000000004, 2093.490000000001, 110.81, 2174.7200000000003, 7016.080000000007, 133.84, 4338.7199999999975, 3921.6400000000012, 2009.39, 2553.719999999998, 5982.84, 6111.9500000000035, 586.2599999999999, 5952.9899999999925, 3721.6499999999965, 0.0, 2345.209999999999, 3035.940000000002, 6873.780000000001, 673.8600000000001, 7924.11, 7803.250000000002, 1116.379999999999, 4854.689999999999, 1512.1199999999997, 660.9999999999999, 1146.6999999999996, 6655.919999999999, 10156.069999999994, 4889.74, 3881.7299999999987, 3329.969999999999, 3168.87, 106.44, 8025.229999999999, 9519.519999999995, 878.5100000000002, 355.9300000000001, 4457.300000000004, 7373.350000000004, 10588.879999999997, 6156.559999999995, 171.74000000000004, 164.08, 1872.0099999999995, 4464.579999999998, 1751.1899999999998, 184.39, 1194.0299999999997, 4066.5199999999973, 5008.320000000005, 472.82000000000005, 4658.760000000001, 26.5, 1744.059999999999, 1366.3700000000006, 435.83000000000004, 4397.620000000002, 3673.36, 3789.279999999998, 318.1499999999998, 9350.449999999995, 4171.550000000001, 11305.130000000006, 0.0, 2420.1200000000013, 1726.6299999999999, 145.62, 4787.450000000001, 1615.7000000000003, 6340.979999999998, 3318.1599999999994, 7012.619999999997, 231.48999999999998, 3509.0600000000013, 4658.959999999998, 7162.319999999997, 5247.719999999996, 4794.070000000002, 2362.67, 4479.509999999996, 199.17000000000004, 2994.730000000001, 8931.659999999993, 94.88999999999997, 2790.919999999999, 5140.970000000001, 6538.96, 2138.1700000000005], "type": "choropleth", "uid": "a40fc06b-c401-44b1-a316-dfc88cf7fc24"}], {"geo": {"projection": {"type": "equirectangular"}, "showframe": false}, "title": {"text": "Mapa de calor de suic\u00eddios 1985-2016"}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"}); 
}
});</script><script type="text/javascript">window.addEventListener("resize", function(){if (document.getElementById("0190bb3c-9166-471e-9b5b-450853a0d06a")) {window._Plotly.Plots.resize(document.getElementById("0190bb3c-9166-471e-9b5b-450853a0d06a"));};})</script>


## Brazil Facts

As a Brazilian, I have a particular interest in the suicide rate in Brazil. So I'm going to try to analyze the specific indices of this country.


```python
df_brasil = df[df['country'] == 'Brazil']
```

Country and country fields are all the same, then discarded.


```python
df_brasil.drop(['country', 'country_world'], axis = 1, inplace = True)
```

    /home/luis/anaconda3/envs/dl/lib/python3.6/site-packages/pandas/core/frame.py:3940: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


I'm going to repeat a lot of the graphics already done.


```python
suicides_no_year = []

for y in df_brasil['year'].unique():
    suicides_no_year.append(sum(df_brasil[df_brasil['year'] == y]['suicides_no']))

n_suicides_year = pd.DataFrame(suicides_no_year, columns=['suicides_no_year'])
n_suicides_year['year'] = df_brasil['year'].unique()

top_year = n_suicides_year.sort_values('suicides_no_year', ascending=False)['year']
top_suicides = n_suicides_year.sort_values('suicides_no_year', ascending=False)['suicides_no_year']

plt.figure(figsize=(8,5))
plt.xticks(rotation=90)
sns.barplot(x = top_year, y = top_suicides)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff848cb4e10>




![png](output_61_1.png)



```python
suicides_no_age = []

for a in df['age'].unique():
    suicides_no_age.append(sum(df_brasil[df_brasil['age'] == a]['suicides_no']))

plt.xticks(rotation=30)
sns.barplot(x = df_brasil['age'].unique(), y = suicides_no_age)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff8473727f0>




![png](output_62_1.png)



```python
suicides_no_sex = []

for s in df['sex'].unique():
    suicides_no_sex.append(sum(df_brasil[df_brasil['sex'] == s]['suicides_no']))

sns.barplot(x = df_brasil['sex'].unique(), y = suicides_no_sex)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff8472f21d0>




![png](output_63_1.png)



```python
suicides_no_gen = []
for g in df['generation'].unique():
    suicides_no_gen.append(sum(df_brasil[df_brasil['generation'] == g]['suicides_no']))

plt.figure(figsize=(8,5))
sns.barplot(x = df_brasil['generation'].unique(), y = suicides_no_gen)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff8472ace48>




![png](output_64_1.png)



```python
sns.scatterplot(x = 'gdp_for_year', y = 'suicides_no', data = df_brasil)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff84722e780>




![png](output_65_1.png)



```python
sns.scatterplot(x = 'gdp_per_capita', y = 'suicides_no', data = df_brasil)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff847194128>




![png](output_66_1.png)



```python
plt.figure(figsize=(8,7))
sns.heatmap(df_brasil.corr(), cmap = 'coolwarm', annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff8471e62e8>




![png](output_67_1.png)



```python

```
