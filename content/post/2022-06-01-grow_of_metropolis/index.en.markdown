---
title: Grow of Metropolis
categories: ["Python"]
tags: ["python", "geocoding"]
date: '2022-10-25'
toc: true
editor_options: 
  markdown: 
    wrap: 72
---

## About

Aims:

1.  FIlter out the data localisation where people are supposed to live
    in (cities; villages etc)

2.  Select cities which are metropolis based on population on external
    source

3.  Write a script which will ascribe metropolis to a city. Additionally
    model a process of consecutive cities which would be ascribed to the
    city bearing in mind that city will grow thanks to this process.

[Data Source](http://www.geonames.org/)

[Documentation](http://www.geonames.org/export/codes.html)

## Preparing environment

Setting default markdown option responsible of code chunk behaviour.



Firstly I choose prefered python environment on which I have installed
useful libraries.


```r
library(reticulate)
myenvs=conda_list()
envname=myenvs$name[3]
use_condaenv(envname, required = TRUE)
#reticulate::py_config()
```





Bellow I present two function:

-   radius - function which based on city population is calculating a
    radius within which city is able to absorb cities from this range

-   \_calcualate_metrocity_impact - calculate impact on metrocity on
    given city


```python
def radius(population):
    METRO_CITY_POPULATION_CONSTANT = -1/1443000
    MIN_METRO_CITY_RADIUS = 10
    MAX_METRO_CITY_RADIUS = 100 - MIN_METRO_CITY_RADIUS
    return MIN_METRO_CITY_RADIUS + MAX_METRO_CITY_RADIUS * (1 - np.exp(METRO_CITY_POPULATION_CONSTANT *  population))

def _calcualate_metrocity_impact(max_radius, distance_to_metro_city):
    METRO_CITY_POWER_CONSTANT = -1.4
    impact = np.exp(METRO_CITY_POWER_CONSTANT  * distance_to_metro_city / max_radius)
    return impact
```

Function responsible for calculating distances between 2 points on earth
surface.


```python
#https://towardsdatascience.com/heres-how-to-calculate-distance-between-2-geolocations-in-python-93ecab5bbba4
def haversine_distance_code(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)
```


```python
df= pd.read_csv(file_path_name, sep="\t", 
                names=['geonameid','name','asciiname','alternatenames','latitude','longitude','feature class','feature code','country code','cc2','admin1 code','admin2 code','admin3 code','admin4 code','population','elevation','dem','timezone','modification date',])

```

Dataset readme states that column feature classes contains level
information of:

A: country, state, region,...

H: stream, lake, ...

L: parks,area, ...

P: city, village,...

R: road, railroad

S: spot, building, farm

T: mountain,hill,rock,...

U: undersea

V: forest,heath,...

We will be interested on object of level P, and maybe A.


```python
import requests

url = 'http://www.geonames.org/export/codes.html'
html = requests.get(url).content
df_list = pd.read_html(html)
df_legend = df_list[-1]
df_legend = df_legend.rename(columns={df_legend.columns[0]: 'feature code',
                                     df_legend.columns[1]: 'short  descr',
                                     df_legend.columns[2]: 'long descr'})
df_legend = pd.merge(df[['feature code','feature class']].drop_duplicates(),df_legend, on='feature code')
df_legend
##     feature code  ...                                         long descr
## 0           PPLQ  ...                                                NaN
## 1            STM  ...  a body of running water moving to a lower leve...
## 2           HLLS  ...  rounded elevations of limited extent rising ab...
## 3            CNL  ...                          an artificial watercourse
## 4            PPL  ...  a city, town, village, or other agglomeration ...
## ..           ...  ...                                                ...
## 186          BCN  ...                 a fixed artificial navigation mark
## 187         HSEC  ...  a large house, mansion, or chateau, on a large...
## 188          RES  ...  a tract of public land reserved for future use...
## 189         STNR  ...  a facility for producing and transmitting info...
## 190         BLDA  ...  a building containing several individual apart...
## 
## [191 rows x 4 columns]
```


```python
df = df[df['feature class'].isin(['P','A'])]
df_check = pd.merge(df,df_legend, on=['feature code','feature class'])

# sorting by the biggest objects I can see that those are cities
df_check[df_check['feature class']=='P'].sort_values('population', ascending=False).head(5)

# administrative object located in object of level P
df_check[df_check['feature class']=='A'].sort_values('population', ascending=False).head(5)

#z tej tabeli wynika, ze PPLX to sekcje zaludnionych miejsc, sa to ulice, dzielnice, wiec wykluczam, sa czescia miast
df_check[['feature class','feature code', 'short  descr']].drop_duplicates()

#finalnie musze skupic sie na na obu klasach, jednoczesnie usuwajac duplikaty
df = df[(df['feature class'].isin(['P'])) & 
        (df.population != 0) & 
        ~(df['feature code'].isin(['PPLX']))].drop_duplicates('name')


df.index.name = 'city_id'
df.reset_index(inplace=True)
```


```python
df.groupby(['feature class','feature code']).agg({'population': ['mean', 'min', 'max']})
##                               population                  
##                                     mean      min      max
## feature class feature code                                
## P             PPL           3.238169e+03        5   244969
##               PPLA          3.652322e+05   118433   768755
##               PPLA2         3.543081e+04     5696   226794
##               PPLA3         5.619329e+03      110   248125
##               PPLC          1.702139e+06  1702139  1702139
##               PPLF          1.750000e+02      175      175
```

## Metropolis in Poland

[wikipedia](https://pl.wikipedia.org/wiki/Obszar_metropolitalny)
Warszawa, Katowice, Kraków, Łódź, Trójmiasto, Poznań, Wrocław,
Bydgoszcz, Szczecin, Lublin.


```python
df_metropolie = df[df.name.isin(
    ['Warsaw','Katowice','Kraków','Łódź',
     'Gdańsk','Gdynia',#'Trójmiasto',
     'Poznań','Wrocław','Bydgoszcz','Szczecin','Lublin'])][
    ['city_id','name','population','latitude','longitude']]
df_metropolie['iteration']=0 
#df_metropolie['radius'] = radius(df_metropolie['population'])
df_metropolie=df_metropolie.add_suffix('_metro')
df_metropolie
##       city_id_metro name_metro  ...  longitude_metro  iteration_metro
## 188            3191     Warsaw  ...         21.01178                0
## 917           12873     Lublin  ...         22.56667                0
## 1734          25287    Wrocław  ...         17.03333                0
## 1916          27732   Szczecin  ...         14.55302                0
## 2279          32047     Poznań  ...         16.92993                0
## 2654          36976       Łódź  ...         19.47395                0
## 2774          38634     Kraków  ...         19.93658                0
## 2889          40299   Katowice  ...         19.02754                0
## 3089          43232     Gdynia  ...         18.53188                0
## 3090          43241     Gdańsk  ...         18.64912                0
## 3298          45802  Bydgoszcz  ...         18.00762                0
## 
## [11 rows x 6 columns]
```

## metropolis absorption algorithm

### Instruction

1.  stworze id kolumne z indeksem

2.  zlacze tabele z metropoliami i wszystkimi miastami im do tej pory
    przypisanymi, wylicze zagregowana ludnosc oraz promien metropoli

3.  croos joinuje do kazdego miasta bez przypisanej metropolii tabele z
    metropolia

4.  wylicze odleglosc miejscowosci od metropoli i pozbede sie tych
    wierszy ktore sa poza promieniem

5.  dla pozostalych miejscowosci wylicze moc metropolii

6.  zrobie slice max groupujac po id miejscowosci pozostawiajc
    metropolie wchlaniajaca - tak powstanie tabela incrementalna do
    ktorej potem bede rbindowal nastepne tego typu tabele

7.  w obu tabelach powstanie tabele z indeksem mowiacy o n-iteracji z
    jakiej pochodzi przypisanie miejscowosci do metropolii oraz stan
    populacji

8.  wszystko zamkne w lupie while ktory bedzie wykonywany tak dlugo jak
    zostanie odnotowany przyrost w tabeli incrementalnej


```python
df_cities = df[['city_id','name','population','latitude','longitude']]
df_cities = df_cities.loc[~df_cities.city_id.isin(df_metropolie.city_id_metro)]
df_cities.head(5)
##    city_id      name  population  latitude  longitude
## 0       13  Prędocin         536  51.14791   21.32704
## 1       16     Poraj         266  50.89962   23.99191
## 2       37    Żyrzyn        1400  51.49918   22.09170
## 3       41  Żyrardów       41179  52.04880   20.44599
## 4       42   Żyraków        1400  50.08545   21.39622
```

### wlasciwy algorytm


```python
df_miasta_w_puli =df_cities
column_names = ['city_id','name','population'] +df_metropolie.columns.values.tolist()
df_miasta_wchloniete=pd.DataFrame(columns=column_names)
start = True
iteration =0


# start funkcji
while start == True:
    df_metropolie_powiekszone=df_metropolie.append(df_miasta_wchloniete, ignore_index=True)
    df_metropolie_powiekszone.population = df_metropolie_powiekszone.population.combine_first(df_metropolie_powiekszone.population_metro)
    
    df_metropolie_powiekszone_popul = df_metropolie_powiekszone.groupby(
        ['city_id_metro','name_metro','population_metro','latitude_metro','longitude_metro',]).agg(
        {'population':['sum']}).reset_index()
    df_metropolie_powiekszone_popul.columns = df_metropolie_powiekszone_popul.columns.droplevel(1)
    df_metropolie_powiekszone_popul['radius'] = radius(df_metropolie_powiekszone_popul['population'])
    df_miasta_w_puli['key'] = 1
    df_metropolie_powiekszone_popul['key'] = 1
    df_x = pd.merge(df_miasta_w_puli, df_metropolie_powiekszone_popul, on='key', suffixes=('','_y')).drop("key", 1)
    #calculating distance between two coordinates 
    #https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    distances_km = []
    for row in df_x.itertuples():
        distances_km.append(
            haversine_distance_code( row.latitude, row.longitude ,row.latitude_metro, row.longitude_metro)
        )
    df_x['distance_km'] = distances_km
    df_x = df_x[df_x.radius >= df_x.distance_km]
    df_x['impact'] = _calcualate_metrocity_impact(df_x.radius,df_x.distance_km)
    #stwierdzam do ktorej finalnie metropoli miejscowosci zostaje zaliczon
    idx = df_x.groupby(['name','population'])['impact'].transform(max) == df_x['impact']
    df_x = df_x[idx]
    iteration+= 1
    df_x['iteration_metro']=iteration
    pre_rows_num=df_miasta_wchloniete.shape[0]
    df_miasta_wchloniete=df_miasta_wchloniete.append(
        df_x[column_names], ignore_index=True)
    #pozbywam sie miast juz wchlonietych
    indx = df_miasta_w_puli.city_id.isin(df_miasta_wchloniete.city_id)
    df_miasta_w_puli = df_miasta_w_puli[~indx]
    if pre_rows_num == df_miasta_wchloniete.shape[0]:
        start = False
## <string>:12: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
## <string>:10: SettingWithCopyWarning: 
## A value is trying to be set on a copy of a slice from a DataFrame.
## Try using .loc[row_indexer,col_indexer] = value instead
## 
## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
df_metropolie_powiekszone_popul = df_metropolie_powiekszone.groupby(
    ['city_id_metro','name_metro','population_metro','latitude_metro','longitude_metro',]).agg(
    {'population':['sum']}).reset_index()
df_metropolie_powiekszone_popul.columns = df_metropolie_powiekszone_popul.columns.droplevel(1)
df_metropolie_powiekszone_popul['radius'] = radius(df_metropolie_powiekszone_popul['population'])

```


```python
#finalne populacje metropoli
df_metropolie_powiekszone_popul.head(5)

#przypisanie miast do metropoli wraz numerem iteracji
##    city_id_metro name_metro  ...  population     radius
## 0           3191     Warsaw  ...     5167905  97.494601
## 1          12873     Lublin  ...      531712  37.739146
## 2          25287    Wrocław  ...     1038518  56.178876
## 3          27732   Szczecin  ...      589846  40.197589
## 4          32047     Poznań  ...     1116528  58.484992
## 
## [5 rows x 7 columns]
df_miasta_wchloniete.head(5)
##   city_id           name  ... longitude_metro iteration_metro
## 0      41       Żyrardów  ...        21.01178               1
## 1     189       Zręczyce  ...        19.93658               1
## 2     215       Żoliborz  ...        21.01178               1
## 3     291          Złota  ...        21.01178               1
## 4     339  Zielonki-Wieś  ...        21.01178               1
## 
## [5 rows x 9 columns]
```
