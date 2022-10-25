---
title: Grow of Metropolis
categories: ["R"]
tags: ["python", "geocoding", "assesment center"]
date: '2022-10-25'
toc: true
---
 
 
[<img src="https://simpleicons.org/icons/github.svg" style="max-width:15%;min-width:40px;float:right;" alt="Github repo" />](https://github.com/yihui/hugo-xmin)



```r
Sys.setenv(RETICULATE_PYTHON = "/Users/lrabalski1/miniforge3/envs/everyday_use/bin/python")
reticulate::py_config()
```

```
## python:         /Users/lrabalski1/miniforge3/envs/everyday_use/bin/python
## libpython:      /Users/lrabalski1/miniforge3/envs/everyday_use/lib/libpython3.8.dylib
## pythonhome:     /Users/lrabalski1/miniforge3/envs/everyday_use:/Users/lrabalski1/miniforge3/envs/everyday_use
## version:        3.8.12 | packaged by conda-forge | (default, Oct 12 2021, 21:21:17)  [Clang 11.1.0 ]
## numpy:          /Users/lrabalski1/miniforge3/envs/everyday_use/lib/python3.8/site-packages/numpy
## numpy_version:  1.21.4
## 
## NOTE: Python version was forced by RETICULATE_PYTHON
```


```python
import pandas as pd
import numpy as np
#!pip install mpu --user
import mpu
import geopy.distance
```


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

```python
#https://towardsdatascience.com/heres-how-to-calculate-distance-between-2-geolocations-in-python-93ecab5bbba4
#funkcja do liczenia odleglosci miedzy dwiema wspołrzednymi
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
df= pd.read_csv('/Users/lrabalski1/Desktop/prv/x_predict/PL/PL copy.csv', sep="\t", 
                names=['geonameid','name','asciiname','alternatenames','latitude','longitude','feature class','feature code','country code','cc2','admin1 code','admin2 code','admin3 code','admin4 code','population','elevation','dem','timezone','modification date',])
df
```

```
##        geonameid                     name  ...       timezone modification date
## 0         462259                  Zodenen  ...  Europe/Warsaw        2015-09-05
## 1         477032              Variazhanka  ...  Europe/Warsaw        2021-08-04
## 2         490932                 Solokiya  ...  Europe/Warsaw        2021-02-07
## 3         502656                     Rata  ...  Europe/Warsaw        2014-07-08
## 4         558461  Hrodzyenskaye Uzvyshsha  ...  Europe/Warsaw        2010-09-15
## ...          ...                      ...  ...            ...               ...
## 57943   12324160           WA2 - Parzniew  ...  Europe/Warsaw        2021-09-16
## 57944   12324253                  Polanki  ...  Europe/Warsaw        2021-09-22
## 57945   12324489    Tunezeal Headquarters  ...  Europe/Warsaw        2021-10-04
## 57946   12358469                   Zębice  ...  Europe/Warsaw        2021-10-12
## 57947   12358470                  Sulęcin  ...  Europe/Warsaw        2021-10-12
## 
## [57948 rows x 19 columns]
```
# z readme wynika:
feature classes:
A: country, state, region,...
H: stream, lake, ...
L: parks,area, ...
P: city, village,...
R: road, railroad 
S: spot, building, farm
T: mountain,hill,rock,... 
U: undersea
V: forest,heath,...
## interesuje nas P oraz byc może A


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
```

```
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

#sorujac po najwiekszych obiektach widac ze mamy doczynienia z miastami
df_check[df_check['feature class']=='P'].sort_values('population', ascending=False)


#nazwy jednostek administracyjnych ktore znajduja sie w zbiorze obiektow P
```

```
##        geonameid  ...                                         long descr
## 47339     756135  ...                                                NaN
## 47350    3093133  ...  seat of a first-order administrative division ...
## 47351    3094802  ...  seat of a first-order administrative division ...
## 47346    3081368  ...  seat of a first-order administrative division ...
## 47348    3088171  ...  seat of a first-order administrative division ...
## ...          ...  ...                                                ...
## 16321     772147  ...  a city, town, village, or other agglomeration ...
## 16322     772148  ...  a city, town, village, or other agglomeration ...
## 16323     772149  ...  a city, town, village, or other agglomeration ...
## 16324     772150  ...  a city, town, village, or other agglomeration ...
## 50396   12111088  ...            a populated place that no longer exists
## 
## [47413 rows x 21 columns]
```

```python
df_check[df_check['feature class']=='A'].sort_values('population', ascending=False).head(50)

#z tej tabeli wynika, ze PPLX to sekcje zaludnionych miejsc, sa to ulice, dzielnice, wiec wykluczam, sa czescia miast
```

```
##        geonameid  ...                                         long descr
## 47378     798544  ...                                                NaN
## 47381     858787  ...  a primary administrative division of a country...
## 47305     756136  ...  an administrative division of a country, undif...
## 47391    3337497  ...  a primary administrative division of a country...
## 47392    3337498  ...  a primary administrative division of a country...
## 47380     858786  ...  a primary administrative division of a country...
## 47386    3337492  ...  a primary administrative division of a country...
## 47328    3093134  ...  an administrative division of a country, undif...
## 47387    3337493  ...  a primary administrative division of a country...
## 47390    3337496  ...  a primary administrative division of a country...
## 47335    3099435  ...  an administrative division of a country, undif...
## 47379     858785  ...  a primary administrative division of a country...
## 47382     858788  ...  a primary administrative division of a country...
## 47394    3337500  ...  a primary administrative division of a country...
## 47337    3102015  ...  an administrative division of a country, undif...
## 47416    6695624  ...  a subdivision of a first-order administrative ...
## 48624    7531926  ...  a subdivision of a second-order administrative...
## 47393    3337499  ...  a primary administrative division of a country...
## 47311     763168  ...  an administrative division of a country, undif...
## 47385     858791  ...  a primary administrative division of a country...
## 47316     769251  ...  an administrative division of a country, undif...
## 47384     858790  ...  a primary administrative division of a country...
## 47318     776070  ...  an administrative division of a country, undif...
## 47383     858789  ...  a primary administrative division of a country...
## 47389    3337495  ...  a primary administrative division of a country...
## 47388    3337494  ...  a primary administrative division of a country...
## 47403    6690154  ...  a subdivision of a first-order administrative ...
## 48491    7531791  ...  a subdivision of a second-order administrative...
## 47417    6697536  ...  a subdivision of a first-order administrative ...
## 48454    7531754  ...  a subdivision of a second-order administrative...
## 47492    7530801  ...  a subdivision of a first-order administrative ...
## 47995    7531292  ...  a subdivision of a second-order administrative...
## 48536    7531836  ...  a subdivision of a second-order administrative...
## 47549    7530858  ...  a subdivision of a first-order administrative ...
## 48589    7531890  ...  a subdivision of a second-order administrative...
## 47693    7531002  ...  a subdivision of a first-order administrative ...
## 48466    7531766  ...  a subdivision of a second-order administrative...
## 47531    7530840  ...  a subdivision of a first-order administrative ...
## 47685    7530994  ...  a subdivision of a first-order administrative ...
## 48081    7531378  ...  a subdivision of a second-order administrative...
## 47505    7530814  ...  a subdivision of a first-order administrative ...
## 49563    7532869  ...  a subdivision of a second-order administrative...
## 47661    7530970  ...  a subdivision of a first-order administrative ...
## 48293    7531591  ...  a subdivision of a second-order administrative...
## 47463    7530772  ...  a subdivision of a first-order administrative ...
## 47483    7530792  ...  a subdivision of a first-order administrative ...
## 48731    7532033  ...  a subdivision of a second-order administrative...
## 47459    7530768  ...  a subdivision of a first-order administrative ...
## 48309    7531607  ...  a subdivision of a second-order administrative...
## 47634    7530943  ...  a subdivision of a first-order administrative ...
## 
## [50 rows x 21 columns]
```

```python
df_check[['feature class','feature code', 'short  descr']].drop_duplicates()

#finalnie musze skupic sie na na obu klasach, jednoczesnie usuwajac duplikaty
```

```
##       feature class  ...                                     short  descr
## 0                 P  ...                        abandoned populated place
## 142               P  ...                                  populated place
## 43484             P  ...                       section of populated place
## 45716             P  ...    seat of a third-order administrative division
## 47126             P  ...   seat of a second-order administrative division
## 47304             A  ...                          administrative division
## 47339             P  ...                    capital of a political entity
## 47340             P  ...    seat of a first-order administrative division
## 47356             P  ...                                     farm village
## 47362             P  ...                               populated locality
## 47378             A  ...                     independent political entity
## 47379             A  ...              first-order administrative division
## 47395             A  ...             second-order administrative division
## 47775             A  ...  historical fourth-order administrative division
## 47780             P  ...                                              NaN
## 47781             A  ...              third-order administrative division
## 50259             P  ...                                 populated places
## 50262             P  ...                        religious populated place
## 50263             A  ...   historical third-order administrative division
## 50264             A  ...             fourth-order administrative division
## 50332             P  ...   seat of a fourth-order administrative division
## 50333             P  ...                       historical populated place
## 
## [22 rows x 3 columns]
```

```python
df = df[(df['feature class'].isin(['P'])) & 
        (df.population != 0) & 
        ~(df['feature code'].isin(['PPLX']))].drop_duplicates('name')

df.groupby('feature code').apply(lambda x: x.sample(1)).reset_index(drop=True)
```

```
##    geonameid       name  asciiname  ...  dem       timezone  modification date
## 0    3101453     Chyżne     Chyzne  ...  651  Europe/Warsaw         2010-10-30
## 1    3102014  Bydgoszcz  Bydgoszcz  ...   37  Europe/Warsaw         2019-09-05
## 2     765927   Lubartów   Lubartow  ...  159  Europe/Warsaw         2013-10-31
## 3     759814   Rytwiany   Rytwiany  ...  180  Europe/Warsaw         2010-09-30
## 4     756135     Warsaw     Warsaw  ...  113  Europe/Warsaw         2019-11-04
## 5     766060       Łoje       Loje  ...  111  Europe/Warsaw         2014-10-02
## 
## [6 rows x 19 columns]
```

```python
df.index.name = 'city_id'
df.reset_index(inplace=True)
```

```python
df.groupby(['feature class','feature code']).agg({'population': ['mean', 'min', 'max']})
```

```
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

# Metropolie w Polsce (wikipedia: https://pl.wikipedia.org/wiki/Obszar_metropolitalny)
Warszawa,
Katowice,
Kraków,
Łódź,
Trójmiasto,
Poznań,
Wrocław,
Bydgoszcz,
Szczecin,
Lublin.


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
```

```
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
#algorytm przypisywania metropolii
## Instrukcja
0. stworze id kolumne z indeksem
1. zlacze tabele z metropoliami i wszystkimi miastami im do tej pory przypisanymi, wylicze zagregowana ludnosc oraz  promien metropoli
2. croos joinuje do kazdego miasta bez przypisanej metropolii tabele z metropolia 
2. wylicze odleglosc miejscowosci od metropoli i pozbede sie tych wierszy ktore sa poza promieniem
3. dla pozostalych miejscowosci wylicze moc metropolii
4. zrobie slice max  groupujac po id miejscowosci pozostawiajc metropolie wchlaniajaca - tak powstanie tabela incrementalna do ktorej potem bede rbindowal nastepne tego typu tabele
5. w obu tabelach powstanie tabele z indeksem mowiacy o n-iteracji z jakiej pochodzi przypisanie miejscowosci do metropolii oraz stan populacji

6. wszystko zamkne w lupie while ktory bedzie wykonywany tak dlugo jak zostanie odnotowany przyrost w tabeli incrementalnej


```python
df_cities = df[['city_id','name','population','latitude','longitude']]
df_cities = df_cities.loc[~df_cities.city_id.isin(df_metropolie.city_id_metro)]
df_cities
```

```
##       city_id            name  population  latitude  longitude
## 0          13        Prędocin         536  51.14791   21.32704
## 1          16           Poraj         266  50.89962   23.99191
## 2          37          Żyrzyn        1400  51.49918   22.09170
## 3          41        Żyrardów       41179  52.04880   20.44599
## 4          42         Żyraków        1400  50.08545   21.39622
## ...       ...             ...         ...       ...        ...
## 3571    55679           Gądki         529  52.31202   17.04696
## 3572    55894  Stare Kaleńsko         333  53.52572   16.15265
## 3573    57439       Lipczynek          40  53.88156   17.25829
## 3574    57590   Wola Bykowska         200  51.45739   19.65062
## 3575    57591        Krajanów         160  50.59751   16.44484
## 
## [3565 rows x 5 columns]
```
## wlasciwy algorytm

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
```

```
## <string>:12: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
## <string>:10: SettingWithCopyWarning: 
## A value is trying to be set on a copy of a slice from a DataFrame.
## Try using .loc[row_indexer,col_indexer] = value instead
## 
## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
```

```python
df_metropolie_powiekszone_popul = df_metropolie_powiekszone.groupby(
    ['city_id_metro','name_metro','population_metro','latitude_metro','longitude_metro',]).agg(
    {'population':['sum']}).reset_index()
df_metropolie_powiekszone_popul.columns = df_metropolie_powiekszone_popul.columns.droplevel(1)
df_metropolie_powiekszone_popul['radius'] = radius(df_metropolie_powiekszone_popul['population'])

```


```python
#finalne populacje metropoli
df_metropolie_powiekszone_popul

#przypisanie miast do metropoli wraz numerem iteracji
```

```
##     city_id_metro name_metro  ...  population     radius
## 0            3191     Warsaw  ...     5167905  97.494601
## 1           12873     Lublin  ...      531712  37.739146
## 2           25287    Wrocław  ...     1038518  56.178876
## 3           27732   Szczecin  ...      589846  40.197589
## 4           32047     Poznań  ...     1116528  58.484992
## 5           36976       Łódź  ...     1522125  68.657469
## 6           38634     Kraków  ...     1624389  70.801803
## 7           40299   Katowice  ...     4402773  95.742488
## 8           43232     Gdynia  ...      445607  33.910908
## 9           43241     Gdańsk  ...      785065  47.764670
## 10          45802  Bydgoszcz  ...     1011566  55.352705
## 
## [11 rows x 7 columns]
```

```python
df_miasta_wchloniete
```

```
##      city_id                  name  ... longitude_metro iteration_metro
## 0         41              Żyrardów  ...        21.01178               1
## 1        189              Zręczyce  ...        19.93658               1
## 2        215              Żoliborz  ...        21.01178               1
## 3        291                 Złota  ...        21.01178               1
## 4        339         Zielonki-Wieś  ...        21.01178               1
## ...      ...                   ...  ...             ...             ...
## 1781   41015                Jeżewo  ...        18.00762               6
## 1782   44601      Dąbrowa Biskupia  ...        18.00762               6
## 1783   47909  Aleksandrów Kujawski  ...        18.00762               6
## 1784   32645              Płużnica  ...        18.00762               7
## 1785   38419             Kruszwica  ...        18.00762               7
## 
## [1786 rows x 9 columns]
```
