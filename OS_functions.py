# date: 10/2019
# Author: Reginaldo Ferreira and Liza Macedo
# email: 2008reginaldo@gmail.com

#---------------------------------------------------------------------------
# Este arquivo contem as principais funções que foram usadas no notebook
# 'Projeto_Final_OS_v1.ipynb'. Cada função esta devidamente comentada.
#---------------------------------------------------------------------------


def getCititesCoordOffline(cities_dict):
    '''
     Esta função gera um dataframe com algumas informações importantes
     sobre as cidades que se pretende visitar.
     input: dicionário com o nome das cidades (como chave),
            e tempos de estadia mínima e máxima (tuple) em cada cidade.
     output: dataframe com o respectivo país estadias máxima e mínima,
             Latitude e Longitude de cada cidade
    '''

    # Fonte de onde são extraidos os dados País, lat e lon:
    df = pd.read_csv('worldcities.csv')

    # cities: lista com as cidades as quais se pretente visitar
    cities   = list(cities_dict.keys())

    # Arrays que receberão valore no laço for
    cityName = np.empty_like(cities).astype(str)
    country  = np.empty_like(cities).astype(str)
    eMin     = np.zeros(len(cities)).astype(int)
    eMax     = np.zeros(len(cities)).astype(int)
    lat      = np.zeros(len(cities))
    lon      = np.zeros(len(cities))

    # dic: Dicionário que receberá os valores e será transformado no dataframe final
    dic      = {}
    for n, city in np.ndenumerate(cities):
        _                = df[(df.city == city)].sort_values('population', ascending=False).iloc[0]
        cityName[n]      = _['city']
        country[n]       = _['country']
        eMin[n], eMax[n] = cities_dict[city]
        lat[n]           = _['lat']
        lon[n]           = _['lng']

    dic['cityName'] = cityName
    dic['country']  = country
    dic['eMin']     = eMin
    dic['eMax']     = eMax
    dic['lat']      = lat
    dic['lon']      = lon

    df = pd.DataFrame(dic)
    print (f'Quantidade mínima de dias de viagem: {df.eMin.sum()}')
    print (f'Quantidade mmáxima de dias de viagem: {df.eMax.sum()}')
    print ('-----------------------------------------')

    # Limpando memória
    cities   = 0
    cityName = 0
    country  = 0
    eMin     = 0
    eMax     = 0
    lat      = 0
    lon      = 0
    dic      = 0

    return df
#---------------------------------------------------------------------------


def distance(origin, destination):
    '''
    Esta função retorna a distância em km dadas as coordenadas entre dois pointos
    input-> origin: tuple -> com as coordenadas da origem
            destination: tuple -> com as coordenadas do destino
    output-> d: float > distancia entre os pontos.
    '''

    lat1, lon1 = origin
    lat2, lon2 = destination

    # Raio da Terra em km
    radius = 6371

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d
#---------------------------------------------------------------------------



def rotasGen(dataframe, vpd=1, data_inicio = '2020-01-01 00:00', dias_de_viagem = 30, ppkm= 0.15, sigma= 0.2):
    '''
    Este função retorna um dataframe com datas e preços de passagens gerados
    "aleatóriamente". Os preço de cada rota é extraído de uma distribuição
    normal cujo preço médio é proporcial à distancia entre as respectivas cidades.
    A relação entre o preço médio e a distância é dado por:
    preçoMédio = 0.10 * distancia. (default = 0.15 U$ por km)

    input-> dataframe: dataframe -> Dataframe com dados de entrada (cidade, país, emin, emax, lat, lon)
            vdp: int -> número de voos por dia.
            data_inicio: string -> Data de chegada na primeira cidade
            dias_de_viagens: int -> Duração da viagem (deve ser maior que a soma das estadias mínimas em cada cidade e menor que a soma das estadias máximas)
    output-> df: dataframe -> dataframe com origem, destino, horários, distâncias
    '''


    # number of rows of the new dataframe:
    rows = dataframe.shape[0]
    n = int(math.factorial(rows) / math.factorial(rows - 2)) * vpd * dias_de_viagem
    print (f'Gerando Dataframe com {n} variávais...')

    # definindo colunas
    dias     = np.array([data_inicio]*n, dtype='datetime64')
    dia_ref  = dias[0].copy()
    preco    = np.zeros(n)
    origem   = np.array(['']*n, dtype='<U16')
    destino  = np.array(['']*n, dtype='<U16')
    dist     = np.zeros(n)
    X = np.random.rand(n)

    listaDeCidades = dataframe['cityName']
    k = 0
    for orig in listaDeCidades:
        for dest in listaDeCidades[listaDeCidades != orig]:
            for dia in range(1, dias_de_viagem+1):
                for _ in range(vpd):
                    dias[k]      = dias[k] + dia * 1440 + np.random.randint(1440)
                    (olat, olon) = dataframe[dataframe.cityName == orig][['lat', 'lon']].to_numpy()[0]
                    (dlat, dlon) = dataframe[dataframe.cityName == dest][['lat', 'lon']].to_numpy()[0]
                    dist[k]      = np.around(distance((olat, olon),(dlat, dlon)), decimals=2)
                    preco[k]     = np.around((dist[k] * ppkm * (1 - sigma/2 + sigma*X[k]) ), decimals=2)
                    origem[k]    = orig
                    destino[k]   = dest
                    k += 1

    df = pd.DataFrame({'Origem': origem, 'Destino': destino, 'Distancia (km)': dist,
                       'Preço (US$)': preco, 'Horários-Voos': dias})

    dias     = 0
    dia_ref  = 0
    preco    = 0
    origem   = 0
    destino  = 0
    dist     = 0
    index    = 0
    print ('Pronto!')

    return df.sort_values(by=['Horários-Voos']).reset_index(drop=True)
#---------------------------------------------------------------------------



def plotMap(cities, x_lim = (-180,180), y_lim = (-90,90), size=(12,12)):
    worldMap = gpd.read_file('TM_WORLD_BORDERS-0.3.shp')
#     worldMap = gpd.read_file('TM_WORLD_BORDERS_SIMPL-0.3.shp')

    lat = cities['lat'].to_list()
    lon = cities['lon'].to_list()
    text = cities['cityName'].to_list()

    fig, ax = plt.subplots(1, figsize=size)
    worldMap.plot(ax=ax, color='gray', edgecolor='black', alpha=0.3)
    plt.scatter(lon, lat, color='blue' )
    for i, txt in enumerate(text):
        plt.annotate(txt, (lon[i]+0.3, lat[i]+0.3), color='blue', size=14 )
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.ylabel('Latitue')
    plt.xlabel('Longitude')
    plt.grid(ls='--')



# ----------------------------------------------------------------------------------

def plotMapRoutes(LpProb, df_cities, df, x_lim = (-180,180), y_lim = (-90,90), size=(12,12)):

    worldMap = gpd.read_file('TM_WORLD_BORDERS-0.3.shp')
#     worldMap = gpd.read_file('TM_WORLD_BORDERS_SIMPL-0.3.shp')

    mask_RE = [1 if i.value() == 1 else 0 for i in TPS.variables()]
    # rotas = np.array(mask_RE).astype(bool)
    _from   = df['Origem'].iloc[np.array(mask_RE).astype(bool)].to_numpy()
    _to     = df['Destino'].iloc[np.array(mask_RE).astype(bool)].to_numpy()
    preco   = df['Preço (US$)'].iloc[np.array(mask_RE).astype(bool)].to_numpy()
    dataHor = df['Horários-Voos'].iloc[np.array(mask_RE).astype(bool)].to_numpy()

    estadia = []
    for i in range(sum(mask_RE)):
        if i == 0:
            estadia.append(pd.Timestamp(dataHor[i]) - pd.Timestamp('2020-01-01 00:00'))
        else:
            estadia.append(pd.Timestamp(dataHor[i]) - pd.Timestamp(dataHor[i-1]))

    solution = pd.DataFrame({'Origem': _from, 'Destino': _to, 'Horários-Voos': dataHor,
                       'Valor Total (US$)': preco, 'Estadia cidade de Origem ': estadia})
    
    solution = solution.sort_values(by=['Horários-Voos']).reset_index(drop=True)

    print ('Valor da função objetivo:', pl.value(LpProb.objective))
    print ('Valor Total: ', solution['Valor Total (US$)'].sum())



    olat , olon, dlat , dlon = [], [], [], []
    for i in range(sum(mask_RE)):
        olat.append(df_cities[df_cities.cityName == _from[i]]['lat'].to_numpy()[0])
        olon.append(df_cities[df_cities.cityName == _from[i]]['lon'].to_numpy()[0])
        dlat.append(df_cities[df_cities.cityName == _to[i]]['lat'].to_numpy()[0])
        dlon.append(df_cities[df_cities.cityName == _to[i]]['lon'].to_numpy()[0])

    lat = df_cities['lat'].to_list()
    lon = df_cities['lon'].to_list()
    text = df_cities['cityName'].to_list()

    fig, ax = plt.subplots(1, figsize=size)
    worldMap.plot(ax=ax, color='gray', edgecolor='black', alpha=0.3)
    plt.scatter(lon, lat, color='blue' )
    for i, txt in enumerate(text):
        plt.arrow(olon[i], olat[i], dlon[i]-olon[i], dlat[i]-olat[i],
                  head_width=0.5, head_length=0.5, length_includes_head=True , fc='r', ec='r')
        plt.annotate(txt, (lon[i]+0.3, lat[i]+0.3), color='blue', size=14 )
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    plt.ylabel('Latitue')
    plt.xlabel('Longitude')
    plt.grid(ls='--')


    return solution
