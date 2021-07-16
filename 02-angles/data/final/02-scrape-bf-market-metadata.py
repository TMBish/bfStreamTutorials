import requests
import pandas as pd
import datetime

def getBfMarkets(dte):

    url = 'https://apigateway.betfair.com.au/hub/racecard?date={}'.format(dte)

    responseJson = requests.get(url).json()

    marketList = []

    for meeting in responseJson['MEETINGS']:
        for markets in meeting['MARKETS']:
            marketList.append(
                {
                    'date': dte,
                    'track': meeting['VENUE_NAME'],
                    'country': meeting['COUNTRY'],
                    'race_type': meeting['RACE_TYPE'],
                    'race_number': markets['RACE_NO'],
                    'market_id': str('1.' + markets['MARKET_ID']),
                    'start_time': markets['START_TIME']
                }
            )
    
    marketDf = pd.DataFrame(marketList)

    return(marketDf)

def getBfRaceMeta(market_id):

    url = 'https://apigateway.betfair.com.au/hub/raceevent/{}'.format(market_id)

    responseJson = requests.get(url).json()

    if 'error' in responseJson:
        return(pd.DataFrame())

    raceList = []

    for runner in responseJson['runners']:

        if 'isScratched' in runner and runner['isScratched']:
            continue

        # Jockey not always populated
        try:
            jockey = runner['jockeyName']
        except:
            jockey = ""

        # Place not always populated
        try:
            placeResult = runner['placedResult']
        except:
            placeResult = ""

        # Place not always populated
        try:
            trainer = runner['trainerName']
        except:
            trainer = ""

        raceList.append(
            {
                'market_id': market_id,
                'weather': responseJson['weather'],
                'track_condition': responseJson['trackCondition'],
                'race_distance': responseJson['raceLength'],
                'selection_id': runner['selectionId'],
                'selection_name': runner['runnerName'],
                'barrier': runner['barrierNo'],
                'place': placeResult,
                'trainer': trainer,
                'jockey': jockey,
                'weight': runner['weight']
            }
        )

    raceDf = pd.DataFrame(raceList)

    return(raceDf)


def scrapeBfDate(dte):

    markets = getBfMarkets(dte)

    thoMarkets = markets.query('country == "AUS" and race_type == "R"')

    raceMetaList = []

    for market in thoMarkets.market_id:
        raceMetaList.append(getBfRaceMeta(market))

    raceMeta = pd.concat(raceMetaList)

    return(markets.merge(raceMeta, on = 'market_id'))


dataList = []
dateList = pd.date_range(datetime.date(2020,7,1),datetime.date.today()-datetime.timedelta(days=1),freq='d')

for dte in dateList:
    dte = dte.date()
    print(dte)
    dataList.append(scrapeBfDate(dte))

data = pd.concat(dataList)

data.to_csv("/media/hdd/tmp/thoroughbred-parsed/thoroughbred-race-data.csv")