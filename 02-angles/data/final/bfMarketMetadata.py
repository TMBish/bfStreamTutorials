def getBfRaces(dte):

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
                    'market_id': str(markets['MARKET_ID']),
                    'start_time': markets['START_TIME']
                }
            )
    
    marketDf = pd.DataFrame(marketList)

    return(marketDf)

def getBfRaceMeta(market_id):

    url = 'https://apigateway.betfair.com.au/hub/raceevent/{}'.format(market_id)

    responseJson = requests.get(url).json()

    raceList = []

    for runners in responseJson['runners']:
        raceList.append(
            {
                'market_id': market_id,
                'weather': responseJson['weather'],
                'track_condition': responseJson['trackCondition'],
                'race_distance': responseJson['raceLength'],
                'selection_id': runners['selectionId'],
                'selection_name': runners['runnerName'],
                'barrier': runners['barrierNo'],
                'place': runners['placedResult'],
                'trainer': runners['trainerName'],
                'jockey': runners['jockeyName'],
                'weight': runners['weight']
            }
        )

    raceDf = pd.DataFrame(raceList)

    return(raceDf)
