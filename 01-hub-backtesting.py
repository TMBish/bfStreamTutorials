#!/usr/bin/env python
# coding: utf-8

# # Backtesting wagering models with Betfair JSON stream data

# ## 0.1 Context
# 
# Backtesting is the life-blood of most succesful wagering systems. In short it attempts to answer a single question for you:
# 
# > $ \tau $ : How much money will I win or lose if I started using this system to place bets with real money? 
# 
# Without a rigorous and quantitative backtesting approach it's really quite hard to estimate the answer to this question $ \tau $ that will be even reliably on the right side of zero. 
# 
# You could live test your system with real bets at small stakes, however, this isn't the panacea it seems. It will take time (more than you think) for your results to converge to their long term expectation. How long? Answering this question will require some expertise with probability and statistics you might not have. Even more than that though is that depending on where you're betting your results at small stakes could be very different than at larger stakes. You might not be able get a good answer to $ \tau $ until betting at full stakes at which point finding the answer might coincide with blowing up your gambling bankroll.
# 
# Backtesting is also very hard. To perfectly backtest your own predicted probablility on a historical race or sporting match you need to produce 2 things:
# 
# > (1) What would my predicted chance have been **exactly** for this selection in this market on this day in the past?
# 
# > (2) What would have I decided to bet **at what odds (exactly)** and **for how much stake (exactly)** based on this prediction? 
# 
# The devil in the detail of backtesting tends to be in those exactlys. 
# 
# The aim of the backtesting game is answering (2) as accurately as possible because it tells you exactly how much you would have made over your backtesting period, from there you can confidently project that rate of profitability forward. 
# 
# It's easy to make mistakes and small errors in the quantitative reasoning can lead you to extremely misguided projections downstream. 
# 
# Question (1) won't be in the scope of this notebook but it's equally (and probably more) important thant (2) but it is the key challenge of all predictive modelling exercises so there's plenty of discussion about it elsewhere.

# ## 0.2 Backtesting on Betfair
# 
# Answering quistion (2) for betting on the betfair exchange is difficult. The exhange is a dynamic system that changes from one micro second to the next.
# 
# > What number should you use for odds? How much could you assume to get down at those odds?
# 
# The conventional and easiest approach is to backtest at the BSP. The BSP is simple because it's a single number (to use for both back and lay bets) and is a taken price (there's no uncertainty about getting matched). Depending on the liquidity of the market a resonably sized stake might also not move the BSP very much. For some markets you may be able to safely assume you could be $10s of dollars at the BSP without moving it an inch. However, that's definitely not true of all BSP markets and you need to be generally aware that your betfair orders in the future **will** change the state of the exchange, and large bets **will** move the BSP in an unfavourable direction.
# 
# Aside from uncertainty around the liquidity and resiliance of the BSP, many many markets don't have a BSP. So what do we do then?
# 
# Typically what a lot of people (who have a relationship with betfair australia) do at this point is request a data dump. They might request an odds file for all australian harness race win markets since june 2018 with results and 4 different price points: the BSP, the last traded price, the weighted average price (WAP) traded in 3 minutes before the race starts, and the WAP for all bets matched prior to 3 mins before the race. 
# 
# However, you will likely need to be an existing VIP customer to get this file and it's not a perfect solution: it might take 2 weeks to get, you can't refresh it, you can't test more hypothetical price points after your initial analysis amongst many other problems. 
# 
# > What if you could produce this valuable data file yourself?

# ## 0.3 Betfair Stream Data
# 
# Betfair's historical stream data is an extremely rich source of data. However, in it's raw form it's difficult to handle for the uninitiated. It also might not be immediately obvious how many different things this dataset could be used for without seeing some examples. These guides will hopefully demystify how to turn this raw data into a familiar and usable format whilst also hopefully providing some inpiration for the kinds of value that can be excavated from it.

# ## 0.4 This example: backtesting Betfair Hub thoroughbred model

# To illustrate how you can use the stream files to backtest the outputs of a rating system we'll use the Australian Thoroughbred Rating model available on the Betfair Hub. The most recent model iteration only goes back till Feb 28th 2021 however as an illustrative example this is fine. We'd normally want to backtest with a lot more historical data than this, which just means in this case our estimation of future performance will be unreliable.
# 
# I'm interested to see how we would have fared betting all selections rated by this model according to a few different staking schemes and also at a few different times / price points. 

# In[2]:


# Some python setup
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# ## 1.0 Scrape The Model Ratings

# If you travel to the [betfair hub ratings page](https://www.betfair.com.au/hub/horse-racing-tips/) you'll find that URL links behind the ratings download buttons have a consistent URL pattern that looks very scrape friendly.

# ![](img/finding-hub-ratings-url.PNG)

# We can take advantage of this consistency and use some simple python code to scrape all the ratings into a pandas dataframe.

# In[3]:


import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

# Function to return Pandas DF of hub ratings for a particular date
def getHubRatings(dte):
    
    # Substitute the date into the URL
    url = 'https://betfair-data-supplier-prod.herokuapp.com/api/widgets/kash-ratings-model/datasets?date={}presenter=RatingsPresenter&json=true'.format(dte)
    
    # Convert the response into JSON
    responseJson = requests.get(url).json()
        
    hubList = []
    
    if not responseJson:
        return(None)
    
    
    # Want an normalised table (1 row per selection)
    # Brute force / simple approach is to loop through meetings / races / runners and pull out the key fields
    for meeting in responseJson['meetings']:
        for race in meeting['races']:
            for runner in race['runners']:
                hubList.append(
                    {
                        'date': dte,
                        'track': meeting['name'],
                        'race_number': race['number'],
                        'race_name': race['name'],
                        'market_id': race['bfExchangeMarketId'],
                        'selection_id':  str(runner['bfExchangeSelectionId']),
                        'selection_name': runner['name'],
                        'model_odds': runner['ratedPrice']
                    }
                )
                
    out = pd.DataFrame(hubList)
                
    return(out)


# In[4]:


# See the response from a single day
getHubRatings(date(2021,3,1)).head(5)


# In[5]:


# Loop through all recent history
dateDFList = []
dateList = pd.date_range(date(2021,2,18),date.today()-timedelta(days=1),freq='d')

for dte in dateList:
    dateDFList.append(getHubRatings(dte))
    
# Concatenate (add rows to rows) all the dataframes within the list
hubRatings = pd.concat(dateDFList)
# hubRatings.to_csv("outputs/hub-ratings.csv")


# In[6]:


hubRatings.shape


# # 2.0 Assembling the odds file

# So part 1 was very painless. This is how we like data: served by some API or available in a nice tabular format on a webpage ready to be scraped with standard tools available in popular languages.
# 
# Unfortunately, it won't be so painless to assemble our odds file. We'll find out why it's tricky as we go.

# ## 2.1 The Data

# The data we'll be using is the historical exchange data available from [this website](https://historicdata.betfair.com/#/home). The data available through this service is called streaming JSON data. There are a few options available relating to granularity (how many time points per second the data updates at) but we'll be using the most granular "PRO" set which has updates every 50 milliseconds. 
# 
# Essentially what the data allows us to do is, for a particular market, recreate the exact state of the betfair exchange at say: 150 milliseconds before the market closed. When people say the **state of the exchange** they mean two things a) what are all the current open orders on all the selections b) what are the current traded volumes on each selection at each price point. We obviously don't have access to any information about which accounts are putting up which prices and other things betfair has themselves. We're essentially getting a snapshot of what you can see through the website by clicking on each selection manually and looking at the graphs, tables and ladders.
# 
# However, with just these 2 pieces of information we can build a rich view of the dynamics of exchange and also build out all of the summary metrics (WAP etc) we might have previously needed betfair to help with.
# 
# For our purposes 50 milli-second intervaled data is huge overkill. But you could imagine needing this kind of granularity for other kinds of wagering systems - eg a high frequency trading algorithm of some sort that needs to make many decisions and actions every second. 
# 
# Let's take a look at what the stream data looks like for a single market:

# ![](img/stream-data-example.PNG)

# So it looks pretty intractable. For this particular market there's 14,384 lines which each consists of a single json packet of data. If you're not a data engineer (neither am I) your head might explode thinking about how you could read this into your computer and transform it into something usable.
# 
# The data looks like this because it is saved from a special betfair API called the Stream API which which is used by high end betfair API users and which delivers fast speeds other performance improvements over the normal "polling" API.
# 
# Now what's good about that, for the purposes of our exercise, is that the very nice python package `betfairlightweight` has the functionality built to not only parse the Stream API when connected to it live but also these historical saved versions of the stream data. Without it we'd be *very* far away from the finish line, with `betfairlightweight` we're pretty close.

# ## 2.2 Unpacking / flattening the data
# 
# Because these files are so large and unprocessed this process won't look the same as your normal data ETL in python: where you can read a raw data file (csv, json, text etc.) into memory and use python functions to transform into usable format.
# 
# I personally had no idea how to use python and `betfairlightweight` to parse these data until I saw [betfair's very instructive overview](https://betfair-datascientists.github.io/historicData/jsonToCsvTutorial/) which you should read for a more detailed look at some of the below code. 
# 
# By my count there were 4 key conceptual components that I had to get my head around to understand and be able to repurpose that code. So if you're like me (a bit confused by some of the steps in that piece) this explanation might help.
# 
# I'll assume you don't do any decompression and keep the monthly pro files as the `.tar` archives as they are.
# 
# Conceptually the process looks something like this:
# 
# #### (1) Load the "archives" into a "generator"
# #### (2) Scan across the generator (market_ids) and the market states within those markets to extract useful objects
# #### (3) Process those useful objects to pull out some metatdata + useful summary numbers derived from the available orders and traded volumes snapshot data
# #### (4) Write this useful summarised data to a file that can be read and understood with normal data analysis workflows
# 
# 
# 

# First we'll run a bunch of setup code setting up my libraries and creating some utility functions that will be used throughout the main parsing component. It'll also point to the two stream files I'll be parsing for this exercise.

# In[7]:


# Setup
import pandas as pd
import os
import re
import betfairlightweight
from betfairlightweight import StreamListener
import logging
import requests
import tarfile
import zipfile
import bz2
import glob
from unittest.mock import patch
from typing import List, Set, Dict, Tuple, Optional
from itertools import zip_longest
import functools
import betfairlightweight
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)
### BF LIGHTWEIGHT BOILERPLATE

# create trading instance (don't need username/password)
trading = betfairlightweight.APIClient("username", "password")

# create listener
listener = StreamListener(max_latency=None)

### Utility Functions

# rounding to 2 decimal places or returning '' if blank
def as_str(v: float) -> str:
    return '%.2f' % v if v is not None else ''

# splitting race name and returning the parts 
def split_anz_horse_market_name(market_name: str) -> (str, str, str):
    # return race no, length, race type
    # input sample: R6 1400m Grp1
    parts = market_name.split(' ')
    race_no = parts[0] # return example R6
    race_len = parts[1] # return example 1400m
    race_type = parts[2].lower() # return example grp1, trot, pace

    return (race_no, race_len, race_type)

# creating a flag that is True when markets are australian thoroughbreds
def filter_market(market: MarketBook) -> bool: 
    d = market.market_definition
    return (d.country_code == 'AU' 
        and d.market_type == 'WIN' 
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace')


# ### (1) `.tar` load

# * This function I stole from betfair's instructional article
# * The stream files are downloaded as `.tar` archive files which are a special kind of file that we'll need to unpack
# * Instead of loading each file into memory this function returns a "generator" which is a special python object that is to be iterated over
# * This basically means it contains the instructions to unpack and scan over files on the fly
# * This function also contains the logic to deal with if these files are zip archives or you've manually unpacked the archive and have the `.bz2` zipped files

# In[8]:


# loading from tar and extracting files
def load_markets(file_paths):
    for file_path in file_paths:
        print(file_path)
        if os.path.isdir(file_path):
            for path in glob.iglob(file_path + '**/**/*.bz2', recursive=True):
                f = bz2.BZ2File(path, 'rb')
                yield f
                f.close()
        elif os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1]
            # iterate through a tar archive
            if ext == '.tar':
                with tarfile.TarFile(file_path) as archive:
                    for file in archive:
                        yield bz2.open(archive.extractfile(file))
            # or a zip archive
            elif ext == '.zip':
                with zipfile.ZipFile(file_path) as archive:
                    for file in archive.namelist():
                        yield bz2.open(archive.open(file))

    return None


# ### (2) Scan across market states and extract useful objects
# 
# * So this function will take a special "stream" object which we'll create with `betfairlightweight`
# * The function takes a stream object input and returns 4 instances of the market state
#     + The market state 3 mins before the scheduled off
#     + The market state immediately before it goes inplay
#     + The market state immediately before it closes for settlement
#     + The final market state with outcomes
# * It basically just loops over all the market states and has a few checks to determine if it should save the current market state as key variables and then returns those

# In[9]:


# Extract Components From Generated Stream
def extract_components_from_stream(s):
    
    with patch("builtins.open", lambda f, _: f):   
    
        # Will return 3 market books t-3mins marketbook, the last preplay marketbook and the final market book
        evaluate_market = None
        prev_market = None
        postplay_market = None
        preplay_market = None
        t3m_market = None

        gen = stream.get_generator()

        for market_books in gen():
            
            for market_book in market_books:

                # If markets don't meet filter return None's
                if evaluate_market is None and ((evaluate_market := filter_market(market_book)) == False):
                    return (None, None, None, None)

                # final market view before market goes in play
                if prev_market is not None and prev_market.inplay != market_book.inplay:
                    preplay_market = market_book
                    
                # final market view before market goes is closed for settlement
                if prev_market is not None and prev_market.status == "OPEN" and market_book.status != prev_market.status:
                    postplay_market = market_book

                # Calculate Seconds Till Scheduled Market Start Time
                seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()
                    
                # Market at 3 mins before scheduled off
                if t3m_market is None and seconds_to_start < 3*60:
                    t3m_market = market_book
                    
                # update reference to previous market
                prev_market = market_book

        # If market didn't go inplay
        if postplay_market is not None and preplay_market is None:
            preplay_market = postplay_market
        
        return (t3m_market, preplay_market, postplay_market, prev_market) # Final market is last prev_market


# ### (3) + (4) Summarise those useful objects and write to `.csv`
# 
# * This next chunk contains a wrapper function that will do all the execution
# * It will open a csv output file
# * Use the load_markets utility to iterate over the `.tar` files
# * Use `betfairlightweight` to instantiate the special stream object
# * Pass that stream object to the `extract_components_from_stream` which will scan across the market states and pull out 4 key market books
# * Convert those marketbooks into simple summary numbers or dictionaries that will be written to the output `.csv` file

# In[10]:


def run_stream_parsing():
    
    # Run Pipeline
    with open("outputs/tho-odds.csv", "w+") as output:

        # Write Column Headers To File
        output.write("market_id,event_date,country,track,market_name,selection_id,selection_name,result,bsp,ltp,matched_volume,atb_ladder_3m,atl_ladder_3m\n")

        for file_obj in load_markets(data_path):

            # Instantiate a "stream" object
            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            # Extract key components according to the custom function above (outputs 4 objects)
            (t3m_market, preplay_market, postplay_market, final_market) = extract_components_from_stream(stream)

            # If no price data for market don't write to file
            if postplay_market is None:
                continue; 

            # Runner metadata and key fields available from final market book
            runner_data = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in final_market.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'sp': r.sp.actual_sp
                }
                for r in final_market.runners 
            ]

            # Last Traded Price
            # _____________________

            # From the last marketbook before inplay or close
            ltp = [runner.last_price_traded for runner in preplay_market.runners]

            # Total Matched Volume  
            # _____________________

            # Calculates the traded volume across all traded price points for each selection
            def ladder_traded_volume(ladder):
                return(sum([rung.size for rung in ladder]))

            selection_traded_volume = [ ladder_traded_volume(runner.ex.traded_volume) for runner in postplay_market.runners ]

            # Top 3 Ladder
            # ______________________

            # Extracts the top 3 price / stakes in available orders on both back and lay sides. Returns python dictionaries

            def top_3_ladder(availableLadder):
                out = {}
                price = []
                volume = []
                if len(availableLadder) == 0:
                    return(out)        
                else:
                    for rung in availableLadder[0:3]:
                        price.append(rung.price)
                        volume.append(rung.size)
                    out["price"] = price
                    out["volume"] = volume
                    return(out)

            # Sometimes t-3 mins market book is empty
            try:
                atb_ladder_3m = [ top_3_ladder(runner.ex.available_to_back) for runner in t3m_market.runners]
                atl_ladder_3m = [ top_3_ladder(runner.ex.available_to_lay) for runner in t3m_market.runners]
            except:
                atb_ladder_3m = {}
                atl_ladder_3m = {}

            # Writing To CSV
            # ______________________

            for (runnerMeta, ltp, selection_traded_volume, atb_ladder_3m, atl_ladder_3m) in zip(runner_data, ltp, selection_traded_volume, atb_ladder_3m, atl_ladder_3m):

                if runnerMeta['selection_status'] != 'REMOVED':

                    output.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{} \n".format(
                            str(final_market.market_id),
                            final_market.market_definition.market_time,
                            final_market.market_definition.country_code,
                            final_market.market_definition.venue,
                            final_market.market_definition.name,
                            runnerMeta['selection_id'],
                            runnerMeta['selection_name'],
                            runnerMeta['selection_status'],
                            runnerMeta['sp'],
                            ltp,
                            selection_traded_volume,
                            '"' + str(atb_ladder_3m) + '"', # Forcing the dictionaries to strings so we don't run into issues loading the csvs with the dictionary commas
                            '"' + str(atl_ladder_3m) + '"'
                        )
                    )


# In[11]:


# This will execute the files (it took me ~2 hours for 2 months of data)
#run_stream_parsing()


# **Extra Thoughts**
# 
# * The only thing to note is that because this process is very slow you might want to save more information than you think you need
# * I currently think I only want the best back and lay prices at t-3 mins before the off but I've saved the top 3 boxes in the available to back and lay ladders as dictionary strings
# * From these ladders I can calculate not only just the best back and lay prices but also WAP prices and also sizes at those boxes which I could use for much more accurate backtesting if I wanted to later without having the run the whole process again
# * However, I could easily save the entire open and traded orders ladders in the same way amongst other ways of retaining more of the data for post-processing analysis

# ![](img/ladder-columns.png)

# # 3.0 Backtesting Analysis

# Let's take stock of where we are. We currently have model ratings (about 1.5 months worth) and Betfair Odds (2 months worth). 
# 
# Circling back to the original backtesting context we needed to solve for 2 key questions:
# 
# 1. What would my predicted chance have been exactly for this selection in this market on this day in the past?
# 2. What would have I decided to bet at what odds (exactly) and for how much stake (exactly) based on this prediction?
# 
# Backtesting with someone else's publicly available and historically logged ratings solves question 1. With these ratings we're fine but generally we should just be aware there are some sketchy services that might make retroactive adjustments to historical ratings to juice their performance which obviously violates 1.
# 
# For the second part we now have several real betfair odds values to combine with the ratings and some chosen staking formula to simulate actual bets. I won't dwell too much on the stake size component but it's important.
# 
# Similarly we aren't out of the woods with the "what odds exactly" question either. I'll show performance of backtesting at the "Last Traded Price" however, there's literally no way of actually being the last bet matched order on every exchange market so there's some uncertainty in a few of these prices. 
# 
# Further, and from experience, if you placing bets at the BSP and you're using some form of proportional staking (like kelly) then you're calculated stake size will need to include a quantity (the BSP) which you will literally never be 100% sure of. You'll need to estimate the BSP as close to market suspension as you can and place your BSP bets with a stake sized derived from that estimation. This imprecision in stake calculation WILL cost you some profit relative to your backtested expectation. 
# 
# These are the gory details of the many ways becoming succesful on betfair is difficult. Generally, you'll need to spend quite a while in this section: testing, ironing out all these little kinks and trying to account for all your uncertainties. I'm just running over the skeleton.

# ## 3.1 Setting up your master data

# In[12]:


# First we'll load and tidy our odds data

# Load in odds file we created above
bfOdds = pd.read_csv("outputs/tho-odds.csv", dtype={'market_id': object, 'selection_id': object, 'atb_ladder_3m': object, 'atl_ladder_3m': object})

# Convert dictionary columns
import ast
bfOdds['atb_ladder_3m'] = [ast.literal_eval(x) for x in bfOdds['atb_ladder_3m']]
bfOdds['atl_ladder_3m'] = [ast.literal_eval(x) for x in bfOdds['atl_ladder_3m']]

# Convert LTP to Numeric
bfOdds['ltp'] = pd.to_numeric(bfOdds['ltp'], errors='coerce')

# Filter after 18th Feb
bfOdds = bfOdds.query('event_date >= "2021-02-18"')

bfOdds.head(5)


# When backtesting, and developing wagering systems more generally, I've found it really helpful to have a set of standard patterns or ways of representing common datasets. For a task like this it's really helpful to keep everything joined and together in a wide table.
# 
# So we want a dataframe with everything we need to conduct the backtest: your model ratings, the odds you're betting at, the results on the bets, and ultimately betting logic will all become columns in a dataframe.
# 
# It's helpful to have consistent column names so that the code for any new test you run looks much like previous tests and you can leverage custom functions that can be reused across tests and other projects. I like to have the following columns in my backtesting dataframe:
# * date
# * market_id (can be a surrogate id if dealing with fixed odds markets)
# * selection_id (could be selection name)
# * win (a binary win loss)
# * model_odds
# * model_prob
# * market_odds
# * market_prob
# * bet_side
# * stake
# * gpl
# * commission
# * npl
# 
# This analysis will be a little more complex as we're considering different price points so I'll leave out the `market_odds` and `market_prob` columns.

# In[119]:


# Joining the ratings data and odds data and combining
rawDF = pd.merge(
        hubRatings[hubRatings['market_id'].isin(bfOdds.market_id.unique())], 
        bfOdds[['market_name', 'market_id', 'selection_id', 'result', 'matched_volume', 'bsp', 'ltp', 'atb_ladder_3m', 'atl_ladder_3m']],
        on = ['market_id', 'selection_id'],
        how = 'inner'
    )

# Filter out markets with missing odds values
# rawDF = rawDF.groupby('market_id').filter(lambda x: (x.bsp.notnull()).all())

rawDF


# In[120]:



df = (
    rawDF
    # Extra Best Back + Lay 3 mins before of
    .assign(best_back_3m = lambda x: [np.nan if d.get('price') is None else d.get('price')[0] for d in x['atb_ladder_3m']])
    .assign(best_lay_3m = lambda x: [np.nan if d.get('price') is None else d.get('price')[0] for d in x['atl_ladder_3m']])
    # Coalesce LTP to BSP (about 60 rows)
    .assign(ltp = lambda x: np.where(x["ltp"].isnull(), x["bsp"], x["ltp"]))
    # Add a binary win / loss column
    .assign(win=lambda x: np.where(x['result'] == "WINNER", 1, 0))
    # Extra columns
    .assign(model_prob=lambda x: 1 / x['model_odds'])
    # Reorder Columns
    .reindex(columns = ['date', 'track', 'race_number', 'market_id', 'selection_id', 'bsp', 'ltp','best_back_3m','best_lay_3m','atb_ladder_3m', 'atl_ladder_3m', 'model_prob', 'model_odds', 'win'])

)

df.head(5)


# ## 3.2 Staking + Outcome Functions
# 
# Now we can create a set of standard staking functions that a dataframe with an expected set of columns and add staking and bet outcome fields.
# 
# We'll also add the ability of these functions to reference a different odds column so that we can backtest against our different price points.
# 
# For simplicity we'll assume you're paying 5% commission on winnings however it could be higher or lower depending on your betfair account.

# In[121]:


def bet_apply_commission(df, com = 0.05):
    
    # Total Market GPL
    df['market_gpl'] = df.groupby('market_id')['gpl'].transform(sum)

    # Apply 5% commission
    df['market_commission'] = np.where(df['market_gpl'] <= 0, 0, 0.05 * df['market_gpl'])

    # Sum of Market Winning Bets
    df['floored_gpl'] = np.where(df['gpl'] <= 0, 0, df['gpl'])
    df['market_netwinnings'] = df.groupby('market_id')['floored_gpl'].transform(sum)

    # Partition Commission According to Selection GPL
    df['commission'] = np.where(df['market_netwinnings'] == 0, 0, (df['market_commission'] * df['floored_gpl']) / (df['market_netwinnings']))

    # Calculate Selection NPL
    df['npl'] = df['gpl'] - df['commission']
    
    # Drop excess columns
    df = df.drop(columns = ['floored_gpl', 'market_netwinnings', 'market_commission', 'market_gpl'])
    
    return(df)
    

def bet_flat(df, stake = 1, back_odds = 'market_odds', lay_odds = 'market_odds'):
    
    """
    Betting DF should always contain: model_odds, and win (binary encoded), and the specified odds column columns
    """
    
    df['bet_side'] = np.where((df["model_odds"] >= df[back_odds]) & (df["model_odds"] <= df[lay_odds]),
                        "P",
                       np.where(
                            df["model_odds"] < df[back_odds],
                            "B",
                            "L"
                        )
                       )
    
    df['stake'] = np.where(df['bet_side'] == "P", 0, stake)
    
    df['gpl'] = np.where(df['bet_side'] == "B", 
                         np.where(df['win'] == 1, df['stake'] * (df[back_odds]-1), -df['stake']), # PL for back bets
                         np.where(df['win'] == 1, -df['stake'] * (df[lay_odds]-1), df['stake']) # PL for lay bets
                        )   
    
    # Apply commission and NPL
    df = bet_apply_commission(df)
    
    return(df)

def bet_kelly(df, stake = 1, back_odds = 'market_odds', lay_odds = 'market_odds'):
    
    """
    Betting DF should always contain: model_odds, and win (binary encoded), and the specified odds column columns
    """
    
    df['bet_side'] = np.where((df["model_odds"] >= df[back_odds]) & (df["model_odds"] <= df[lay_odds]),
                            "P",
                            np.where(
                                df["model_odds"] < df[back_odds],
                                "B",
                                "L"
                            )
                       )
    
    df['stake'] = np.where(df['bet_side'] == "P",
                           0,
                           np.where(
                             df['bet_side'] == "B",
                             ( (1 / df['model_odds']) - (1 / df[back_odds]) ) / (1 - (1 / df[back_odds])),
                             ( (1 / df[lay_odds]) - (1 / df['model_odds']) ) / (1 - (1 / df[lay_odds])),
                           )
                          )
    
    df['gpl'] = np.where(df['bet_side'] == "B", 
                         np.where(df['win'] == 1, df['stake'] * (df[back_odds]-1), -df['stake']), # PL for back bets
                         np.where(df['win'] == 1, -df['stake'] * (df[lay_odds]-1), df['stake']) # PL for lay bets
                        )
    
    # Apply commission and NPL
    df = bet_apply_commission(df)
    
    return(df)


# In[122]:


# Testing one of these functions
flat_bets_bsp = bet_flat(df, stake = 1, back_odds = 'bsp', lay_odds = 'bsp')
flat_bets_bsp.head(5)


# ## 3.3 Evalutaion Functions
# 
# In my experience it's great to develop a suite of functions and analytical tools that really dig into every aspect of your simulated betting performance. You want to be as thorough and critical as possible, even when you're results are good.
# 
# Another tip for this process is to have a reasonable benchmark. Essentially no one wins at 10% POT on thoroughbreds at the BSP so if your analysis suggests you can... there's a bug. Similar logic can be applied to <-10% returns. Ruling out unreasonable results can save you a lot of time and delusion.
# 
# I'm keeping it pretty simple here but you might also want to create functions to analyse:
# * Track / distance based performance
# * Performance across odds ranges
# * Profit volatility (maybe using sharpe ratio to optimise volatility adjusted profit)
# * Date ranges (weeks / months etc)

# In[123]:


# Create simple PL and POT table
def bet_eval_metrics(d, side = False):
    
    if side:
        metrics = (d
         .groupby('bet_side', as_index=False)
         .agg({"npl": "sum", "stake": "sum"})
         .assign(pot=lambda x: x['npl'] / x['stake'])
        )
    else:
        metrics = pd.DataFrame(d
         .agg({"npl": "sum", "stake": "sum"})
        ).transpose().assign(pot=lambda x: x['npl'] / x['stake'])
    
    return(metrics[metrics['stake'] != 0])

# Cumulative PL by market to visually see trend and consistency
def bet_eval_chart_cPl(d):
    
    d = (
        d
        .groupby('market_id')
        .agg({'npl': 'sum'})
    )
    
    d['market_number'] = np.arange(len(d))
    d['cNpl'] = d.npl.cumsum()
    
    chart = px.line(d, x="market_number", y="cNpl", title='Cumulative Net Profit', template='simple_white')

    return(chart)


# To illustrate these evalatuation functions let's analyse flat staking at the BSP.

# In[124]:


import plotly.express as px

bets = bet_flat(df, stake = 1, back_odds = 'bsp', lay_odds = 'bsp')

bet_eval_metrics(bets, side = True)


# In[125]:


bet_eval_chart_cPl(bets)


# So this isn't gonna build us an art gallery! This is to be expected though, it's not easy to make consistent profit certainly from free ratings sources available online.

# ## 3.4 Testing different approaches
# 
# We pulled those extra price points for a reason. Let's set up a little test harness that enables us to use different price points and bet using different staking functions.

# In[126]:


# We'll test a 2 different staking schemes on 3 different price points
grid = {
        "flat_bsp": (bet_flat, "bsp", "bsp"),
        "flat_ltp": (bet_flat, "ltp", "ltp"),
        "flat_3m": (bet_flat, "best_back_3m", "best_lay_3m"),
        "kelly_bsp": (bet_kelly, "bsp", "bsp"),
        "kelly_ltp": (bet_kelly, "ltp", "ltp"),
        "kelly_3m": (bet_kelly, "best_back_3m", "best_lay_3m")
    }


# In[127]:


# Evaluate Metrics For Strategy Grid
metricSummary = None
for strategy, objects in grid.items():

    # Assemble bets based on staking function and odds column
    # objects[0] is the staking function itself
    bets = objects[0](df, back_odds = objects[1], lay_odds = objects[2])
    
    betMetrics = (
        bet_eval_metrics(bets)
        .assign(strategy=lambda x: strategy)
        .reindex(columns = ['strategy', 'stake', 'npl', 'pot'])
    )
    
    try:
        metricSummary = pd.concat([metricSummary, betMetrics], ignore_index=True)
    except:
        metricSummary = betMetrics
        
metricSummary.sort_values(by=['pot'], ascending=False)


# In[128]:


# Compare Cumulative PL Charts
cumulativePLs = None
for strategy, objects in grid.items():

    # Assemble bets based on staking function and odds column
    bets = objects[0](df, back_odds = objects[1], lay_odds = objects[2])
    
    d = (
           bets
           .groupby('market_id')
           .agg({'npl': 'sum', 'stake': 'sum'})
    )

    d['market_number'] = np.arange(len(d))
    # Normalise to $10,000 stake for visual comparison
    d['npl'] = d['npl'] / (d.stake.sum() / 10000)
    d['cNpl'] = d.npl.cumsum()
    d['strategy'] = strategy
    
    try:
        cumulativePLs = pd.concat([cumulativePLs, d], ignore_index=True)
    except:
        cumulativePLs = d

px.line(cumulativePLs, x="market_number", y="cNpl", color="strategy", title='Cumulative Net Profit', template='simple_white')


# ## 3.4 Searching For Profit
# 
# So this is often where you're going to arrive developing many wagering models: there's no indication of reliable long term profit. Where do you go from here?
# 
# Aside from packing it in there's 3 main options: 
# * Make the underlying model better
# * Search for better prices via detailed price analysis and clever bet placement
# * Try to find a subset of these selections with these ratings and these price points that are sustainably profitable
# 
# Obviously each situation is different but I think option 3 isn't a bad way to go initially because it will definitely help you understand your model better. For a racing model you might want to split your performance by:
# * tracks or states
# * track conditions or weather
# * barriers
# * race quality or grade
# * odds ranges
# * selection sample size (you likely perform worse on horses with little form for eg)
# * percieved model value
# 
# Finding a big enough slice across those dimensions thats either really good or really bad might reveal to you a bug in the data or workflow in your model development that you can go back and fix.
# 
# As an example of a simple approach to selectiveness I'll quickly run through how being more selective about your perceived value might make a difference in final profitability.

# So our best performing strategy using our simple analysis above was kelly staking at the last traded price. We'll start with that but be aware of that there's no way of implenting a LTP bet placement engine, you could imagine a proxy being placing limit bets "just before" the race jumps which is a whole other kettle of fish. 
# 
# Anyway, let's plot our profitablity under this strategy at different percieved "edges". If we are more selective of only large overlays according to the hub's rated chance you can see we can increase the profitability.

# In[129]:


bets = bet_kelly(df, back_odds = 'ltp', lay_odds = 'ltp')

metricSummary = None

for bVal in [0.05, 0.1, 0.15, 0.2, 0.3]:
    for lVal in [0.05, 0.1, 0.15, 0.2, 0.3]:
        x = bets.query('((ltp-model_odds) / ltp) > {}  | ((model_odds-ltp) / ltp) > {}'.format(bVal, lVal))
        
        betMetrics = bet_eval_metrics(x, side = False)
        betMetrics['bVal'] = bVal
        betMetrics['lVal'] = lVal
        
        try:
            metricSummary = pd.concat([metricSummary, betMetrics], ignore_index=True)
        except:
            metricSummary = betMetrics
            
metricSummary.sort_values(by=['pot'], ascending=False).head(4)


# In[130]:


betsFilters = bets.query('((ltp-model_odds) / ltp) > {}  | ((model_odds-ltp) / ltp) > {}'.format(0.3, 0.3))
bet_eval_chart_cPl(betsFilters)


# We were doing ok till the last 200 market nightmare! Might be one to test with more data.
# 
# * So we still haven't found a clear profitable edge with these ratings, however we got a bit closer to break even which is positive.
# * This step also indicates that this rating system performs better for large overlays which is a good model indicator (if you can't improve by selecting for larger overlays it's usually a sign you need to go back to the drawing board)
# * You could imagine a few more iterations of analysis you might be able to eek out a slight edge
# * However, be wary as these steps optimisation steps are very prone to overfitting so you need to be careful.

# # 4.0 Conclusion and Next Steps
# 
# 

# While using someone elses model is easy it's also not likely to end in riches for you. Developing your own model with your own tools and on a sport or racing code you know about is probably where you should start. However, hopefully this short guide helps you think about what to do when you finish the modelling exercise:
# 
# > How much money will I win or lose if I started using this system to place bets with real money?
# 
# If you want to expand your backtesting analysis, here's a list (in no particular order) of things that I've omitted or angles I might look at next:
# * Get more data
#     + more rating data and odds data is needed for draw a good conlusion about long term expectation
# * Cross reference performance against race or selection metadata (track, # races run etc.) to improve performance with betting selectivity
# * Extract more price points from the stream data to try to gain an pricing edge on these ratings
# 

# In[ ]:




