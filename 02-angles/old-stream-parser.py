import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import os
import re
import tarfile
import zipfile
import bz2
import glob
import logging
from unittest.mock import patch
from typing import List, Set, Dict, Tuple, Optional
from itertools import zip_longest
import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)

# FUNCTIONS
#______________________________________________

# rounding to 2 decimal places or returning '' if blank
def as_str(v) -> str:
    return '%.2f' % v if type(v) is float else v if type(v) is str else ''

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
                    
      

    

