import pandas as pd
import numpy as np
import requests
import os
import re
import tarfile
import zipfile
import bz2
import glob
import logging
import yaml

from datetime import date, timedelta
from unittest.mock import patch
from typing import List, Set, Dict, Tuple, Optional
from itertools import zip_longest
import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)

# Utility Functions
# _________________________________

def as_str(v) -> str:
    return '%.2f' % v if type(v) is float else v if type(v) is str else ''

def split_anz_horse_market_name(market_name: str) -> (str, str, str):
    parts = market_name.split(' ')
    race_no = parts[0] # return example R6
    race_len = parts[1] # return example 1400m
    race_type = parts[2].lower() # return example grp1, trot, pace
    return (race_no, race_len, race_type)

def filter_market(market: MarketBook) -> bool: 
    d = market.market_definition
    return (d.country_code == 'AU' 
        and d.market_type == 'WIN' 
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace')

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

def slicePrice(l, n):
    try:
        x = l[n].price
    except:
        x = np.nan
    return(x)

def sliceSize(l, n):
    try:
        x = l[n].size
    except:
        x = np.nan
    return(x)

def wapPrice(l, n):
    try:
        x = round(sum( [rung.price * rung.size for rung in l[0:(n-1)] ] ) / sum( [rung.size for rung in l[0:(n-1)] ]),2)
    except:
        x = np.nan
    return(x)

def ladder_traded_volume(ladder):
    return(sum([rung.size for rung in ladder]))

# Parse Functions
# _________________________________

def extract_components_from_stream(s):
    
    with patch("builtins.open", lambda f, _: f):   
    
        # Will return 3 market books t-3mins marketbook, the last preplay marketbook and the final market book
        evaluate_market = None
        prev_market = None
        postplay = None
        preplay = None
        t5m = None
        t30s = None
        inplay_min_lay = None

        gen = s.get_generator()

        for market_books in gen():
            
            for market_book in market_books:

                # If markets don't meet filter return None's
                if evaluate_market is None and ((evaluate_market := filter_market(market_book)) == False):
                    return (None, None, None, None, None, None)

                # final market view before market goes in play
                if prev_market is not None and prev_market.inplay != market_book.inplay:
                    preplay = market_book

                # final market view before market goes is closed for settlement
                if prev_market is not None and prev_market.status == "OPEN" and market_book.status != prev_market.status:
                    postplay = market_book

                # Calculate Seconds Till Scheduled Market Start Time
                seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()
                    
                # Market at 30 seconds before scheduled off
                if t30s is None and seconds_to_start < 30:
                    t30s = market_book

                # Market at 30 seconds before scheduled off
                if t5m is None and seconds_to_start < 5*60:
                    t5m = market_book

                # Manage Inplay Vectors
                if market_book.inplay:

                    if inplay_min_lay is None:
                        inplay_min_lay = [ slicePrice(runner.ex.available_to_lay,0) for runner in market_book.runners]
                    else:
                        inplay_min_lay = np.fmin(inplay_min_lay, [ slicePrice(runner.ex.available_to_lay,0) for runner in market_book.runners])

                # update reference to previous market
                prev_market = market_book

        # If market didn't go inplay
        if postplay is not None and preplay is None:
            preplay = postplay
            inplay_min_lay = ["" for runner in market_book.runners]

        return (t5m, t30s, preplay, postplay, inplay_min_lay, prev_market) # Final market is last prev_market

def parse_stream(stream_files, output_file):
    
    with open(output_file, "w+") as output:

        output.write("market_id,selection_id,selection_name,wap_5m,wap_30s,bsp,ltp,traded_vol,inplay_min_lay\n")

        for file_obj in load_markets(stream_files):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            (t5m, t30s, preplay, postplay, inplayMin, final) = extract_components_from_stream(stream)

            # If no price data for market don't write to file
            if postplay is None or final is None or t30s is None:
                continue; 

            # All runner removed
            if all(runner.status == "REMOVED" for runner in final.runners):
                continue

            runnerMeta = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in final.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'sp': r.sp.actual_sp
                }
                for r in final.runners 
            ]

            ltp = [runner.last_price_traded for runner in preplay.runners]

            tradedVol = [ ladder_traded_volume(runner.ex.traded_volume) for runner in postplay.runners ]

            wapBack30s = [ wapPrice(runner.ex.available_to_back, 3) for runner in t30s.runners]

            wapBack5m = [ wapPrice(runner.ex.available_to_back, 3) for runner in t5m.runners]

            # Writing To CSV
            # ______________________

            for (runnerMeta, ltp, tradedVol, inplayMin, wapBack5m, wapBack30s) in zip(runnerMeta, ltp, tradedVol, inplayMin, wapBack5m, wapBack30s):
                
                if runnerMeta['selection_status'] != 'REMOVED':

                    output.write(
                        "{},{},{},{},{},{},{},{},{}\n".format(
                            str(final.market_id),
                            runnerMeta['selection_id'],
                            runnerMeta['selection_name'],
                            wapBack5m,
                            wapBack30s,
                            runnerMeta['sp'],
                            ltp,
                            round(tradedVol),
                            inplayMin
                        )
                    )

# Parameters
# _________________________________

with open("secrets.yaml", 'r') as stream:
    creds = yaml.safe_load(stream)

trading = betfairlightweight.APIClient(creds['uid'], creds['pwd'],  app_key=creds["api_key"])

listener = StreamListener(max_latency=None)

stream_files = glob.glob("/media/hdd/data/betfair-stream/thoroughbred/*.tar")
#stream_files = ["/media/hdd/data/betfair-stream/thoroughbred/2021_02_FebRacingAUPro.tar"]
output_file = "/media/hdd/tmp/thoroughbred-parsed/thoroughbred-odds-2021.csv"

# Run
# _________________________________

if __name__ == '__main__':
    parse_stream(stream_files, output_file)
