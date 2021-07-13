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
        x = ""
    return(x)

def sliceSize(l, n):
    try:
        x = l[n].size
    except:
        x = ""
    return(x)

# Parse Functions
# _________________________________

def loop_stream_markets(s, o):

    with patch("builtins.open", lambda f, _: f):

        gen = s.get_generator()

        marketID = None
        time = None

        for market_books in gen():

            for market_book in market_books:
                
                # Only Evaluate Thoroughbred Races
                # ________________________________

                if ((evaluate_market := filter_market(market_book)) == False):
                    break

                # Time Step Management
                # _____________________

                if marketID is None:
                    # print(1)
                    marketID = market_book.market_id
                    time =  market_book.publish_time
                else:
                    
                    # Update data at different time steps depending on inplay vs preplay
                    wait = np.where(market_book.inplay, inPlayTimeStep, prePlayTimeStep)

                    # New Market
                    if market_book.market_id != marketID:
                        marketID = market_book.market_id
                        time =  market_book.publish_time
                    # (wait) seconds elapsed since last write
                    elif (market_book.publish_time - time).total_seconds() > wait:
                        time = market_book.publish_time
                    # fewer than (wait) seconds elapsed continue to next loop
                    else:
                        continue

                                
                for runner in market_book.runners:

                    o.write(
                        "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                            market_book.market_id,
                            runner.selection_id,
                            market_book.publish_time,
                            market_book.status,
                            market_book.inplay,
                            sum([rung.size for rung in runner.ex.traded_volume]),
                            runner.last_price_traded or "",
                            slicePrice(runner.ex.available_to_back, 0),
                            slicePrice(runner.ex.available_to_lay, 0),
                            sliceSize(runner.ex.available_to_back, 0),
                            sliceSize(runner.ex.available_to_lay, 0)
                        )
                    )

def parse_stream(stream_files, output_file):
    
    with open(output_file, "w+") as output:

        output.write("market_id,selection_id,time,market_status,inplay_status,traded_volume,ltp,best_back,best_lay,best_back_volume,best_lay_volume\n")

        for file_obj in load_markets(stream_files):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            loop_stream_markets(stream, output)

# Parameters
# _________________________________

with open("../secrets.yaml", 'r') as stream:
    creds = yaml.safe_load(stream)

trading = betfairlightweight.APIClient(creds['uid'], creds['pwd'],  app_key=creds["api_key"])

listener = StreamListener(max_latency=None)

prePlayTimeStep = 10
inPlayTimeStep = 2

stream_files = glob.glob("/media/hdd/data/betfair-stream/thoroughbred/*.tar")
output_file = "/media/hdd/tmp/thoroughbred-parsed/thoroughbred-odds-2021.csv"

# Run
# _________________________________

if __name__ == '__main__':
    parse_stream(stream_files)
