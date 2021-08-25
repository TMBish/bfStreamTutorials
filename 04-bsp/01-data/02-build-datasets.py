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
import csv

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

def pull_ladder(availableLadder, n = 5):
        out = {}
        price = []
        volume = []
        if len(availableLadder) == 0:
            return(out)        
        else:
            for rung in availableLadder[0:n]:
                price.append(rung.price)
                volume.append(rung.size)

            out["p"] = price
            out["v"] = volume
            return(out)

def filter_market(market: MarketBook) -> bool: 
    
    d = market.market_definition

    return (d.country_code == 'AU' 
        and d.market_type == 'WIN' 
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace')


# Credentials
# ________________________________

with open("../../secrets.yaml", 'r') as stream:
    creds = yaml.safe_load(stream)

trading = betfairlightweight.APIClient(creds['uid'], creds['pwd'],  app_key=creds["api_key"])

listener = StreamListener(max_latency=None)

# Market Metadata
# ________________________________

def final_market_book(s):
    with patch("builtins.open", lambda f, _: f):
        gen = s.get_generator()
        for market_books in gen():
            # Check if this market book meets our market filter ++++++++++++++++++++++++++++++++++
            if ((evaluate_market := filter_market(market_books[0])) == False):
                    return(None)
            for market_book in market_books:
                last_market_book = market_book
        return(last_market_book)

def parse_final_selection_meta(dir, out_file):
    
    with open(out_file, "w+") as output:

        output.write("market_id,selection_id,venue,market_time,selection_name,win,bsp\n")

        for file_obj in load_markets(dir):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            last_market_book = final_market_book(stream)
            if last_market_book is None:
                continue 

            # Extract Info ++++++++++++++++++++++++++++++++++
            runnerMeta = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in last_market_book.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'win': np.where(r.status == "WINNER", 1, 0),
                    'sp': r.sp.actual_sp
                }
                for r in last_market_book.runners 
            ]

            # Return Info ++++++++++++++++++++++++++++++++++
            for runnerMeta in runnerMeta:
                if runnerMeta['selection_status'] != 'REMOVED':
                    output.write(
                        "{},{},{},{},{},{},{}\n".format(
                            str(last_market_book.market_id),
                            runnerMeta['selection_id'],
                            last_market_book.market_definition.venue,
                            last_market_book.market_definition.market_time,
                            runnerMeta['selection_name'],
                            runnerMeta['win'],
                            runnerMeta['sp']
                        )
                    )

# Preplay Prices
# ________________________________

def loop_preplay_prices(s, o):

    with patch("builtins.open", lambda f, _: f):

        gen = s.get_generator()

        marketID = None
        tradeVols = None
        time = None
        last_book_recorded = False
        prev_book = None

        for market_books in gen():

            # Check if this market book meets our market filter ++++++++++++++++++++++++++++++++++

            if ((evaluate_market := filter_market(market_books[0])) == False):
                    break
            
            for market_book in market_books:

                # Time Step Management ++++++++++++++++++++++++++++++++++

                if marketID is None:
                    # No market initialised
                    marketID = market_book.market_id
                    time =  market_book.publish_time
                elif market_book.inplay and last_book_recorded:
                    break
                else:
                                            
                    seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()

                    if seconds_to_start > 120:
                        # Too early before off to start logging prices
                        prev_book = market_book
                        continue
                    else:
                        
                        # Update data at different time steps depending on seconds to off
                        wait = 10

                        # New Market
                        if market_book.market_id != marketID:
                            last_book_recorded = False
                            marketID = market_book.market_id
                            time =  market_book.publish_time
                            continue
                        # (wait) seconds elapsed since last write
                        elif (market_book.publish_time - time).total_seconds() > wait:
                            time = market_book.publish_time
                        # if current marketbook is inplay want to record the previous market book as it's the last preplay marketbook
                        elif market_book.inplay:
                            last_book_recorded = True
                            market_book = prev_book
                        # fewer than (wait) seconds elapsed continue to next loop
                        else:
                            prev_book = market_book
                            continue

                # Execute Data Logging ++++++++++++++++++++++++++++++++++
                for runner in market_book.runners:

                    try:
                        atb_ladder = pull_ladder(runner.ex.available_to_back, n = 5)
                        atl_ladder = pull_ladder(runner.ex.available_to_lay, n = 5)
                    except:
                        atb_ladder = {}
                        atl_ladder = {}

                    o.writerow(
                        (
                            market_book.market_id,
                            runner.selection_id,
                            market_book.publish_time,
                            # SP Fields
                            runner.sp.near_price,
                            runner.sp.far_price,
                            int(sum([ps.size for ps in runner.sp.back_stake_taken])),
                            int(sum([ps.size for ps in runner.sp.lay_liability_taken])),
                            # Limit bets available
                            str(atb_ladder).replace(' ',''), 
                            str(atl_ladder).replace(' ','')
                        )
                    )

                prev_book = market_book

def parse_preplay_prices(dir, out_file):
    
    with open(out_file, "w+") as output:

        writer = csv.writer(
            output, 
            delimiter=',',
            lineterminator='\r\n',
            quoting=csv.QUOTE_ALL
        )
        writer.writerow(("market_id","selection_id","time","near_price","far_price","bsp_back_pool_stake","bsp_lay_pool_liability","atb_ladder",'atl_ladder'))

        for file_obj in load_markets(dir):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            loop_preplay_prices(stream, writer)


# Executing
# ________________________________

# Output files +++++++++++++++++++++++++
metaFile = "/media/hdd/tmp/bsp/meta.csv"
priceFile = "/media/hdd/tmp/bsp/prices.csv"

# Input files  +++++++++++++++++++++++++
stream_files = [
    "/media/hdd/data/betfair-stream/thoroughbred/2021_06_JunRacingAUPro.tar",
    "/media/hdd/data/betfair-stream/thoroughbred/2021_05_MayRacingAUPro.tar",
    "/media/hdd/data/betfair-stream/thoroughbred/2021_04_AprRacingAUPro.tar"
]

# Execute Meta Parse  +++++++++++++++++++++++++
if __name__ == '__main__':
    print("__ Parsing Selection Meta ___ ")
    # parse_final_selection_meta(stream_files, metaFile)

# Execute Price Parse  +++++++++++++++++++++++++
if __name__ == '__main__':
    print("__ Parsing Selection Prices ___ ")
    parse_preplay_prices(stream_files, priceFile)