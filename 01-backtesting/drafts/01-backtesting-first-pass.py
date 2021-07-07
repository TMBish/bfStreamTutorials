#!/usr/bin/env python
# coding: utf-8

# # Backtesting Ratings With Historical Stream Data
# 
# Betfair's historical stream data is an extremely rich source of data. However, in it's raw form it's difficult to handle for the uninitiated. It also might not be immediately obvious how many different things this dataset could be used for without seeing some examples. These guides will hopefully demystify how to turn this raw data into a familiar and usable format whilst also hopefully providing some inpiration for the kinds of gold that can be excavated from it.

# In[1]:


import pandas as pd
import os
import re
import betfairlightweight
from betfairlightweight import StreamListener
import logging
import requests
import tarfile
import bz2
from unittest.mock import patch

import logging
from typing import List, Set, Dict, Tuple, Optional

from unittest.mock import patch
from itertools import zip_longest
import functools

import os
import tarfile
import zipfile
import bz2
import glob

# importing data types
import betfairlightweight
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)

# setup logging
# logging.basicConfig(level=logging.INFO)


# In[2]:


data_path = [
    "./data/2021_01_JanRacingPro.tar",
    "./data/2021_01_JanRacingPro.tar"
]

stream_directory = "./data/"


# In[12]:


# KGs tutorial
# https://betfair-datascientists.github.io/historicData/csvTutorialMarketSummary/index.html


# In[3]:


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


# In[4]:


# Betfair Lightweight Boilerplate

# create trading instance (don't need username/password)
trading = betfairlightweight.APIClient("username", "password")

# create listener
listener = StreamListener(max_latency=None)


# In[5]:


# rounding to 2 decimal places or returning '' if blank
def as_str(v: float) -> str:
    return '%.2f' % v if v is not None else ''

# parsing price data and pulling out weighted avg price, matched, min price and max price
def parse_traded(traded: List[PriceSize]) -> (float, float, float, float):
    if len(traded) == 0: 
        return (None, None, None, None)

    (wavg_sum, matched, min_price, max_price) = functools.reduce(
        lambda total, ps: (
            total[0] + (ps.price * ps.size),
            total[1] + ps.size,
            min(total[2], ps.price),
            max(total[3], ps.price),
        ),
        traded,
        (0, 0, 1001, 0)
    )

    wavg_sum = (wavg_sum / matched) if matched > 0 else None
    matched = matched if matched > 0 else None
    min_price = min_price if min_price != 1001 else None
    max_price = max_price if max_price != 0 else None

    return (wavg_sum, matched, min_price, max_price)

# splitting race name and returning the parts 
def split_anz_horse_market_name(market_name: str) -> (str, str, str):
    # return race no, length, race type
    # input sample: R6 1400m Grp1
    parts = market_name.split(' ')
    race_no = parts[0] # return example R6
    race_len = parts[1] # return example 1400m
    race_type = parts[2].lower() # return example grp1, trot, pace

    return (race_no, race_len, race_type)

# filtering markets to those that fit the following criteria
def filter_market(market: MarketBook) -> bool: 
    d = market.market_definition
    return (d.country_code == 'AU' 
        and d.market_type == 'WIN' 
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace')


# # Executing KGs Tutorial Version
# 
# We'll run the exact code KG wrote for her tutorial as a starting point to prove that this works

# In[ ]:


# record prices to a file
with open("outputs/kg-example.csv", "w") as output:
    
    # defining column headers
    output.write("market_id,event_date,country,track,market_name,selection_id,selection_name,result,bsp,pp_min,pp_max,pp_wap,pp_ltp,pp_volume,ip_min,ip_max,ip_wap,ip_ltp,ip_volume\n")

    for file_obj in load_markets(data_path):

        # Instantiate a "stream" object
        stream = trading.streaming.create_historical_generator_stream(
            file_path=file_obj,
            listener=listener,
        )


        # For this stream object execute the following Lambda function
        with patch("builtins.open", lambda f, _: f): 

            evaluate_market = False
            preplay_market = None
            postplay_market = None
            preplay_traded = None
            postplay_traded = None   

            gen = stream.get_generator()
            for market_books in gen():
                for market_book in market_books:

                    # skipping markets that don't meet the filter
                    if evaluate_market == False and filter_market(market_book) == False:
                        continue
                    else:
                        evaluate_market = True

                    # final market view before market goes in play
                    if preplay_market is not None and preplay_market.inplay != market_book.inplay:
                        preplay_traded = [ (r.last_price_traded, r.ex.traded_volume.copy()) for r in preplay_market.runners ]
                        print(preplay_traded)
                    preplay_market = market_book

                    # final market view at the conclusion of the market
                    if postplay_market is not None and postplay_market.status == "OPEN" and market_book.status != postplay_market.status:
                        postplay_traded = [ (r.last_price_traded, r.ex.traded_volume.copy()) for r in market_book.runners ]
                    postplay_market = market_book   

            # no price data for market
            if postplay_traded is None:
                print('didnt find postplay results?')
                continue; 

            # generic runner data
            runner_data = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in postplay_market.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'sp': as_str(r.sp.actual_sp),
                }
                for r in postplay_market.runners 
            ]

            # runner price data for markets that go in play
            if preplay_traded is not None:
                def runner_vals(r):
                    (pre_ltp, pre_traded), (post_ltp, post_traded) = r

                    inplay_only = list(filter(lambda ps: ps.size > 0, [
                        PriceSize(
                            price=post_ps.price, 
                            size=post_ps.size - next((pre_ps.size for pre_ps in pre_traded if pre_ps.price == post_ps.price), 0)
                        )
                        for post_ps in post_traded 
                    ]))

                    (ip_wavg, ip_matched, ip_min, ip_max) = parse_traded(inplay_only)
                    (pre_wavg, pre_matched, pre_min, pre_max) = parse_traded(pre_traded)

                    return {
                        'preplay_ltp': as_str(pre_ltp),
                        'preplay_min': as_str(pre_min),
                        'preplay_max': as_str(pre_max),
                        'preplay_wavg': as_str(pre_wavg),
                        'preplay_matched': as_str(pre_matched),
                        'inplay_ltp': as_str(post_ltp),
                        'inplay_min': as_str(ip_min),
                        'inplay_max': as_str(ip_max),
                        'inplay_wavg': as_str(ip_wavg),
                        'inplay_matched': as_str(ip_matched),
                    }

                runner_traded = [ runner_vals(r) for r in zip_longest(preplay_traded, postplay_traded, fillvalue=PriceSize(0, 0)) ]

            # runner price data for markets that don't go in play
            else:
                def runner_vals(r):
                    (ltp, traded) = r
                    (wavg, matched, min_price, max_price) = parse_traded(traded)

                    return {
                        'preplay_ltp': as_str(ltp),
                        'preplay_min': as_str(min_price),
                        'preplay_max': as_str(max_price),
                        'preplay_wavg': as_str(wavg),
                        'preplay_matched': as_str(matched),
                        'inplay_ltp': '',
                        'inplay_min': '',
                        'inplay_max': '',
                        'inplay_wavg': '',
                        'inplay_matched': '',
                    }

                runner_traded = [ runner_vals(r) for r in postplay_traded ]

            # printing to csv for each runner
            for (rdata, rprices) in zip(runner_data, runner_traded):
                # defining data to go in each column
                output.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        postplay_market.market_id,
                        postplay_market.market_definition.market_time,
                        postplay_market.market_definition.country_code,
                        postplay_market.market_definition.venue,
                        postplay_market.market_definition.name,
                        rdata['selection_id'],
                        rdata['selection_name'],
                        rdata['selection_status'],
                        rdata['sp'],
                        rprices['preplay_min'],
                        rprices['preplay_max'],
                        rprices['preplay_wavg'],
                        rprices['preplay_ltp'],
                        rprices['preplay_matched'],
                        rprices['inplay_min'],
                        rprices['inplay_max'],
                        rprices['inplay_wavg'],
                        rprices['inplay_ltp'],
                        rprices['inplay_matched'],
                    )
                )


# # Stripped Back Version
# 
# What if I just want to pull out the BSP and the runner metadata

# In[ ]:


# record prices to a file
with open("outputs/example-1.csv", "w") as output:
    # defining column headers\
    
    # Column Headers
    #output.write("m")

    for file_obj in load_markets(data_path):

        # Instantiate a "stream" object
        stream = trading.streaming.create_historical_generator_stream(
            file_path=file_obj,
            listener=listener,
        )


        # For this stream object execute the following Lambda function
        with patch("builtins.open", lambda f, _: f): 

            evaluate_market = False
            preplay_market = None
            postplay_market = None
            preplay_traded = None
            postplay_traded = None   

            gen = stream.get_generator()
            for market_books in gen():
                for market_book in market_books:

                    # skipping markets that don't meet the filter
                    if evaluate_market == False and filter_market(market_book) == False:
                        continue
                    else:
                        evaluate_market = True

                    # final market view before market goes in play
                    if preplay_market is not None and preplay_market.inplay != market_book.inplay:
                        preplay_traded = [ (r.last_price_traded, r.ex.traded_volume.copy()) for r in preplay_market.runners ]
                    preplay_market = market_book

                    # final market view at the conclusion of the market
                    if postplay_market is not None and postplay_market.status == "OPEN" and market_book.status != postplay_market.status:
                        postplay_traded = [ (r.last_price_traded, r.ex.traded_volume.copy()) for r in market_book.runners ]
                    postplay_market = market_book   

            # no price data for market
            if postplay_traded is None:
                print('didnt find postplay results?')
                continue; 

            # generic runner data
            runner_data = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in postplay_market.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'sp': as_str(r.sp.actual_sp),
                }
                for r in postplay_market.runners 
            ]

            # runner price data for markets that go in play
            if preplay_traded is not None:
                def runner_vals(r):
                    (pre_ltp, pre_traded), (post_ltp, post_traded) = r

                    inplay_only = list(filter(lambda ps: ps.size > 0, [
                        PriceSize(
                            price=post_ps.price, 
                            size=post_ps.size - next((pre_ps.size for pre_ps in pre_traded if pre_ps.price == post_ps.price), 0)
                        )
                        for post_ps in post_traded 
                    ]))

                    (ip_wavg, ip_matched, ip_min, ip_max) = parse_traded(inplay_only)
                    (pre_wavg, pre_matched, pre_min, pre_max) = parse_traded(pre_traded)

                    return {
                        'preplay_ltp': as_str(pre_ltp),
                        'preplay_min': as_str(pre_min),
                        'preplay_max': as_str(pre_max),
                        'preplay_wavg': as_str(pre_wavg),
                        'preplay_matched': as_str(pre_matched),
                        'inplay_ltp': as_str(post_ltp),
                        'inplay_min': as_str(ip_min),
                        'inplay_max': as_str(ip_max),
                        'inplay_wavg': as_str(ip_wavg),
                        'inplay_matched': as_str(ip_matched),
                    }

                runner_traded = [ runner_vals(r) for r in zip_longest(preplay_traded, postplay_traded, fillvalue=PriceSize(0, 0)) ]

            # runner price data for markets that don't go in play
            else:
                def runner_vals(r):
                    (ltp, traded) = r
                    (wavg, matched, min_price, max_price) = parse_traded(traded)

                    return {
                        'preplay_ltp': as_str(ltp),
                        'preplay_min': as_str(min_price),
                        'preplay_max': as_str(max_price),
                        'preplay_wavg': as_str(wavg),
                        'preplay_matched': as_str(matched),
                        'inplay_ltp': '',
                        'inplay_min': '',
                        'inplay_max': '',
                        'inplay_wavg': '',
                        'inplay_matched': '',
                    }

                runner_traded = [ runner_vals(r) for r in postplay_traded ]

            # printing to csv for each runner
            for (rdata, rprices) in zip(runner_data, runner_traded):
                # defining data to go in each column
                output.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        postplay_market.market_id,
                        postplay_market.market_definition.market_time,
                        postplay_market.market_definition.country_code,
                        postplay_market.market_definition.venue,
                        postplay_market.market_definition.name,
                        rdata['selection_id'],
                        rdata['selection_name'],
                        rdata['selection_status'],
                        rdata['sp'],
                        rprices['preplay_min'],
                        rprices['preplay_max'],
                        rprices['preplay_wavg'],
                        rprices['preplay_ltp'],
                        rprices['preplay_matched'],
                        rprices['inplay_min'],
                        rprices['inplay_max'],
                        rprices['inplay_wavg'],
                        rprices['inplay_ltp'],
                        rprices['inplay_matched'],
                    )
                )

