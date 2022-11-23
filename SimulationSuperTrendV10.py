from colorama import Fore, Back, Style
from pprint import pprint

from time import sleep, time

from binance.client import Client
from binance.exceptions import BinanceAPIException

import pandas_ta as ta

import json
import math
import time

import telegram_send

import pandas as pd     # needs pip install
import numpy as np
import matplotlib.pyplot as plt   # needs pip install
from operator import add
from operator import sub

from mpl_finance import candlestick2_ohlc

def get_data_frame(client, crypto, StartTime, Interval):
    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    # request historical candle (or klines) data using timestamp from above, interval either every min, hr, day or month
    # starttime = '30 minutes ago UTC' for last 30 mins time
    # e.g. client.get_historical_klines(symbol='ETHUSDTUSDT', '1m', starttime)
    # starttime = '1 Dec, 2017', '1 Jan, 2018'  for last month of 2017
    # e.g. client.get_historical_klines(symbol='BTCUSDT', '1h', "1 Dec, 2017", "1 Jan, 2018")
    starttime = StartTime  # to start for 1 day ago
    interval = Interval
    bars = client_binance.futures_historical_klines(crypto, interval, starttime)
    #pprint.pprint(bars)
    
    for line in bars:        # Keep only first 5 columns, "date" "open" "high" "low" "close"
        del line[5:]
    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close']) #  2 dimensional tabular data
    #df.set_index('date', inplace=True)
    
    for i in df.columns:
        df[i] = df[i].astype(float)

    df['date'] = pd.to_datetime(df['date'], unit='ms') 
    
    return df

def order_block_finder(df, periods, threshold, usewicks):
    ob_period = periods + 1
    index = 0

    OB_bull = [None] * len(df['close'])
    OB_bull_high = [None] * len(df['close'])
    OB_bull_low = [None] * len(df['close'])
    OB_bull_avg = [None] * len(df['close'])

    OB_bear = [None] * len(df['close'])
    OB_bear_high = [None] * len(df['close'])
    OB_bear_low = [None] * len(df['close'])
    OB_bear_avg = [None] * len(df['close'])

    OB_bull_indic = [None] * len(df['close'])
    OB_bear_indic = [None] * len(df['close'])


    for i in range(len(df['close'])):
        
        if i > 5:
            absmove   = ((abs(df['close'][index-ob_period] - df['close'][index-1]))/df['close'][index-ob_period]) * 100    # Calculate absolute percent move from potential OB to last candle of subsequent candles
            relmove   = absmove >= threshold


            bullishOB = df['close'][index-ob_period] < df['open'][index-ob_period]
            upcandles  = 0
            for x in range(1, periods+1) :
                value = 0
                if df['close'][index-x] > df['open'][index-x] :
                    value = 1
                else:
                    value = 0

                upcandles = upcandles + value

            OB_bull[i]      = bullishOB and (upcandles == (periods)) and relmove          # Identification logic (red OB candle & subsequent green candles)
            #OB_bull_high[i] = OB_bull ? usewicks ? df['high'][index-ob_period] : df['open'][index-ob_period] : np.nan()   # Determine OB upper limit (Open or High depending on input)
            if OB_bull[i] == True:
                if usewicks == True:
                    OB_bull_high[i] = df['high'][index-ob_period]
                else:
                   OB_bull_high[i] = df['open'][index-ob_period]
            else:
                OB_bull_high[i] = np.nan
            #OB_bull_low[i]  = OB_bull ? df['low'][index-ob_period]  : np.nan()                               # Determine OB lower limit (Low)
            if OB_bull[i] == True:
                OB_bull_low[i] = df['low'][index-ob_period]
            else:
                OB_bull_low[i] = np.nan

            OB_bull_avg[i]  = (OB_bull_high[i] + OB_bull_low[i])/2


            bearishOB = df['close'][index-ob_period] > df['open'][index-ob_period]                             # Determine potential Bearish OB candle (green candle)

            downcandles  = 0
            for x in range(1, periods+1) :
                value = 0
                if df['close'][index-x] < df['open'][index-x]:
                    value = 1
                else:
                    value = 0

                downcandles = downcandles + value               # Determine color of subsequent candles (must all be red to identify a valid Bearish OB)

            OB_bear[i]      = bearishOB and (downcandles == (periods)) and relmove        # Identification logic (green OB candle & subsequent green candles)
            #OB_bear_high[i] = OB_bear ? df['high'][index-ob_period] : np.nan()                               # Determine OB upper limit (High)
            if OB_bear[i] == True:
                OB_bear_high[i] = df['high'][index-ob_period]
            else:
                OB_bear_high[i] = np.nan
            #OB_bear_low[i]  = OB_bear ? usewicks ? df['low'][index-ob_period] : df['open'][index-ob_period] : np.nan()    # Determine OB lower limit (Open or Low depending on input)
            if OB_bear[i] == True:
                if usewicks == True:
                    OB_bear_low[i] = df['low'][index-ob_period]
                else:
                    OB_bear_low[i] = df['open'][index-ob_period]
            else:
                OB_bear_low[i] = np.nan

            OB_bear_avg[i]  = (OB_bear_low[i] + OB_bear_high[i])/2                              # Determine OB middle line
            if OB_bull[i] == True:
                OB_bull_indic[i-5] = df['open'][index-5]
            else:
                OB_bull_indic[i] = np.nan

            if OB_bear[i] == True:
                OB_bear_indic[i-5] = df['open'][index-5]
            else:
                OB_bear_indic[i] = np.nan
        else:
            OB_bull_indic[i] = np.nan
            OB_bull[i] = np.nan
            OB_bull_high[i] = np.nan
            OB_bull_low[i] = np.nan
            OB_bull_avg[i] = np.nan

            OB_bear_indic[i] = np.nan
            OB_bear[i] = np.nan
            OB_bear_low[i] = np.nan
            OB_bear_high[i] = np.nan
            OB_bear_avg[i] = np.nan
        index += 1
            

    return {"bull_indic": OB_bull_indic, "bear_indic": OB_bear_indic,"bull":OB_bull, "bull_high":OB_bull_high, "bull_low":OB_bull_low, "bull_avg":OB_bull_avg, "bear":OB_bear, "bear_high":OB_bear_high, "bear_low":OB_bear_low, "bear_avg":OB_bear_avg}
            
            
PEAK, VALLEY = 1, -1
            
def _identify_initial_pivot(X, up_thresh, down_thresh):
    """Quickly identify the X[0] as a peak or valley."""
    x_0 = X[0]
    max_x = x_0
    max_t = 0
    min_x = x_0
    min_t = 0
    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X)-1
    return VALLEY if x_0 < X[t_n] else PEAK

def peak_valley_pivots_candlestick(close, high, low, up_thresh, down_thresh):
    """
    Finds the peaks and valleys of a series of HLC (open is not necessary).
    TR: This is modified peak_valley_pivots function in order to find peaks and valleys for OHLC.
    Parameters
    ----------
    close : This is series with closes prices.
    high : This is series with highs  prices.
    low : This is series with lows prices.
    up_thresh : The minimum relative change necessary to define a peak.
    down_thesh : The minimum relative change necessary to define a valley.
    Returns
    -------
    an array with 0 indicating no pivot and -1 and 1 indicating valley and peak
    respectively
    Using Pandas
    ------------
    For the most part, close, high and low may be a pandas series. However, the index must
    either be [0,n) or a DateTimeIndex. Why? This function does X[t] to access
    each element where t is in [0,n).
    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias analysis.
    """
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')

    initial_pivot = _identify_initial_pivot(close, up_thresh, down_thresh)

    t_n = len(close)
    pivots = np.zeros(t_n, dtype='i1')
    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    up_thresh += 1
    down_thresh += 1

    trend = -initial_pivot
    last_pivot_t = 0
    last_pivot_x = close[0]
    for t in range(1, len(close)):

        if trend == -1:
            x = low[t]
            r = x / last_pivot_x
            if r >= up_thresh:
                pivots[last_pivot_t] = trend#
                trend = 1
                #last_pivot_x = x
                last_pivot_x = high[t]
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            x = high[t]
            r = x / last_pivot_x
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = -1
                #last_pivot_x = x
                last_pivot_x = low[t]
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t


    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = trend

    return pivots

def trix(df, n):
    """Calculate TRIX for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    EX1 = df['close'].ewm(span=n, min_periods=n).mean()
    EX2 = EX1.ewm(span=n, min_periods=n).mean()
    EX3 = EX2.ewm(span=n, min_periods=n).mean()
    i = 0
    ROC_l = [np.nan]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name='Trix_' + str(n))
    df = df.join(Trix)
    return df 

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

def psar(barsdata, iaf = 0.02, maxaf = 0.2):
    length = len(barsdata)
    dates = list(barsdata['date'])
    high = list(barsdata['high'])
    low = list(barsdata['low'])
    close = list(barsdata['close'])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}
def get_supertrend(high, low, close, lookback, multiplier):
    
    # ATR
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(lookback).mean()
    
    # H/L AVG AND BASIC UPPER & LOWER BAND
    
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()
    
    # FINAL UPPER BAND
    
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:,1] = final_bands.iloc[:,0]
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]
    
    # FINAL LOWER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]
    
    # SUPERTREND
    
    supertrend = pd.DataFrame(columns = [f'supertrend_{lookback}'])
    supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]
    
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
    
    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]
    
    # ST UPTREND/DOWNTREND
    
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if i > 0:
            if close[i] > supertrend.iloc[i, 0]:
                upt.append(supertrend.iloc[i, 0])
                dt.append(np.nan)
            elif close[i] < supertrend.iloc[i, 0]:
                upt.append(np.nan)
                dt.append(supertrend.iloc[i, 0])
            else:
                upt.append(np.nan)
                dt.append(np.nan)
        else:
            upt.append(np.nan)
            dt.append(np.nan)
            
    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    
    return st, upt, dt

if __name__ == "__main__":

    

    

    client_binance = Client('mHTziOVrCssZt95EDryM26G4i0yB2PvOWpMqMOiBqzlDv6Y0UUXcrpJBjJZtObzo', 'nfx2PdnYg1MHpV7PVoRJFXRlF53nCiXDHqE8ybJQdU8x3PlzfTYRevimbx3rmMkx')
    DataCrypto1H = get_data_frame(client_binance, 'BTCUSDT', '4 weeks ago',  '1h') #WOOUSDT #UNFIUSDT #UNIUSDT #TOMOUSDT
    DataCrypto4H = get_data_frame(client_binance, 'BTCUSDT', '4 weeks ago',  '4h') #WOOUSDT #UNFIUSDT #UNIUSDT #TOMOUSDT


    #periods = 5
    #threshold = 0.0
    #usewicks = False



    
    
    #order_block = order_block_finder(src, periods, threshold, usewicks)
    #pivots = peak_valley_pivots_candlestick(src['close'], src['high'], src['low'] ,.02,-.02)
    st, upt, dt = get_supertrend(DataCrypto1H['high'], DataCrypto1H['low'], DataCrypto1H['close'], 7, 3)
    st2, upt2, dt2 = get_supertrend(DataCrypto4H['high'], DataCrypto4H['low'], DataCrypto4H['close'], 7, 3)

    psar_data = psar(DataCrypto1H)

    print(st)
    print(dt)

    # src['Pivots'] = pivots
    # print(pivots)
    # src['Pivot Price'] = np.nan  # This line clears old pivot prices
    # src.loc[src['Pivots'] == 1, 'Pivot Price'] = src['high']
    # src.loc[src['Pivots'] == -1, 'Pivot Price'] = src['low']
    #order_block['bull_indic'].astype(float)
    #order_block['bear_indic'].astype(float)

    #np_bull_indic = np.array(order_block['bull_indic']).astype(float)

    #np_bear_indic = np.array(order_block['bear_indic']).astype(float)
    #src = trix(src, 18)

    #macd_data = get_macd(src['close'], 26, 12, 9)
    #macd_data.tail()


    # lengthDivide = length / 2
    # print(lengthDivide)
    # wma1 = wma(df=src, column='close', n=int(lengthDivide))
    # wma1 = wma1*2
    # wma2 = wma(df=src, column='close', n=length)
    # wmasub = list(set(wma1) - set(wma2))
    # print(wmasub)

    # ema1 = ema(df=src, column='close', n=length)
    # ema2 = ema(df=pd.DataFrame(ema1, columns =['ema1']), column='ema1', n=length)
    # ema3 = ema(df=pd.DataFrame(ema2, columns =['ema2']), column='ema2', n=length)
    # tema = list(map(sub,ema1,ema2))
    # for i in range(len(tema)):
    #     tema[i] *= 3
    # tema = list(map(add,tema,ema3))
    #avg = sma(df=src, column='open', n=20)
    #avg2 = sma(df=src, column='open', n=50)
    #avg3 = sma(df=src, column='open', n=100)
    #KClose = StochRSI_K(src, 'close')
    #KOpen = StochRSI_K(src, 'open')

    
    
    # red_dot = []
    # green_dot = []
    # red2_dot = []
    # green2_dot = []
    long_signal = []
    stoplong_signal = []
    short_signal = []
    stopshort_signal = []
    lastTypeTrade = "None"

    lastBearPrice = -150.0
    lastBullPrice = -150.0

    PercentTotal = 0.0
    Balance = 10
    Mise = 10

    GoodTrade = 0
    WrongTrade = 0

    isLongTake = False
    isShortTake = False
    FirstPoint = False
    lastPricetrade = None
    GhostTrade = False
    WaitNewSignal = False
    WaitSignalColor = 0
    StopLossOnLong = True
    StopLossOnShort = True

    candleStop = 0

    #lastDirectionZigZag = 0


    #print(np_bull_indic)

    for i in range(len(DataCrypto1H)):
        
        long_signal.append(np.nan)
        stoplong_signal.append(np.nan)
        short_signal.append(np.nan)
        stopshort_signal.append(np.nan)

        lenClose = len(DataCrypto1H['close'])

        if i <= 7:
            continue

        # sliceArray = i+1
        # pivotsReel = peak_valley_pivots_candlestick(src['close'].iloc[:sliceArray], src['high'].iloc[:sliceArray], src['low'].iloc[:sliceArray] ,.02,-.02)
        # pivotsReelArray = pivotsReel
        # #print(pivotsReel)

        # for i in range(len(pivotsReelArray)):
        #     if i == 0:
        #         continue
        #     if pivotsReelArray[i-1] != 0:
        #         lastDirectionZigZag = pivotsReelArray[i-1]
                
        

        # if lastDirectionZigZag == 0:
        #     continue


        # if np.isnan(order_block['bull_high'][i]) == False:
        #     lastBullPrice = order_block['bull_high'][i]
        # if np.isnan(order_block['bear_low'][i]) == False:
        #     lastBearPrice = order_block['bear_low'][i]
            
        if isLongTake == True :
            candleStop += 1
            
            if GhostTrade == False :

                if (float(lastPricetrade) - ((float(lastPricetrade) * 0.20)/100) > float(DataCrypto1H['close'][i-1])) or float(lastPricetrade) + ((float(lastPricetrade) * 2.0)/100) < float(DataCrypto1H['close'][i-1]):
                #if (macd_data['macd'][i-2] > 0 and macd_data['macd'][i-1] < 0) or float(lastPricetrade) + ((float(lastPricetrade) * 0.25)/100) < float(src['high'][i]):
                    # if (macd_data['macd'][i-2] > 0 and macd_data['macd'][i-1] < 0):
                    #     stoplong_signal[i] = src['open'][i]
                    #     percent = lastPricetrade - src['open'][i]
                    if float(lastPricetrade) - ((float(lastPricetrade) * 0.20)/100) > float(DataCrypto1H['close'][i-1]):
                        stoplong_signal[i] = DataCrypto1H['open'][i]
                        percent = lastPricetrade - DataCrypto1H['open'][i]
                    elif float(lastPricetrade) + ((float(lastPricetrade) * 2.0)/100) < float(DataCrypto1H['close'][i-1]):
                        stoplong_signal[i] = DataCrypto1H['open'][i]
                        percent = lastPricetrade - DataCrypto1H['open'][i]
                    
                    # elif psar_data['psar'][i-1] > src['close'][i-1]:
                    #     stoplong_signal[i] = src['open'][i]
                    #     percent = lastPricetrade - src['open'][i]
                    percent = (100 * percent) / lastPricetrade
                    PercentTotal += (((percent*-1.0) -0.04) -0.04)
                    #Balance += (Mise * (((percent)-0.04)-0.04)) / 100
                    print(str(i) + " Close Long " + str(percent * -1.0))
                    if percent * -1.0 < 0:
                        WrongTrade += 1
                    else :
                        GoodTrade += 1
                    WaitNewSignal = True
                    WaitSignalColor = 1
                    GhostTrade = False 
                    isLongTake = False   
                        

            GhostTrade = False    
            
                
        elif isShortTake == True :
            candleStop += 1
            if GhostTrade == False :
                if (float(lastPricetrade) + ((float(lastPricetrade) * 0.2)/100) < float(DataCrypto1H['close'][i-1])) or float(lastPricetrade) - ((float(lastPricetrade) * 2.0)/100) > float(DataCrypto1H['close'][i-1]):
                #if (macd_data['macd'][i-2] < 0 and macd_data['macd'][i-1] > 0) or float(lastPricetrade) - ((float(lastPricetrade) * 0.25)/100) > float(src['low'][i]):
                    # if (macd_data['macd'][i-2] < 0 and macd_data['macd'][i-1] > 0):
                    #     stopshort_signal[i] = src['open'][i]
                    #     percent = lastPricetrade - src['open'][i]
                    if float(lastPricetrade) + ((float(lastPricetrade) * 0.2)/100) < float(DataCrypto1H['close'][i-1]):
                        stopshort_signal[i] = DataCrypto1H['open'][i]
                        percent = lastPricetrade - DataCrypto1H['open'][i]
                    elif float(lastPricetrade) - ((float(lastPricetrade) * 2.0)/100) > float(DataCrypto1H['close'][i-1]):
                        stopshort_signal[i] = DataCrypto1H['open'][i]
                        percent = lastPricetrade - DataCrypto1H['open'][i]
                    
                    # elif psar_data['psar'][i-1] < src['close'][i-1]:
                    #     stoplong_signal[i] = src['open'][i]
                    #     percent = lastPricetrade - src['open'][i]
                    percent = (100 * percent) / lastPricetrade
                    PercentTotal += (((percent) -0.04) -0.04)
                    #Balance += (Mise * ((percent * -1.0))) / 100
                    print(str(i) + " Close Short "+ str(percent))
                    if percent < 0:
                        WrongTrade += 1
                    else :
                        GoodTrade += 1
                    WaitNewSignal = True
                    WaitSignalColor = 2
                    GhostTrade = False
                    isShortTake = False

            GhostTrade = False
            
        elif isLongTake == False and isShortTake == False:
            if st[i-1] < DataCrypto1H['close'][i-1] and st2[math.floor(i/4)-1] < DataCrypto4H['close'][math.floor(i/4)-1] and psar_data['psar'][i-1] < DataCrypto1H['close'][i-1]:
                if GhostTrade == False :
                    long_signal[i] = DataCrypto1H['open'][i]
                    lastPricetrade = long_signal[i]
                    candleStop = 0
                    print(str(i) + " Open Long ")
                    
                
                isLongTake = True

            elif dt[i-1] > DataCrypto1H['close'][i-1] and dt2[math.floor(i/4)-1] > DataCrypto4H['close'][math.floor(i/4)-1] and psar_data['psar'][i-1] > DataCrypto1H['close'][i-1]:
                if GhostTrade == False :
                    short_signal[i] = DataCrypto1H['open'][i]
                    lastPricetrade = short_signal[i]
                    candleStop = 0
                    print(str(i) + " Open Short ")
                
                isShortTake = True
                        
                




    #red3_dot = []
    #green3_dot = []
    #red4_dot = []
    #green4_dot = []
    # for i in range(len(avg)) :
    #     if i > 0:
    #         if avg[i] < avg[i-1] :
    #             red_dot.append(avg[i])
    #             green_dot.append(np.nan)
    #             red2_dot.append(avg2[i])
    #             green2_dot.append(np.nan)
    #         else :
    #             red_dot.append(np.nan)
    #             green_dot.append(avg[i])
    #             red2_dot.append(np.nan)
    #             green2_dot.append(avg2[i])
    #     else :
    #         red_dot.append(np.nan)
    #         green_dot.append(np.nan)
    #         red2_dot.append(np.nan)
    #         green2_dot.append(np.nan)

    # for i in range(len(avg)) :
    #     long_signal.append(np.nan)
    #     stoplong_signal.append(np.nan)
    #     short_signal.append(np.nan)
    #     stopshort_signal.append(np.nan)
    #     if i > 1 :
 
    #         if isLongTake == True :
    #             if KOpen[i-1] <= 29 and KOpen[i-2] > 30:
    #                 stoplong_signal[i] = src['open'][i]
    #                 percent = lastPricetrade - src['open'][i]
    #                 percent = (100 * percent) / lastPricetrade
    #                 PercentTotal += ((percent)-0.04)-0.04
    #                 Balance += (Mise * (((percent)-0.04)-0.04)) / 100
    #                 # if percent < 0.0 :
    #                 #     print("Erreur : " + str(percent))
    #                 isLongTake = False
    #                 print(str(i) + " Close Long " + str(percent))
    #         if isShortTake == True :
    #             if KOpen[i-1] >= 71 and KOpen[i-2] < 70:
    #                 stopshort_signal[i] = src['open'][i]
    #                 percent = lastPricetrade - src['open'][i]
    #                 percent = (100 * percent) / lastPricetrade
    #                 # if percent < 0.0 :
    #                 #     print("Erreur : " + str(percent))
    #                 PercentTotal += percent
    #                 Balance += (Mise * (((percent * -1.0)-0.04)-0.04)) / 100
    #                 isShortTake = False
    #                 print(str(i) + " Close Short "+ str(((percent * -1.0)-0.04)-0.04))

    #         if isLongTake == False :
    #             if KOpen[i-1] >= 71 and KOpen[i-2] < 70 and np.isnan(green2_dot[i-1]) == False and np.isnan(green_dot[i-1]) == False:
    #                 long_signal[i] = src['open'][i]
    #                 lastPricetrade = long_signal[i]
    #                 print(str(i) + " Open Long ")
    #                 isLongTake = True
                    
    #         if isShortTake == False :
    #             if KOpen[i-1] <= 29 and KOpen[i-2] > 30 and np.isnan(red2_dot[i-1]) == False and np.isnan(red_dot[i-1]) == False:
    #                 short_signal[i] = src['open'][i]
    #                 lastPricetrade = short_signal[i]
    #                 print(str(i) + " Open Short ")
    #                 isShortTake = True
                        
    print(PercentTotal)
    print(GoodTrade)
    print(WrongTrade)         
    #print(Balance)   

    

    ax1 = plt.subplot2grid((8,1), (0,0), colspan = 1, rowspan = 3)
    ax2 = plt.subplot2grid((8,1), (3,0), colspan = 1, rowspan = 3)
    #ax3 = plt.subplot2grid((8,1), (6,0), colspan = 1, rowspan = 3)


    candlestick2_ohlc(ax1,DataCrypto1H['open'],DataCrypto1H['high'],DataCrypto1H['low'],DataCrypto1H['close'],width=0.6, colorup='#77d879', colordown='#db3f3f')

    #ax1.plot(order_block['bull_indic'], marker = 'o', color = 'brown', markersize = 10, label = 'LONG SIGNAL', linewidth = 0)
    #ax1.plot(order_block['bear_indic'], marker = 'o', color = 'blue', markersize = 10, label = 'LONG CLOSE SIGNAL', linewidth = 0)

    #ax1.plot(src['open'], color = 'black', linewidth = 2, label = 'ZECUSDT')
    #ax1.plot(src['close'], color = 'grey', linewidth = 2, label = 'ZECUSDT')

    ax1.plot(st, color = 'green', label = 'SuperTrend Up', linewidth = 4)
    ax1.plot(dt, color = 'red', label = 'SuperTrend Down', linewidth = 4)
    ax1.plot(st2, color = 'yellow', label = 'SuperTrend2 Up', linewidth = 4)
    ax1.plot(dt2, color = 'skyblue', label = 'SuperTrend2 Down', linewidth = 4)
    #ax1.plot(bollinger_up, color = 'skyblue', label = 'SuperTrend Up', linewidth = 4)
    #ax1.plot(bollinger_down, color = 'skyblue', label = 'SuperTrend Down', linewidth = 4)
    #ax1.plot(green_dot, marker = '.', color = 'green', markersize = 6, label = 'ema5/100 Green', linewidth = 0)
    #ax1.plot(red_dot, marker = '.', color = 'red', markersize = 6, label = 'ema5/100 Red', linewidth = 0)
    #ax1.plot(grey_dot, marker = '.', color = 'grey', markersize = 6, label = 'ema5/100 Grey', linewidth = 0)
    #ax1.plot(green2_dot, marker = '.', color = 'green', markersize = 3, label = 'LONG MA2', linewidth = 0)
    #ax1.plot(red2_dot, marker = '.', color = 'red', markersize = 3, label = 'SHORT MA2', linewidth = 0)

    ax1.plot(psar_data['psar'], marker = 'o', color = 'black', markersize = 4, label = 'PSAR', linewidth = 0)

    ax1.plot(long_signal, marker = '^', color = 'orange', markersize = 10, label = 'LONG SIGNAL', linewidth = 0)
    ax1.plot(stoplong_signal, marker = 'x', color = 'orange', markersize = 10, label = 'LONG CLOSE SIGNAL', linewidth = 0)
    ax1.plot(short_signal, marker = 'v', color = 'purple', markersize = 10, label = 'SHORT SIGNAL', linewidth = 0)
    ax1.plot(stopshort_signal, marker = 'x', color = 'purple', markersize = 10, label = 'SHORT CLOSE SIGNAL', linewidth = 0)
    
    #ax1.plot(bollinger_up, label='Bollinger Up', c='g')
    #ax1.plot(bollinger_down, label='Bollinger Down', c='r')

    #ax1.plot(green3_dot, marker = '.', color = 'green', markersize = 4, label = 'LONG MA3', linewidth = 0)
    #ax1.plot(red3_dot, marker = '.', color = 'red', markersize = 4, label = 'SHORT MA3', linewidth = 0)
    #ax1.plot(green4_dot, marker = '.', color = 'green', markersize = 2, label = 'LONG MA4', linewidth = 0)
    #ax1.plot(red4_dot, marker = '.', color = 'red', markersize = 2, label = 'SHORT MA4', linewidth = 0)
    ax1.legend()
    ax1.set_title('Order Block Finder SIGNALS')



    

    #src['Close'].loc['2020-11-01':].plot(ax=ax, alpha=0.3, secondary_y=True)

    # ax2.plot(src['Trix_18'], label='Trix', c='red')
    # ax2.legend()
    # ax2.set_title('TRIX')


    # ax3.plot(macd_data['macd'], color = 'grey', linewidth = 1.5, label = 'MACD')
    # ax3.plot(macd_data['signal'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    # for i in range(len(src['close'])):
    #     if str(macd_data['hist'][i])[0] == '-':
    #         ax3.bar(src['close'].index[i], macd_data['hist'][i], color = '#ef5350')
    #     else:
    #         ax3.bar(src['close'].index[i], macd_data['hist'][i], color = '#26a69a')

    # ax3.legend()
    # ax3.set_title('MACD')

    #ax2.plot(avg, color = 'green', linewidth = 1.5, label = 'avg')
    #ax2.plot(macd['signal'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    # for i in range(len(macd)):
    #     if str(macd['hist'][i])[0] == '-':
    #         ax2.bar(macd.index[i], macd['hist'][i], color = '#ef5350')
    #     else:
    #         ax2.bar(macd.index[i], macd['hist'][i], color = '#26a69a')
            
    #plt.legend(loc = 'lower right')
    plt.show()
    
