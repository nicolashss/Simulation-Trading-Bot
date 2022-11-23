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

def WMA(s, period):
       return s.rolling(period).apply(lambda x: ((np.arange(period)+1)*x).sum()/(np.arange(period)+1).sum(), raw=True)

def HMA(s, period):
       return WMA(WMA(s, period//2).multiply(2).sub(WMA(s, period)), int(np.sqrt(period)))

def EMA(data, n=20):

    emas = data.ewm(span=n,adjust=False).mean()

    return emas

def HA(df):
    df['HA_close']=(df['open']+ df['high']+ df['low']+df['close'])/4

    idx = df.index.name
    df.reset_index(inplace=True)

    for i in range(0, len(df)):
        if i == 0:
            df.at[i, 'HA_open'] = ((df.at[i, 'open'] + df.at[i, 'close']) / 2)
        else:
            df.at[i, 'HA_open'] = ((df.at[i - 1, 'HA_open'] + df.at[i - 1, 'HA_close']) / 2)

    if idx:
        df.set_index(idx, inplace=True)

    df['HA_high']=df[['HA_open','HA_close','high']].max(axis=1)
    df['HA_low']=df[['HA_open','HA_close','low']].min(axis=1)
    return df

def SSL_Custom(df):
    BBMC = HMA(df['HA_close'], 60)
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift())
    low_cp = np.abs(df['low'] - df['close'].shift())
    TR = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(TR, axis=1)
    rangema = EMA(true_range, 60)

    upperk = BBMC + rangema * 0.2
    lowerk = BBMC - rangema * 0.2

    color_bar = true_range.copy()
    for i in range(len(color_bar)):
        if(df['HA_close'][i] > upperk[i]):        
            color_bar[i] = 1
        elif(df['HA_close'][i] < lowerk[i]):
            color_bar[i] = -1
        else:
            color_bar[i] = 0
                
    return BBMC, color_bar

def get_wr(high, low, close, lookback, lookbackema):
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    emawr = EMA(wr, lookbackema)
    return wr, emawr


if __name__ == "__main__":

    

    

    client_binance = Client('', '')
    

    DataCrypto = get_data_frame(client_binance, 'ATOMUSDT', '3 days ago',  '1m') #WOOUSDT #UNFIUSDT #UNIUSDT #TOMOUSDT


    DataCrypto = HA(DataCrypto)
    #df_ha = df_ha.iloc[1:,:]
    
    
    #order_block = order_block_finder(src, periods, threshold, usewicks)
    #pivots = peak_valley_pivots_candlestick(src['close'], src['high'], src['low'] ,.02,-.02)
    BBMC, color_bar = SSL_Custom(DataCrypto)
    wr, emawr = get_wr(DataCrypto['high'], DataCrypto['low'], DataCrypto['close'], 42, 26)

    
    
    #exit()


    long_signal = []
    stoplong_signal = []
    short_signal = []
    stopshort_signal = []

    stoplossArray = []
    TakeProfitArray = []
    blueArrow = []
    pinkArrow = []
    blueArrowCounter = 0
    pinkArrowCounter = 0

    PercentTotal = 0.0
    Balance = 10
    Mise = 10

    GoodTrade = 0
    WrongTrade = 0

    isLongTake = False
    isShortTake = False
    lastPricetrade = None

    StopLossPrice = None
    TakeProfitPrice = None

    WorstTrade = 0.0
    BestTrade = 0.0

    lastRed = 0
    lastGreen = 0

    lastNumberRed = 0
    lastNumberGreen = 0
    trendCandle = 1

    CumulateLoss = 0.0
    CumulateProfit = 0.0
    LastCumulateLoss = 0.0
    LastCumulateProfit = 0.0

    for i in range(len(DataCrypto['HA_close'])):
        
        long_signal.append(np.nan)
        stoplong_signal.append(np.nan)
        short_signal.append(np.nan)
        stopshort_signal.append(np.nan)
        stoplossArray.append(np.nan)
        TakeProfitArray.append(np.nan)

        lenClose = len(DataCrypto['close'])

        if i < 2:
            continue

        if (DataCrypto['HA_open'][i-1] < DataCrypto['HA_close'][i-1]):

            lastGreen = i
            lastNumberGreen += 1


        elif (DataCrypto['HA_open'][i-1] > DataCrypto['HA_close'][i-1]):

            lastRed = i
            lastNumberRed += 1


        
            
        if isLongTake == True :
            print("Enter is Long Take")
            if color_bar[i-1] == -1:
                stoplong_signal[i] = DataCrypto['open'][i]
                percent = lastPricetrade - DataCrypto['open'][i]
            #if (StopLossPrice >= float(DataCrypto['high'][i]) or StopLossPrice >= float(DataCrypto['low'][i])) or (TakeProfitPrice <= float(DataCrypto['high'][i]) or TakeProfitPrice <= float(DataCrypto['low'][i])):
                # if (StopLossPrice >= float(DataCrypto['high'][i]) or StopLossPrice >= float(DataCrypto['low'][i])):
                #     if DataCrypto['open'][i] < StopLossPrice:
                #         stoplong_signal[i] = DataCrypto['open'][i]
                #         percent = lastPricetrade - DataCrypto['open'][i]
                #     else:
                #         stoplong_signal[i] = StopLossPrice
                #         percent = lastPricetrade - StopLossPrice
                # elif (TakeProfitPrice <= float(DataCrypto['high'][i]) or TakeProfitPrice <= float(DataCrypto['low'][i])):
                #     if DataCrypto['open'][i] > TakeProfitPrice:
                #         stoplong_signal[i] = DataCrypto['open'][i]
                #         percent = lastPricetrade - DataCrypto['open'][i]
                #     else:
                #         stoplong_signal[i] = TakeProfitPrice
                #         percent = lastPricetrade - TakeProfitPrice
                
                percent = (100 * percent) / lastPricetrade
                PercentTotal += (((percent*-1.0) -0.04) -0.04)
                #Balance += (Mise * (((percent)-0.04)-0.04)) / 100
                print(str(i) + " Close Long " + str((((percent*-1.0) -0.04) -0.04)))
                if((((percent*-1.0) -0.04) -0.04) < 0):
                    CumulateProfit = 0.0
                    CumulateLoss += (((percent*-1.0) -0.04) -0.04)
                    if CumulateLoss < LastCumulateLoss:
                        LastCumulateLoss = CumulateLoss
                    if WorstTrade > (((percent*-1.0) -0.04) -0.04):
                        WorstTrade = (((percent*-1.0) -0.04) -0.04)

                if((((percent*-1.0) -0.04) -0.04) > 0):
                    CumulateLoss = 0.0
                    CumulateProfit += (((percent*-1.0) -0.04) -0.04)
                    if CumulateProfit > LastCumulateProfit:
                        LastCumulateProfit = CumulateProfit
                    if BestTrade < (((percent*-1.0) -0.04) -0.04):
                        BestTrade = (((percent*-1.0) -0.04) -0.04)
                if percent * -1.0 < 0:
                    WrongTrade += 1
                else :
                    GoodTrade += 1

                isLongTake = False   
                        

            
                
        elif isShortTake == True :
            print("Enter is Short Take")
            if color_bar[i-1] == 1:
                stopshort_signal[i] = DataCrypto['open'][i]
                percent = lastPricetrade - DataCrypto['open'][i]
            #if (StopLossPrice <= float(DataCrypto['high'][i]) or StopLossPrice <= float(DataCrypto['low'][i])) or (TakeProfitPrice >= float(DataCrypto['high'][i]) or TakeProfitPrice >= float(DataCrypto['low'][i])):
                # if (StopLossPrice <= float(DataCrypto['high'][i]) or StopLossPrice <= float(DataCrypto['low'][i])):
                #     if DataCrypto['open'][i] > StopLossPrice:
                #         stopshort_signal[i] = DataCrypto['open'][i]
                #         percent = lastPricetrade - DataCrypto['open'][i]
                #     else:
                #         stopshort_signal[i] = StopLossPrice
                #         percent = lastPricetrade - StopLossPrice
                # elif (TakeProfitPrice >= float(DataCrypto['high'][i]) or TakeProfitPrice >= float(DataCrypto['low'][i])):
                #     if DataCrypto['open'][i] < TakeProfitPrice:
                #         stopshort_signal[i] = DataCrypto['open'][i]
                #         percent = lastPricetrade - DataCrypto['open'][i]
                #     else:
                #         stopshort_signal[i] = TakeProfitPrice
                #         percent = lastPricetrade - TakeProfitPrice
                
                percent = (100 * percent) / lastPricetrade
                PercentTotal += (((percent) -0.04) -0.04)
                #Balance += (Mise * ((percent * -1.0))) / 100
                print(str(i) + " Close Short "+ str((((percent) -0.04) -0.04)))
                if((((percent) -0.04) -0.04) < 0):
                    CumulateProfit = 0.0
                    CumulateLoss += (((percent) -0.04) -0.04)
                    if CumulateLoss < LastCumulateLoss:
                        LastCumulateLoss = CumulateLoss
                    if WorstTrade > (((percent) -0.04) -0.04):
                        WorstTrade = (((percent) -0.04) -0.04)

                if((((percent) -0.04) -0.04) > 0):
                    CumulateLoss = 0.0
                    CumulateProfit += (((percent) -0.04) -0.04)
                    if CumulateProfit > LastCumulateProfit:
                        LastCumulateProfit = CumulateProfit
                    if BestTrade < (((percent) -0.04) -0.04):
                        BestTrade = (((percent) -0.04) -0.04)
                if (((percent) -0.04) -0.04) < 0:
                    WrongTrade += 1
                else :
                    GoodTrade += 1
                isShortTake = False

            
        elif isLongTake == False and isShortTake == False:
            if wr[i-2] < emawr[i-2] and wr[i-1] > emawr[i-1] and color_bar[i-1] == 1:
                

                StopLossPrice = DataCrypto['close'][lastRed]
                
                if float(DataCrypto['open'][i]) - ((float(DataCrypto['open'][i]) * 3.0)/100) <= float(DataCrypto['close'][lastRed]) and StopLossPrice < DataCrypto['open'][i]:
                    
                    long_signal[i] = DataCrypto['open'][i]
                    lastPricetrade = long_signal[i]
                    
                    TakeProfitPrice =  DataCrypto['open'][i] + (((DataCrypto['low'][lastRed] - DataCrypto['open'][i]) * 1.5) * -1.0)

                    stoplossArray[i] = StopLossPrice
                    TakeProfitArray[i] = TakeProfitPrice
                    print(long_signal[i])
                    print(StopLossPrice)
                    print(TakeProfitPrice)
                    print(str(i) + " Open Long ")
                        
                    
                    isLongTake = True

            elif wr[i-2] > emawr[i-2] and wr[i-1] < emawr[i-1] and color_bar[i-1] == -1:
                print("entershort condition" + str(i))

                StopLossPrice = DataCrypto['close'][lastGreen]
                

                if float(DataCrypto['open'][i]) + ((float(DataCrypto['open'][i]) * 3.0)/100) >= float(DataCrypto['close'][lastGreen]) and StopLossPrice > DataCrypto['open'][i]:
                    print("Take Short")
                    short_signal[i] = DataCrypto['open'][i]
                    
                    lastPricetrade = short_signal[i]

                    TakeProfitPrice =  DataCrypto['open'][i] - (((DataCrypto['high'][lastGreen] - DataCrypto['open'][i]) * 1.5))
                    stoplossArray[i] = StopLossPrice
                    TakeProfitArray[i] = TakeProfitPrice
                    print(short_signal[i])
                    print(StopLossPrice)
                    print(TakeProfitPrice)
                    print(str(i) + " Open Short ")
                    
                    isShortTake = True

    print(" ----------------- ")
    print("Pourcentage Total: " + str(PercentTotal))
    
    print("Nombre de Trade Positif : " + str(GoodTrade))
    print("Nombre de Trade Negatif : " + str(WrongTrade))
    print("Meilleur Trade Positif : " + str(BestTrade))
    print("Pire Trade Negatif : " + str(WorstTrade))

    
    print("Meilleur Profit Consecutif : " + str(LastCumulateProfit))
    print("Pire Perte Consecutive : " + str(LastCumulateLoss))
    ax1 = plt.subplot2grid((8,1), (0,0), colspan = 1, rowspan = 3)
    ax2 = plt.subplot2grid((8,1), (3,0), colspan = 1, rowspan = 3)
    #ax3 = plt.subplot2grid((8,1), (6,0), colspan = 1, rowspan = 3)


    candlestick2_ohlc(ax1,DataCrypto['open'],DataCrypto['high'],DataCrypto['low'],DataCrypto['close'],width=0.6, colorup='#77d879', colordown='#db3f3f')

    # ax1.plot(st, color = 'green', label = 'SuperTrend Up', linewidth = 4)
    

    ax1.plot(long_signal, marker = '^', color = 'orange', markersize = 10, label = 'LONG SIGNAL', linewidth = 0)
    ax1.plot(stoplong_signal, marker = 'x', color = 'orange', markersize = 10, label = 'LONG CLOSE SIGNAL', linewidth = 0)
    ax1.plot(short_signal, marker = 'v', color = 'purple', markersize = 10, label = 'SHORT SIGNAL', linewidth = 0)
    ax1.plot(stopshort_signal, marker = 'x', color = 'purple', markersize = 10, label = 'SHORT CLOSE SIGNAL', linewidth = 0)
    ax1.plot(TakeProfitArray, marker = '.', color = 'green', markersize = 10, label = 'LONG SIGNAL', linewidth = 0)
    ax1.plot(stoplossArray, marker = '.', color = 'red', markersize = 10, label = 'LONG CLOSE SIGNAL', linewidth = 0)
    #ax1.plot(blueArrow, marker = '^', color = 'blue', markersize = 10, label = 'LONG SIGNAL', linewidth = 0)
    #ax1.plot(pinkArrow, marker = 'v', color = 'pink', markersize = 10, label = 'SHORT SIGNAL', linewidth = 0)
    
    ax1.legend()
    ax1.set_title('Order Block Finder SIGNALS')
    
    
    ax2.plot(wr, color = 'orange', label = 'SuperTrend Down', linewidth = 4)
    ax2.plot(emawr, color = 'blue', label = 'SuperTrend Down', linewidth = 4)
    

    plt.show()