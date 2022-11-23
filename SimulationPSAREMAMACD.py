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

def EMA(data, n=20):

    emas = data.ewm(span=n,adjust=False).mean()

    return emas

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



if __name__ == "__main__":


    client_binance = Client('', '')
    

    DataCrypto = get_data_frame(client_binance, 'GMTUSDT', '1 month ago',  '5m') #WOOUSDT #UNFIUSDT #UNIUSDT #TOMOUSDT

    #df_ha = df_ha.iloc[1:,:]
    
    
    #order_block = order_block_finder(src, periods, threshold, usewicks)
    #pivots = peak_valley_pivots_candlestick(src['close'], src['high'], src['low'] ,.02,-.02)
    MACD = get_macd(DataCrypto['close'], 26, 12, 9)
    SAR = psar(DataCrypto)
    EMA_100 = EMA(DataCrypto['close'], 200)

    #colors = []
    #for ind, val in enumerate(DataCrypto['momentum_value']):
    #    if ind > 0:
    #      if val >= 0:
    #        color = 'green'
    #        if val > DataCrypto['momentum_value'][ind-1]:
    #          color = 'lime'
    #      else:
    #        color = 'maroon'
    #        if val < DataCrypto['momentum_value'][ind-1]:
    #          color='red'
    #      colors.append(color)
    #    else:
    #      colors.append('grey')
    
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

    lastBullSAR = -1
    lastBearSAR = -1

    lastBullSARIndex = -1
    lastBearSARIndex = -1
    
    lastMACDCross = 0

    CumulateLoss = 0.0
    CumulateProfit = 0.0
    LastCumulateLoss = 0.0
    LastCumulateProfit = 0.0

    for i in range(len(DataCrypto['close'])):
        
        long_signal.append(np.nan)
        stoplong_signal.append(np.nan)
        short_signal.append(np.nan)
        stopshort_signal.append(np.nan)
        stoplossArray.append(np.nan)
        TakeProfitArray.append(np.nan)

        if i == 0:
            continue

        if SAR['psarbear'][i-1] != None:
            lastBearSAR = SAR['psarbear'][i-1]
            lastBearSARIndex = i-1

        if SAR['psarbull'][i-1] != None:
            lastBullSAR = SAR['psarbull'][i-1]
            lastBullSARIndex = i-1
            
        if MACD['macd'][i-1] > 0 and MACD['signal'][i-1] > 0 and MACD['macd'][i-2] > MACD['signal'][i-2] and MACD['macd'][i-1] < MACD['signal'][i-1]:
            lastMACDCross = 1
        
        if MACD['macd'][i-1] < 0 and MACD['signal'][i-1] < 0 and MACD['macd'][i-2] < MACD['signal'][i-2] and MACD['macd'][i-1] > MACD['signal'][i-1]:
            lastMACDCross = -1
            
        if isLongTake == True :
            print("Enter is Long Take")
            if (StopLossPrice >= float(DataCrypto['high'][i]) or StopLossPrice >= float(DataCrypto['low'][i])) or (TakeProfitPrice <= float(DataCrypto['high'][i]) or TakeProfitPrice <= float(DataCrypto['low'][i])):
                if (StopLossPrice >= float(DataCrypto['high'][i]) or StopLossPrice >= float(DataCrypto['low'][i])):
                    if DataCrypto['open'][i] < StopLossPrice:
                        stoplong_signal[i] = DataCrypto['open'][i]
                        percent = lastPricetrade - DataCrypto['open'][i]
                    else:
                        stoplong_signal[i] = StopLossPrice
                        percent = lastPricetrade - StopLossPrice
                elif (TakeProfitPrice <= float(DataCrypto['high'][i]) or TakeProfitPrice <= float(DataCrypto['low'][i])):
                    if DataCrypto['open'][i] > TakeProfitPrice:
                        stoplong_signal[i] = DataCrypto['open'][i]
                        percent = lastPricetrade - DataCrypto['open'][i]
                    else:
                        stoplong_signal[i] = TakeProfitPrice
                        percent = lastPricetrade - TakeProfitPrice
                
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
            if (StopLossPrice <= float(DataCrypto['high'][i]) or StopLossPrice <= float(DataCrypto['low'][i])) or (TakeProfitPrice >= float(DataCrypto['high'][i]) or TakeProfitPrice >= float(DataCrypto['low'][i])):
                if (StopLossPrice <= float(DataCrypto['high'][i]) or StopLossPrice <= float(DataCrypto['low'][i])):
                    if DataCrypto['open'][i] > StopLossPrice:
                        stopshort_signal[i] = DataCrypto['open'][i]
                        percent = lastPricetrade - DataCrypto['open'][i]
                    else:
                        stopshort_signal[i] = StopLossPrice
                        percent = lastPricetrade - StopLossPrice
                elif (TakeProfitPrice >= float(DataCrypto['high'][i]) or TakeProfitPrice >= float(DataCrypto['low'][i])):
                    if DataCrypto['open'][i] < TakeProfitPrice:
                        stopshort_signal[i] = DataCrypto['open'][i]
                        percent = lastPricetrade - DataCrypto['open'][i]
                    else:
                        stopshort_signal[i] = TakeProfitPrice
                        percent = lastPricetrade - TakeProfitPrice
                
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
            if lastMACDCross == -1 and SAR['psarbull'][i-1] != None and DataCrypto['close'][i-1] > EMA_100[i-1]:
                

                StopLossPrice = DataCrypto['low'][lastBearSARIndex]
                
                if float(DataCrypto['open'][i]) - ((float(DataCrypto['open'][i]) * 3.0)/100) <= float(DataCrypto['low'][lastBearSARIndex]) and StopLossPrice < DataCrypto['open'][i]:
                    
                    long_signal[i] = DataCrypto['open'][i]
                    lastPricetrade = long_signal[i]
                    
                    TakeProfitPrice =  DataCrypto['open'][i] + (((DataCrypto['low'][lastBearSARIndex] - DataCrypto['open'][i]) * 1.0) * -1.0)

                    stoplossArray[i] = StopLossPrice
                    TakeProfitArray[i] = TakeProfitPrice
                    print(long_signal[i])
                    print(StopLossPrice)
                    print(TakeProfitPrice)
                    print(str(i) + " Open Long ")
                        
                    
                    isLongTake = True

            elif lastMACDCross == 1 and SAR['psarbear'][i-1] != None and DataCrypto['close'][i-1] < EMA_100[i-1]:
                print("entershort condition" + str(i))

                StopLossPrice = DataCrypto['high'][lastBullSARIndex]
                

                if float(DataCrypto['open'][i]) + ((float(DataCrypto['open'][i]) * 3.0)/100) >= float(DataCrypto['high'][lastBullSARIndex]) and StopLossPrice > DataCrypto['open'][i]:
                    print("Take Short")
                    short_signal[i] = DataCrypto['open'][i]
                    
                    lastPricetrade = short_signal[i]

                    TakeProfitPrice =  DataCrypto['open'][i] - (((DataCrypto['high'][lastBullSARIndex] - DataCrypto['open'][i]) * 1.0))
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

    ax1.plot(EMA_100, color = 'purple', label = 'SuperTrend Up', linewidth = 2)
    # ax1.plot(dt, color = 'red', label = 'SuperTrend Down', linewidth = 4)
    ax1.plot(SAR['psarbull'], marker = '.', color = 'blue', markersize = 2, label = 'LONG MA3', linewidth = 0)
    ax1.plot(SAR['psarbear'], marker = '.', color = 'skyblue', markersize = 2, label = 'LONG MA3', linewidth = 0)
    #ax1.plot(red3_dot, marker = '.', color = 'red', markersize = 4, label = 'SHORT MA3', linewidth = 0)

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

    

    #for i in range(len(DataCrypto['momentum_value'])):
    #   ax2.bar(DataCrypto.index[i], DataCrypto['momentum_value'][i], color = colors[i])
    
    

    

    plt.show()