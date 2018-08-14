import numpy as np
import pandas as pd
import nsepy as nse
from datetime import date, timedelta
import pandasql as ps

nselisted = pd.read_csv('/Users/gauri/Downloads/LDE_EQUITIES_MORE_THAN_5_YEARS.csv')
nselisted.head(10)
len(nselisted)
nselisted.dtypes
nselisted['First Listing Date'] = pd.to_datetime(nselisted['First Listing Date'])

maxlistdate = max(nselisted['First Listing Date'])
notnew = maxlistdate - timedelta(days=30)

nselisted2 = nselisted[nselisted['First Listing Date'] <= notnew]
nselisted2.shape

nselist = list(nselisted2['Symbol'])
len(nselist)

start_date = date(2017,8,20)
end_date = date(2018,8,10)

histdata = nse.get_history(symbol='SBIN', start=start_date, end=end_date)
print(histdata)
histdata.columns
histdata.drop(columns=['Series'])
histdata = histdata[histdata['Symbol'] != 'SBIN']

for stock in nselist:
    #time.sleep(randrange(10))
    df = nse.get_history(symbol=stock, start=start_date, end=end_date)
    print(stock, df.shape)

    if df.shape[0] >= 30:
        df.drop(columns=['Series'])
        histdata = pd.concat([histdata, df])

histdata.shape
histdata['Symbol'].unique()

histdata.head(30)

histdata.to_csv('/Users/gauri/Downloads/histdata_20180810.csv', sep=',')

histdata.columns




df1 = histdata.copy()

df1.index.min()
df1.index.max()

df1.columns = ['symbol','series','prevclose','open','high','low','last','close','vwap','vol','turnover','numtrades','delvol','pctdel']

df1.head()
df1.shape

df1['dt'] = df1.index
df1.head()

#discard penny stocks
df1.index.max()
pennystocks = df1[df1.index == df1.index.max()]
pennystocks.shape
pennystocks = pennystocks[pennystocks['close'] >= 10]
len(pennystocks.symbol.unique())

df1 = pd.merge(df1, pd.DataFrame(pennystocks['symbol']), left_on='symbol', right_on='symbol')
df1.shape

pysqldf = lambda q: ps.sqldf(q, globals())

q = """
SELECT
symbol, avg(vol) as avgvol, avg(vwap) as avgvwap, avg(delvol) as avgdelvol, avg(pctdel) as avgpctdel
from df1
group by symbol;
"""

df1_q = pysqldf(q)
type(df1_q)

df1_q.to_csv('/Users/gauri/Downloads/histdata_20180810_agg1.csv')

df1.head()

q = """
select a.*, b.mindt from df1 as a, (select symbol, min(dt) as mindt from df1 group by symbol) as b
where a.symbol=b.symbol and a.dt=b.mindt 
"""

df1_q1 = pysqldf(q)
df1_q1.head()
df1_q1.shape

q = """
select a.symbol, a.maxdt, b.prevclose as prevclose_rec, b.open as open_rec, b.high as high_rec, b.low as low_rec, b.last as last_rec,
b.close as close_rec, b.vwap as vwap_rec, b.vol as vol_rec, b.turnover as turnover_rec, b.numtrades as numtrades_rec,
b.delvol as delvol_rec, b.pctdel as pctdel_rec
from df1 as b, (select symbol, max(dt) as maxdt from df1 group by symbol) as a
where a.symbol=b.symbol and b.dt=a.maxdt 
"""

df1_q2 = pysqldf(q)
df1_q2.head()

type(df1)

df1_q3 = pd.merge(left=df1_q1, right=df1_q2, left_on='symbol', right_on='symbol')
df1_q3.shape
df1_q3.head()

df1_q3[df1_q3['symbol']=='SBIN']

df1_q3.dtypes

df1_q3['maxdt'] = pd.to_datetime(df1_q3['maxdt'])
df1_q3['mindt'] = pd.to_datetime(df1_q3['mindt'])

df1_q3['days'] = df1_q3['maxdt'] - df1_q3['mindt']

df1_q3['days'] = df1_q3['days'].astype('timedelta64[D]')

pd.get_option("display.max_columns")
pd.set_option("display.max_columns",30)

def bucketing(a):
    if a<=10000:
        return 1
    elif a<=100000:
        return 2
    elif a<=500000:
        return 3
    elif a<=1000000:
        return 4
    elif a<=2500000:
        return 5
    elif a<=5000000:
        return 6
    elif a<=10000000:
        return 7
    elif a<=50000000:
        return 8
    else:
        return 9

df1_q3['volbucket'] = df1_q3['vol_rec'].apply(bucketing)

df1_q3.head()

q = """
select volbucket, count(*) as num1, avg(vol_rec) as vol_rec, avg(close) as close_prev, avg(close_rec) as close_rec,
avg(close_rec/close) as return,
avg(days/30.0) as mos,
avg(vol) as vol_prev, avg(vol_rec) as vol_rec,
avg(vol_rec/vol) as volchange,
avg(case when (close_rec/close)>=1.30 then 1 else 0 end) as pct30pct,
avg(case when (close_rec/close)>=1.50 then 1 else 0 end) as pct50pct
from df1_q3 
group by volbucket
order by 1
"""

df1_q4 = pysqldf(q)
df1_q4.head(10)

df2 = pd.merge(df1, df1_q3[['symbol','volbucket','mindt','maxdt']], left_on='symbol', right_on='symbol')
df2.head()

import math

def dailyStdDev(row):
    a = row['open']
    b = row['close']
    c = row['high']
    d = row['low']
    e = row['last']
    f = row['vwap']

    return (math.sqrt(((a-f)**2 + (b-f)**2 + (c-f)**2 + (d-f)**2 + (e-f)**2)/4))/f

df2.isnull().values.any()

df2['dailystd'] = df2.apply(dailyStdDev, axis=1)

df2['volpertrade'] = df2['vol']/df2['numtrades']

df2['volpertrade'].mean()

def volpertrade(row):
    a = row['volpertrade']

    if a<=50:
        return 1
    elif a<=100:
        return 2
    elif a<=200:
        return 3
    elif a<=400:
        return 4
    elif a<=800:
        return 5
    else:
        return 6

df2['flag_volpertrade'] = df2.apply(volpertrade, axis=1)

df2.head()

q = """
select volbucket, count(distinct symbol) as num1, avg(vol) as vol, avg(dailystd) as dailystd,
avg(volpertrade) as volpertrade
from df2
group by volbucket
order by 1
"""

df2_q1 = pysqldf(q)
df2_q1.head(10)

stocks = df2[['symbol','volbucket']].drop_duplicates().copy()

stocks[stocks['symbol'].str.match('TATAM')]
stocks[stocks['symbol'].str.match('BANDHAN')]
stocks[stocks['symbol'].str.match('STR')]
stocks[stocks['symbol'].str.match('KALPA')]
stocks[stocks['symbol'].str.match('HCL')]
stocks[stocks['symbol'].str.match('VIP')]
stocks[stocks['symbol'].str.match('BAJAJ')]
stocks[stocks['symbol'].str.match('DABUR')]
stocks[stocks['symbol'].str.match('GRAPH')]
stocks[stocks['symbol'].str.match('SUPER')]

pd.set_option('display.max_rows', 101)

stocks[stocks['volbucket']==5]

#revisit volbucket==5
df1_q3.shape
df1_q3.head()

q = """
select symbol, avg(vol_rec) as vol_rec, avg(close) as close_prev, avg(close_rec) as close_rec,
avg(close_rec/close) as return,
avg(days/30.0) as mos,
avg(vol) as vol_prev, avg(vol_rec) as vol_rec,
avg(vol_rec/vol) as volchange,
avg(case when (close_rec/close)>=1.30 then 1 else 0 end) as pct30pct,
avg(case when (close_rec/close)>=1.50 then 1 else 0 end) as pct50pct
from df1_q3 
where volbucket=5
group by symbol
order by 5 desc
"""

df1_q5 = pysqldf(q)
df1_q5.head(101)

q = """
select symbol, avg(vol) as vol, avg(dailystd) as dailystd,
avg(volpertrade) as volpertrade
from df2
where volbucket=5
group by symbol
order by 3 desc
"""

df1_q6 = pysqldf(q)
df1_q6.head(101)

df1_q7 = pd.merge(df1_q5, df1_q6, left_on='symbol', right_on='symbol')
df1_q7 = df1_q7.drop(columns=['pct30pct','pct50pct'])
df1_q7.head()
df1_q7[df1_q7['return']>=3]

q = """
select (case when return>1 then 1 else 0 end) as flag_return, 
(case when volpertrade <= 50 then 1 
        when volpertrade <= 100 then 2
        when volpertrade <= 200 then 3
        when volpertrade <= 400 then 4
        when volpertrade <= 800 then 5
        else 6 end) as flag_volpertrade,
count(*) as num1, 
avg(return) as return,
avg(vol_rec) as vol, avg(dailystd) as dailystd, avg(volpertrade) as volpertrade,
avg(close_rec) as close
from df1_q7
group by (case when return>1 then 1 else 0 end),
(case when volpertrade <= 50 then 1 
        when volpertrade <= 100 then 2
        when volpertrade <= 200 then 3
        when volpertrade <= 400 then 4
        when volpertrade <= 800 then 5
        else 6 end) 
"""

df1_q8 = pysqldf(q)
df1_q8.head(101)

df1_q7.shape

df1_q7[(df1_q7['volpertrade']<=400) & (df1_q7['return']>1) & (df1_q7['dailystd']<0.02)]

df2.shape

df2[df2['symbol']=='WSTCSTPAPR'].shape

from matplotlib import pyplot as plt

singlestock = df2[df2['symbol']=='KPIT']

plt.plot(singlestock['dt'], singlestock['close'])
plt.show()

#stocks in play as of Aug 2017
nselisted2.shape
nselisted2.head()

oldstocks = nselisted2[nselisted2['First Listing Date'] <= '2017-07-01']
oldstocks.shape
oldstocks.head()

oldstocks2 = oldstocks.sample(random_state=1, frac=0.01)
oldstocks2.shape
oldstocks2.head(25)

oldstocks2 = oldstocks.sample(random_state=3, frac=0.01)
oldstocks2.shape
oldstocks2.head(25)

oldstocks2 = oldstocks.sample(random_state=4, frac=0.01)
oldstocks2.shape
oldstocks2.head(25)

oldstocks3=[
'BANSWRAS.NS.csv',
'TRIGYN.NS.csv',
'EMAMIINFRA.NS.csv',
'SGFL.NS.csv',
'KALYANIFRG.NS.csv',
'KOTARISUG.NS.csv',
'IGARASHI.NS.csv',
'BALAMINES.NS.csv',
'GDL.NS.csv',
'AFL.NS.csv',
'RAMKY.NS.csv',
'SREEL.NS.csv',
'ROLLT.NS.csv',
'SANGHVIFOR.NS.csv',
'SADBHAV.NS.csv',
'ARROWTEX.NS.csv',
'MAHASTEEL.NS.csv',
'HONAUT.NS.csv',
'TVSELECT.NS.csv',
'THOMASCOTT.NS.csv',
'AXISBANK.NS.csv',
'PPAP.NS.csv',
'WEIZFOREX.NS.csv',
'GODREJIND.NS.csv',
'UCALFUEL.NS.csv',
'COSMOFILMS.NS.csv',
'SAMBHAAV.NS.csv',
'URJA.NS.csv',
'CHAMBLFERT.NS.csv',
'BAGFILMS.NS.csv',
'EICHERMOT.NS.csv',
'LOKESHMACH.NS.csv',
'ORIENTBANK.NS.csv',
'DAMODARIND.NS.csv',
'GTNTEX.NS.csv',
'MANGCHEFER.NS.csv',
'GOKEX.NS.csv',
'KOTHARIPRO.NS.csv',
'ZENTEC.NS.csv',
'ORIENTREF.NS.csv',
'TAKE.NS.csv',
'DTIL.NS.csv',
'BLS.NS.csv',
'NIPPOBATRY.NS.csv']

len(oldstocks3)

histdata2 = pd.read_csv('/Users/gauri/Downloads/NIPPOBATRY.NS.csv')
histdata2.head()
histdata2['symbol'] = 'NIPPOBATRY.NS'
histdata2 = histdata2[histdata2['symbol'] != 'NIPPOBATRY.NS']

for stk in oldstocks3:
    fname = '/Users/gauri/Downloads/'+stk
    #print(fname)
    df = pd.read_csv(fname)
    stk = stk.replace('.NS.csv','')
    df['symbol'] = stk
    histdata2 = pd.concat([histdata2, df])

histdata2.head()

histdata2.columns = ['date','open','high','low','close','adjclose','vol','symbol']
histdata2.index.min()

histdata2['volbucket'] = histdata2['vol'].apply(bucketing)

q = """
select a.*, b.mindt, c.maxdt from histdata2 as a, (select symbol, min(date) as mindt from histdata2) as b,
(select symbol, max(date) as mindt from histdata2) as c
where 
"""