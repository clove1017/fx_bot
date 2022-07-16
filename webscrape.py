#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Import Statements
import time
from selenium import webdriver
from selenium.common import exceptions
import numpy as np 
from selenium.webdriver.chrome.options import Options
import os
from datetime import datetime
import pandas as pd

#Webscraping
def scraper(time_gap):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(options=options,executable_path='C:/webdrivers/chromedriver.exe')
    driver.get('https://www.investing.com/currencies/eur-usd-candlestick')
    time.sleep(5)
    try:
        onehour_link = driver.find_element_by_link_text(time_gap)
        onehour_link.click()
    except exceptions.ElementClickInterceptedException:
        element = driver.find_element_by_xpath("//span[@class='closeIcon js-close']")
        driver.execute_script("arguments[0].style.visibility='hidden'", element)
    time.sleep(5)
    body_colors=[]
    candle_coord=[]
    start=False
    endcolor=False
    for item in driver.find_elements_by_tag_name("path"):
        try:
            if not endcolor:
                if (item.get_attribute("fill")=="#32ea32") or (item.get_attribute("fill")=="#fe3232"):
                    body_colors.append(item.get_attribute("fill"))
                    candle_coord.append((item.get_attribute('d')))
                    startcolor=True
                elif start:
                    endcolor=True
        except exceptions.StaleElementReferenceException as e:
            pass
    yaxis=[]
    xaxis=[]
    yaxcoord=[]
    xaxcoord=[]
    for item in driver.find_elements_by_tag_name("text"):
        try:
            xadd=False
            yadd=False
            if item.text!='':
                holder=(item.text).split('.')
                if len(holder)>1:
                    yadd=True
                else:
                    xadd=True
                if xadd:
                    xaxis.append(item.text)
                    xaxcoord.append(item.get_attribute("x"))
                elif yadd:
                    yaxis.append(float(item.text))
                    yaxcoord.append(float(item.get_attribute("y")))
            else:
                break
        except exceptions.StaleElementReferenceException as e:
            pass  

    price_at_runtime=driver.find_element_by_id("chart-info-last")
    price_at_runtime=price_at_runtime.text
    driver.quit()
    x = np.array(yaxcoord)
    y = np.array(yaxis)
    m, b = np.polyfit(x,y,1)
    candle_coordsp=[]
    dates=['x']*len(candle_coord)
    years=['x']*len(candle_coord)
    months=['x']*len(candle_coord)
    days=['x']*len(candle_coord)
    hours=['x']*len(candle_coord)
    open_data=[]
    close_data=[]
    high_data=[]
    low_data=[]
    spacingcheck=[]

    for x in range(0,len(candle_coord)):
        holder=(candle_coord[x]).split()
        if body_colors[x]=='#32ea32':
            open_data.append(m*float(holder[2])+b)
            close_data.append(m*float(holder[5])+b)
            high_data.append(m*float(holder[18])+b)
            low_data.append(m*float(holder[24])+b)
        elif body_colors[x]=='#fe3232':
            open_data.append(m*float(holder[5])+b)
            close_data.append(m*float(holder[2])+b)
            high_data.append(m*float(holder[18])+b)
            low_data.append(m*float(holder[24])+b)
        for l in range(0,len(xaxcoord)):
            if (float(holder[1])-1)<=(float(xaxcoord[l]))<=(float(holder[7])+1):
                spacingcheck.append(x)
    uneven=False
    unevindex=[]
    from datetime import datetime
    today = datetime.today()
    cyear = ((str(today)).split('-'))[0]
    cmonth= ((str(today)).split('-'))[1]
    cday1= ((str(today)).split('-'))[2]
    cday= (str(cday1).split(' '))[0]
    chour1= ((str(today)).split(' '))[1]
    chour=(str(chour1).split(':'))[0]

    htrack=int(chour)
    dtrack=int(cday)
    mtrack=int(cmonth)
    ytrack=int(cyear)
    for i in range(len(hours)-1,-1,-1):
        hours[i]=htrack
        days[i]=dtrack
        months[i]=mtrack
        years[i]=ytrack
        if htrack==0:
            htrack=23
            if dtrack==1:
                if cmonth==2 or cmonth==4 or cmonth==6 or cmonth==9 or cmonth==11 or cmonth==1 or cmonth==8:
                    dtrack=31
                    if cmonth==1:
                        ytrack=ytrack-1
                elif cmonth==3:
                    dtrack=28
                else:
                    dtrack=30
            else:
                dtrack=dtrack-1                   
        else:
            htrack=htrack-1
    for i in range(0,len(dates)):
        dates[i]=datetime(year=int(years[i]),month=int(months[i]),day=int(days[i]),hour=int(hours[i]))

    x = 0
    y = len(dates)
    ohlc = []

    df=pd.DataFrame()
    df['Date']=pd.to_datetime(dates)
    df['Open']=open_data
    df['High']=high_data
    df['Low']=low_data
    df['Close']=close_data
    df.set_index('Date',inplace=True)
    #create time delt array
    time_delt=(datetime.now()-pd.to_datetime(dates))
    td_days=np.zeros(len(time_delt))
    for i in range(0,len(time_delt)):
        holder=str(time_delt[i]).split(' ')
        split_hms=holder[2].split(':')
        hours=float(split_hms[0])+float(split_hms[1])/60+float(split_hms[2])/3600
        days=float(holder[0])+hours/24
        td_days[i]=days

    return open_data,close_data,high_data,low_data,td_days


# In[ ]:




