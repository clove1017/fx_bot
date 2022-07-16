from webscrape import scraper
from prediction import form_prediction
import torch

if __name__=="__main__":
    
    #retrieve data
    scrape=True
    while scrape:
        try:
            open_data,close_data,high_data,low_data,td_days=scraper('1H')
            scrape=False
        except:
            print('Error obtaining data')
            pass
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #form prediction and obtain trade size
    trade_size=form_prediction(open_data,close_data,high_data,low_data,td_days,device) #still need to add prediction value/direction
    print(trade_size)