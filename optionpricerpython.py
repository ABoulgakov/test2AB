# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:14:12 2022

@author: alric
"""

'BSM option pricer'

import numpy as np
from datetime import date
from scipy.stats import norm
import matplotlib.pyplot as plt


def NumOfDays(d1, d2):
    return (d1-d2).days



def BS_call_price(S, K, r, sigma, TradeDate, StrikeDate):
    
    sigma2 = sigma**2
    TimeToMatu_Y = NumOfDays(StrikeDate, TradeDate)/365
    
    d1 = ( np.log(S/K) - (r - sigma2/2)*TimeToMatu_Y ) / (sigma*np.sqrt(TimeToMatu_Y))
    d2 = d1 - sigma * np.sqrt(TimeToMatu_Y)
    
    return S*norm.cdf(d1) - K*np.exp(-r*TimeToMatu_Y)*norm.cdf(d2)



def BS_put_price(S, K, r, sigma, TradeDate, StrikeDate):
    
    sigma2 = sigma**2
    TimeToMatu_Y = NumOfDays(StrikeDate, TradeDate)/365
    
    d1 = ( np.log(S/K) - (r - sigma2/2)*TimeToMatu_Y ) / (sigma*np.sqrt(TimeToMatu_Y))
    d2 = d1 - sigma * np.sqrt(TimeToMatu_Y)
    
    return  K*np.exp(-r*TimeToMatu_Y)*norm.cdf(-d2) - S*norm.cdf(-d1) 


test_call_BS = BS_call_price(300, 320, 0.04, 0.20, date(2022, 10, 16), date(2023, 10, 16))
test_put_BS = BS_put_price(300, 299, 0.03, 0.20, date(2022, 10, 16), date(2023, 10, 16))

print("call price : ", test_call_BS)
'print("put price : ", test_put_BS)'

def Bachelier_call_price(S, K, r, TradeDate, StrikeDate, sigma):
    TTMY = NumOfDays(StrikeDate, TradeDate)/365
    u = np.sqrt((1-np.exp(-2*r*TTMY))/(2*r))
    z = (S- K*np.exp(-r*TTMY)) / (sigma*u)
    return (S-K*np.exp(-r*TTMY))*norm.cdf(z) + sigma*u*norm.pdf(z)

test_call_Bachelier = Bachelier_call_price(300, 320, 0.04, date(2022, 10, 16), date(2023, 10, 16), 0.20 )
print('Bachelier call price', test_call_Bachelier) 

abs = list(range(1,500))
abslittle = list(map(lambda number : number/100, abs))
prix_BS = np.array(list(map(lambda number : BS_call_price(300, number, 0.04, 0.20, date(2022, 10, 16), date(2023, 10, 16)), abs)))
prix_Bach = np.array(list(map(lambda number : Bachelier_call_price(300, number, 0.04, date(2022, 10, 16), date(2023, 10, 16), 0.20 ), abs)))
diff = prix_BS - prix_Bach 


plt.grid(True)
plt.plot(abs, prix_BS, label = "Prix BS")
plt.plot(abs, prix_Bach, label = "Prix Bachelier")
plt.plot(abs, diff, label = "Différence de prix")
plt.legend()
plt.show()











