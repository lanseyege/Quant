import pandas as pd
import numpy as np
import time 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import svm

class Ress():
    def __init__(self):
        pass

    def get_data(self,fname,is_read,begd,endd):
        #alldata = pd.read_csv(fname).fillna(0)
        alldata = pd.read_csv(fname).dropna()
        if is_read:
            alldata = alldata[alldata.TRADE_DT>begd and alldata.TRADE_DT<endd]
        dastock = alldata.loc[:,'Code'].drop_duplicates().tolist()
        dadates = alldata.loc[:,'TRADE_DT'].drop_duplicates().tolist()
        dadates.sort()
        dall1 = alldata.set_index('TRADE_DT')
        return dadates, dall1

    def get_pred(self,dadates,dall1,days,fw,params,tbegd,tendd,is_trans):
        n = len(dadates)
        cod = -4
        temp = dall1.iloc[1,:].tolist()
        print('temp: '+str(len(temp)))
        xx = np.arange(len(temp)-1)
        xx = np.delete(xx,cod)

        for i in range(n-1,days-1,-1):
            if is_trans and (dadates[i]<tbegd or dadates[i]>tendd):
                continue
            data = dall1.loc[dadates[i-days:i-1]]
            x = data.iloc[:,xx]
            y = data.iloc[:,-1]
            #clf = RandomForestClassifier(**params)
            #clf = make_pipeline(PolynomialFeatures(2), Ridge())
            clf = svm.SVR()
            clf.fit(x,y)
            data_ = dall1.loc[dadates[i]]
            x_ = data_.iloc[:,xx]
            y_ = data_.iloc[:,-1].tolist()
            stock = data_.iloc[:,cod-1].tolist()
            returnm3 = data_.loc[:,'Return_m3'].tolist()
            turnoverd5 = data_.loc[:,'TurnOver_D_m5'].tolist()
            Pred = clf.predict(x_)
            self.get_rate(y_,Pred,stock,returnm3,turnoverd5,dadates[i],fw)
        fw.close()

    def get_rate(self,Actl,Pred,stock,returnm3,turnoverd5,date,fw):
        num = len(Pred)
        for i in range(num):
            #print(str(stock[i])+','+str(date)+','+str(Actl[i])+','+str(Pred[i])+','+str(turnoverd5[i])+','+str(returnm3[i])+'\n')
            fw.write(stock[i]+','+date+','+str(Actl[i])+','+str(Pred[i])+','+str(turnoverd5[i])+','+str(returnm3[i])+'\n')

if __name__ == '__main__':
    is_read = False
    begd = '2017-08-01'
    endd = '9999-12-30'
    is_trans = True
    today = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    tbegd = '2018-01-01'
    tendd = '9999-12-30'
    params = {
        'random_state':10,
        'n_estimators':300,
        'n_jobs':-1,
    }
    days = 10
    fname='/home/wang1/YWork/data/regss/Data_for_Regression_UpLine.csv'
    fstore = '/home/wang1/YWork/regss/res/p0_svr_1.csv'
    fw = open(fstore, 'w')
    fw.write('Code,TRADE_DT,Actual,Predict,TurnOver_D_m5,Return_m3\n')
    ress = Ress()
    print('read head')
    print('read data')
    dadates, dall1 = ress.get_data(fname, is_read, begd, endd)
    print('predict...')
    ress.get_pred(dadates,dall1,days,fw,params,tbegd,tendd,is_trans)
    print('done!')
