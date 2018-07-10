import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time 

class RFST():
    def __init__(self):
        pass

    def get_head(self,fhead):
        fn = open(fhead)
        i = 0
        head = []
        tite = {}
        for line in fn:
            if i == 0:
                head = line.strip().split(',')
                i = 1
                continue
            line = line.strip().split(',')
            A = {}
            for j, l in enumerate(line):
                A[l] = j
            A[0] = 0
            tite[head[i-1]] = A
            i += 1
        fn.close()
        head.insert(0,'Label1')
        tite['Label1'] = {'UP':1,'DOWN':2,'HOLD':3, 0:0}
        #print(head)
        #print(tite)
        return head, tite
 
    def get_data(self,fname,is_read,begd,endd,head,tite):
        alldata = pd.read_csv(fname).fillna(0)
        #alldata = alldata[alldata['Return_m3']<0.98]
        cols = alldata.columns.values
        if is_read:
            alldata = alldata[alldata.TRADE_DT>begd and alldata.TRADE_DT<endd]
        for hd in head:
            if hd in cols:
                alldata[hd] = alldata[hd].map(tite[hd])       
        dastock = alldata.loc[:,'Code'].drop_duplicates().tolist()
        dadates = alldata.loc[:,'TRADE_DT'].drop_duplicates().tolist()
        dadates.sort()
        dall1 = alldata.set_index('TRADE_DT')
        return dadates, dall1

    def get_pred(self,dadates,dall1,days,fw,params,tbegd,tendd,is_trans):
        n = len(dadates)
        yzsz_r = 0
        yzsp_r = 0
        yzsd_r = 0
        yzsz = 0
        yzsp = 0
        yzsd = 0
        ydsz_r = 0
        ydsd_r = 0
        ydsp_r = 0
        ydsz = 0
        ydsd = 0
        ydsp = 0
        ypsz_r = 0
        ypsd_r = 0
        ypsp_r = 0
        ypsz = 0
        ypsd = 0
        ypsp = 0
        sums = 0
        for i in range(n-1,days-1,-1):
            if is_trans and (dadates[i]<tbegd or dadates[i]>tendd):
                continue
            data = dall1.loc[dadates[i-days:i]]
            x = data.drop(['Label1','Code'],axis=1)
            y = data.Label1.values
            clf = RandomForestClassifier(**params)
            clf.fit(x,y)
            data_ = dall1.loc[dadates[i]]
            x_ = data_.drop(['Label1','Code'],axis=1)
            y_ = data_.Label1.values
            px = np.asarray(clf.predict_proba(x)).T
            px_ = np.asarray(clf.predict_proba(x_)).T
            x = x.assign(l11=px[0],l12=px[1],l13=px[2])
            x_ = x_.assign(l11=px_[0],l12=px_[1],l13=px_[2])
            clf.fit(x,y)
            px = np.asarray(clf.predict_proba(x)).T
            px_ = np.asarray(clf.predict_proba(x_)).T
            x = x.assign(l21=px[0],l22=px[1],l23=px[2])
            x_ = x_.assign(l21=px_[0],l22=px_[1],l23=px_[2])
            clf.fit(x,y)
            px = np.asarray(clf.predict_proba(x)).T
            px_ = np.asarray(clf.predict_proba(x_)).T
            x = x.assign(l31=px[0],l32=px[1],l33=px[2])
            x_ = x_.assign(l31=px_[0],l32=px_[1],l33=px_[2])
            clf.fit(x,y)
            px = np.asarray(clf.predict_proba(x)).T
            px_ = np.asarray(clf.predict_proba(x_)).T
            x = x.assign(l41=px[0],l42=px[1],l43=px[2])
            x_ = x_.assign(l41=px_[0],l42=px_[1],l43=px_[2])
            clf.fit(x,y)
            px = np.asarray(clf.predict_proba(x)).T
            px_ = np.asarray(clf.predict_proba(x_)).T
            x = x.assign(l51=px[0],l52=px[1],l53=px[2])
            x_ = x_.assign(l51=px_[0],l52=px_[1],l53=px_[2])
            clf.fit(x,y)
            px = np.asarray(clf.predict_proba(x)).T
            px_ = np.asarray(clf.predict_proba(x_)).T
            x = x.assign(l61=px[0],l62=px[1],l63=px[2])
            x_ = x_.assign(l61=px_[0],l62=px_[1],l63=px_[2])
            clf.fit(x,y)
            stock = data_.Code.values
            returnm3 = data_.loc[:,'Return_m3'].tolist()
            turnoverd5 = data_.loc[:,'TurnOver_D_m5'].tolist()
            Pred = clf.predict(x_)
            depam = clf.get_params(True)
            #print(depam)
            ayzsz,ayzsd,ayzsp,aydsz,aydsd,aydsp,aypsz,aypsd,aypsp,ayz,ayd,ayp=self.get_rate(y_,Pred,stock,returnm3,turnoverd5,dadates[i],fw)
            print(len(x))
            print(len(y))
            print(len(y_))
            if ayz == 0:
                ayz = 0.5
            if ayd == 0:
                ayd = 0.5
            if ayp == 0:
                ayp = 0.5
            yzsz += ayzsz
            yzsd += ayzsd
            yzsp += ayzsp
            ydsz += aydsz
            ydsd += aydsd
            ydsp += aydsp
            ypsz += aypsz
            ypsd += aypsd
            ypsp += aypsp
            a1 = float(ayzsz)/ayz
            a2 = float(ayzsd)/ayz
            a3 = float(ayzsp)/ayz
            a4 = float(aydsz)/ayd
            a5 = float(aydsd)/ayd
            a6 = float(aydsp)/ayd
            a7 = float(aypsz)/ayp
            a8 = float(aypsd)/ayp
            a9 = float(aypsp)/ayp
            yzsz_r += a1
            yzsd_r += a2
            yzsp_r += a3
            ydsz_r += a4
            ydsd_r += a5
            ydsp_r += a6
            ypsz_r += a7
            ypsd_r += a8
            ypsp_r += a9
            sums += 1
            print(str(dadates[i])+',zdp,|'+str(ayzsz)+','+str(a1)+','+str(ayzsd)+','+str(a2)+','+str(ayzsp)+','+str(a3)+'|,|'+str(aydsz)+','+str(a4)+','+str(aydsd)+','+str(a5)+','+str(aydsp)+','+str(a6)+'|,|'+str(aypsz)+','+str(a7)+','+str(aypsd)+','+str(a8)+','+str(aypsp)+','+str(a9))
        yzsz_r /= sums
        yzsd_r /= sums
        yzsp_r /= sums
        ydsz_r /= sums
        ydsd_r /= sums
        ydsp_r /= sums
        ypsz_r /= sums
        ypsd_r /= sums
        ypsp_r /= sums
        print()
        print('zong zdp |'+str(yzsz_r)+','+str(yzsd_r)+','+str(yzsp_r)+'|,|'+str(ydsz_r)+','+str(ydsd_r)+','+str(ydsp_r)+'|,|'+str(ypsz_r)+','+str(ypsd_r)+','+str(ypsp_r))
        print('zong num zdp |'+str(yzsz)+','+str(yzsd)+','+str(yzsp)+'|,|'+str(ydsz)+','+str(ydsd)+','+str(ydsp)+'|,|'+str(ypsz)+','+str(ypsd)+','+str(ypsp))
        allyz = yzsz + yzsd + yzsp
        allyd = ydsz + ydsd + ydsp
        allyp = ypsz + ypsd + ypsp
        zzr = float(yzsz)/allyz
        zdr = float(yzsd)/allyz
        zpr = float(yzsp)/allyz
        dzr = float(ydsz)/allyd
        ddr = float(ydsd)/allyd
        dpr = float(ydsp)/allyd
        pzr = float(ypsz)/allyp
        pdr = float(ypsd)/allyp
        ppr = float(ypsp)/allyp
        print('all number rate: zdp |'+str(zzr)+','+str(zdr)+','+str(zpr)+'|,|'+str(dzr)+','+str(ddr)+','+str(dpr)+'|,|'+str(pzr)+','+str(pdr)+','+str(ppr))
        fw.close()

    def get_rate(self,Actl,Pred,stock,returnm3,turnoverd5,date,fw):
        num = len(Pred)
        ayz = 0
        ayd = 0
        ayp = 0
        ayzsz = 0
        ayzsd = 0
        ayzsp = 0
        aydsz = 0
        aydsd = 0
        aydsp = 0
        aypsz = 0
        aypsd = 0
        aypsp = 0
        for i in range(num):
            fw.write(stock[i]+','+date+','+str(Actl[i])+','+str(Pred[i])+','+str(turnoverd5[i])+','+str(returnm3[i])+'\n')
            if Pred[i] == 1:
                ayz += 1
                if Pred[i] == Actl[i]:
                    ayzsz += 1
                elif Actl[i] == 2:
                    ayzsd += 1
                else:
                    ayzsp += 1
            elif Pred[i] == 2:
                ayd += 1
                if Pred[i] == Actl[i]:
                    aydsd += 1
                elif Actl[i] == 1:
                    aydsz += 1
                else:
                    aydsp += 1
            elif Pred[i] == 3:
                ayp += 1
                if Pred[i] == Actl[i]:
                    aypsp += 1
                elif Actl[i] == 1:
                    aypsz += 1
                else:
                    aypsd += 1
            else:
                print('hehe')
        return ayzsz,ayzsd,ayzsp,aydsz,aydsd,aydsp,aypsz,aypsd,aypsp,ayz,ayd,ayp
if __name__ == '__main__':
    is_read = False
    begd = '2017-08-01'
    endd = '9999-12-30'
    is_trans = True
    today = '20180521'
    today = time.strftime("%Y%m%d",time.localtime(time.time()))
    tbegd = '2018-01-02'
    tendd = '9999-12-30'
    params = {
        'random_state':10,
        'n_estimators':300,
        'n_jobs':-1,
        'max_depth':20,
        #'class_weight':'balanced'
        #'class_weight':{1:1,2:0.01,3:1}
        #'class_weight':{1:5,2:1,3:1}
        #'class_weight':{{0:1,1:5},{0:1,1:1},{0:1,1:1}}
    }
    days = 30
    fname='/home/wang1/YWork/data/newths/ths5_d19_ZZ500_3C_XG_623dims_online.csv'
    fstore = '/home/wang1/YWork/newlabel/newths/res/ths_dp2.csv'
    fhead = '/home/wang1/YWork/data/svmdata/head'
    fw = open(fstore, 'w')
    fw.write(str(params)+',days:'+str(days)+'\n')
    fw.write('Code,TRADE_DT,Actual,Predict,TurnOver_D_m5,Return_m3\n')
    print('read data')
    rfst = RFST()
    head, tite = rfst.get_head(fhead)
    dadates, dall1  = rfst.get_data(fname, is_read, begd, endd, head, tite)
    print('predict...')
    rfst.get_pred(dadates,dall1,days,fw,params,tbegd,tendd,is_trans)
    print('done!')
