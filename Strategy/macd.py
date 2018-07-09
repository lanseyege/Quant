import pandas as pd
import numpy as np
import json
import pylab
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression


def get_name(fname):
    fo = open(fname)
    line = fo.readline().strip()
    zz500 = line.split(',')
    return zz500

def get_data(fname):
    alldata = pd.read_csv(fname)
    dclose = np.array(alldata.iloc[:,5])
    dopen  = np.array(alldata.iloc[:,2])
    dstock = list(alldata.iloc[:,0])
    ddates = list(alldata.iloc[:,1])
    i = 0
    n = 0
    ls = []
    dcls = []
    stock = str(dstock[0])
    dsk = []
    dsk.append(stock)
    for ds in dstock:
        #print(str(ds))
        if str(ds) == stock:
            i += 1
        else:
            n += i
            ls.append(i)
            i = 1
            stock = str(ds)
            dsk.append(stock)
    ls.append(i)
    i = 0
    a = 0
    ddcls = []
    ddtes = []
    ddopn = []
    for i in range(len(ls)):
        A = dclose[a:a+ls[i]]
        O = dopen[a:a+ls[i]]
        B = ddates[a:a+ls[i]]
        C = [int(l) for l in B]
        ddtes.append(C)
        ddopn.append(O)
        dcls.append(A)
        ddcls.extend(A)
        a += ls[i]
    #print(dcls)
    return dclose, dopen, dstock, ddates, ls, dcls, ddcls, dsk, ddtes, ddopn

def get_ema(dclose, ls, t1 = 13, t2 = 27, t3 = 10):
    s = 0
    EMA12 = []
    EMA26 = []
    DIF   = []
    DEA   = []
    MACD  = []
    for ln in ls:
        A = []
        B = []
        C = []
        D = []
        M = []
        for i in range(ln):
            if i == 0:
                ema12 = dclose[i + s]
                ema26 = dclose[i + s]
                dif   = (ema12 - ema26)
                dea   = (ema12 - ema26) 
                macd  = (dif - dea) * 2
            else:
                ema12 = ema12 * (t1-2)/t1 + dclose[i + s] * 2 / t1
                ema26 = ema26 * (t2-2)/t2 + dclose[i + s] * 2 / t2
                dif   = ema12 - ema26
                dea   = dea * (t3-2)/t3 + dif *2 / t3
                macd  = (dif - dea) * 2
            A.append(ema12)
            B.append(ema26)
            C.append(dif)
            D.append(dea)
            M.append(macd)
        s += ln
        EMA12.append(A)
        EMA26.append(B)
        DIF.append(C)
        DEA.append(D)
        MACD.append(M)
    return DIF, DEA, MACD, EMA12, EMA26

def get_m(MACD, t1 = 13, t2 = 27, t3 = 10):
    n = len(MACD)
    s = 0
    EMA12 = []
    EMA26 = []
    DIF   = []
    DEA   = []
    MMACD = []

    for i in range(n):
        M = []
        A = []
        B = []
        C = []
        D = []
        for j in range(len(MACD[i])):
            if j == 0:
                ema12 = MACD[i][j]
                ema26 = MACD[i][j]
                dif   = (ema12 - ema26)
                dea   = (ema12 - ema26) 
                macd  = (dif - dea) * 2
            else:
                ema12 = ema12 * (t1-2)/t1 + MACD[i][j] * 2 / t1
                ema26 = ema26 * (t2-2)/t2 + MACD[i][j] * 2 / t2
                dif   = ema12 - ema26
                dea   = dea * (t3-2)/t3 + dif *2 / t3
                macd  = (dif - dea) * 2
            A.append(ema12)
            B.append(ema26)
            C.append(dif)
            D.append(dea)
            M.append(macd)
        #s += len(MACD[i])
        EMA12.append(A)
        EMA26.append(B)
        DIF.append(C)
        DEA.append(D)
        MMACD.append(M)
    return MMACD, EMA12, EMA26, DIF, DEA

def get_cal2(MACD, days, dopen, dclose, dstock, ddates):
    n = len(MACD)
    k = 0
    PMACD = []
    c2c1 = 0
    o2o1 = 0
    c2o1 = 0
    o2c1 = 0
    sumd = 0
    for i in range(n):
        m = len(MACD[i])
        x = np.arange(days).reshape(-1,1)
        PM  = []
        for j in range(days, m-1):
            model = make_pipeline(PolynomialFeatures(4), Ridge())
            model.fit(x, np.array(MACD[i][j-days:j]).reshape(-1, 1))
            #print(MACD[i][j-days:j])
            #print(MACD[i][j])
            pred1 = model.predict(np.array([days]).reshape(-1,1))
            #print(pred1[0][0])
            #pred2 = model.predict(np.array([days+1]).reshape(-1,1))
            PM.append(pred1[0][0])
            
        PMACD.append(PM)
        k += len(MACD[i])
        
    return PMACD

def get_store(MACD, EMA12, EMA26, DIF, DEA, TMACD, TEMA12, TEMA26, TDIF, TDEA, HMACD, HEMA12, HEMA26, HDIF, HDEA, dsk, fname, ddtes, dcls, ddopn):
    days = 7
    A = []
    print(len(MACD))
    print(len(dsk))
    for i in range(-days,0,1):
        B = {}
        for j in range(len(MACD)):
            B[dsk[j]] = [EMA12[j][i],EMA26[j][i],DEA[j][i],MACD[j][i],TEMA12[j][i],TEMA26[j][i],TDEA[j][i],TMACD[j][i],HEMA12[j][i],HEMA26[j][i],HDEA[j][i],HMACD[j][i],dcls[j][i],ddopn[j][i], 0, 0, 0]
        A.append(B)
    with open(fname, 'w') as fo:
        fo.write(json.dumps(A))

def get_cal4(MACD, days, dopen, dclose, dstock, ddates, fbuy, alpha, beta, ffo, fnos):
    fo = open(fbuy, 'w')
    fo.write('stock,date_buy,close_buy,open_buy,date_sell,close_sell,open_sell,days,c_buy/c_sell,ccrate,o_buy/o_sell,oorate,c_buy/o_sell,corate,o_buy/c_sell,ocrate\n')
    fs = open(fnos, 'w')
    fs.write('stock,date,close,open,date2\n')
    n = len(MACD)
    k = 0
    #PMACD = []
    c2c1 = 0
    o2o1 = 0
    c2o1 = 0
    o2c1 = 0
    sumd = 0
    sums = 0
    for i in range(n):
        m = len(MACD[i])
        x = np.arange(days).reshape(-1,1)
        kpp = 0
        krp = 0
        kpn = 0
        krn = 0
        #PM  = []
        s = 0
        d = 0
        das = ''
        l = 0
        A = []
        for j in range(days, m-1):
            model = make_pipeline(PolynomialFeatures(4),Ridge())
            model.fit(x, np.array(MACD[i][j-days:j]).reshape(-1,1))
            pred1 = model.predict(np.array([days]).reshape(-1,1))
            #if pred1 > 0 and MACD[i][j-1] < 0 :
            if pred1 < 0 and MACD[i][j-1] > 0 :
                if dclose[k+j-1]/dclose[k+j-3] >= alpha:
                    continue
                #if s != 0:
                #    continue
                s = 1
                l = j
                das = str(ddates[k+j])
                d = dclose[k+j]
                op = dopen[k+j]
                A.append([l,das,d,op])
            #elif pred1 < 0 and MACD[i][j-1] > 0 :
            elif pred1 > 0 and MACD[i][j-1] < 0 :
                if dclose[k+j-1]/dclose[k+j-3] <= beta:
                    continue
                if s != 1:
                    continue
                s = 0
                for ons in A:
                    fo.write(str(dstock[k+j])+','+ons[1]+','+str(ons[2])+','+str(ons[3])+','+str(ddates[k+j])+','+str(dclose[k+j])+','+str(dopen[k+j])+','+str(j-ons[0])+','+str(dclose[k+j]/ons[2])+','+str((dclose[k+j]/ons[2]-1)/(j-ons[0]))+','+str(dopen[k+j]/ons[3])+','+str((dopen[k+j]/ons[3]-1)/(j-ons[0]))+','+str(dclose[k+j]/ons[3])+','+str((dclose[k+j]/ons[3]-1)/(j-ons[0]))+','+str(dopen[k+j]/ons[2])+','+str((dopen[k+j]/ons[2]-1)/(j-ons[0]))+'\n')
                    sums += j-ons[0]
                    sumd += 1
                    c2c1 += (dclose[k+j]/ons[2]-1)/(j-ons[0])
                    o2o1 += (dopen[k+j]/ons[3]-1)/(j-ons[0])
                    c2o1 += (dclose[k+j]/ons[3]-1)/(j-ons[0])
                    o2c1 += (dopen[k+j]/ons[2]-1)/(j-ons[0])
                A = []

        for ons in A:
            fs.write(str(dstock[k+m-2])+','+ons[1]+','+str(ons[2])+','+str(ons[3])+','+str(ddates[k+m-2])+'\n')
        k += len(MACD[i])
    fo.close()
    ffo.write(str(alpha)+','+str(beta)+','+str(sumd)+','+str(c2c1/sumd)+','+str(o2o1/sumd)+','+str(c2o1/sumd)+','+str(o2c1/sumd)+','+str(1.0*sums/sumd)+'\n')

if __name__ == '__main__':
    fzz_name = 'zz500_name'
    fzz_name = '../data/regss/stockName'
    s = get_name(fzz_name)
    fname = '../data/regss/stock_data_price_qfq_20150102_20180428.csv'
    dclose, dopen, dstock, ddates, ls, dcls, ddcls, dsk, ddtes, ddopn = get_data(fname)
    DIF, DEA, MACD, EMA12, EMA26 = get_ema(dclose, ls)
    days = 7
    fjson = 'zzindex2.json'
    TMACD, TEMA12, TEMA26, TDIF, TDEA = get_m( MACD, 13, 27, 10)
    HMACD, HEMA12, HEMA26, HDIF, HDEA = get_m(TMACD, 13, 27, 10)
    PMACD = get_cal2(HMACD, days, dopen, dclose, dstock, ddates)
    fbuy = './mres/pred_h9.csv'
    fnos = './mres/no_sell9.csv'
    ffo = open('./mres/ave_h9.csv', 'w')
    ffo.write('buy_rm3,sell_rm3,counts,avec2c1,aveo2o1,avec2o1,aveo2c1,avedays\n')
    alpha = 1.01
    beta  = 1.08
    get_cal4(HMACD, days, dopen, dclose, dstock, ddates, fbuy, alpha, beta, ffo,fnos)
    ffo.close()
    '''
    al = [10.0, 1.04,1.03,1.02,1.01,1.0, 0.99, 0.98, 0.97, 0.96, 0.95]
    #al = [1.04, 1.03]
    be = [-10, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05,1.06,1.07,1.08,1.09]
    #be = [1.06,1.07,1.08,1.09]
    ffm = 'stk_ave_days.csv'
    dirs = './res/'
    unew = 'revsmacd/'
    dirs += unew
    if not os.path.isdir(dirs):
        os.mkdir(dirs)
    ffo = open(dirs+ffm, 'w')
    ffo.write('buy_rm3,sell_rm3,counts,avec2c1,aveo2o1,avec2o1,aveo2c1,avedays\n')
    for i in range(len(al)):
    #for i in range(1):
        for j in range(len(be)):
            fbuy = dirs+'stk_qfq_' + str(al[i]) + '_rmsell_'+str(be[j]) + '.csv'
            get_cal4(MACD, days, dopen, dclose, dstock, ddates, fbuy, al[i], be[j], ffo)
    ffo.close()
    '''
