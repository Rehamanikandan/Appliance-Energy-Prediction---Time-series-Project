import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,kpss,pacf
import statsmodels.api as sm
import seaborn as sns
from scipy import signal
np.random.seed(12345)
#Function to Calculate rolling mean and variance
def cal_rolling_mean_var(x,y):
    #calculating rolling mean and variance
    rolling_mean=[]
    rolling_variance=[]
    for i in range(1,len(x)+1):
        result=np.mean(x[:i])
        result_var=np.var(x[:i])
        rolling_mean.append(result)
        rolling_variance.append(result_var)
    print("Rolling mean:",rolling_mean)
    print("Rolling variance:",rolling_variance)
    print("*"*100)

    #Plotting mean
    plt.figure(figsize=(10,10))
    plt.plot(y,rolling_mean, color='red')
    plt.ylabel("Rolling mean")
    plt.xlabel('Time')
    plt.xticks(y[::1000],rotation='vertical')
    plt.title('Plot of Rolling mean')
    plt.show()
    #Plotting variance
    plt.figure(figsize=(10,10))
    plt.plot(y,rolling_variance, color="Purple")
    plt.xticks(y[::1000], rotation='vertical')
    plt.ylabel(f"Rolling variance")
    plt.xlabel('Time')
    plt.title('Plot of Rolling variance')
    plt.show()


#Function to perform ADF test
def ADF_Cal(x):
 result = adfuller(x)
 print("ADF Statistic: %f" %result[0])
 print('p-value: %f' % result[1])
 print('Critical Values:')
 for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

#Function to perform KPSS test.
def kpss_test(z):
    kpsstest = kpss(z, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

#First order differencing
def first_order_diff(l):
    diff=[]
    diff[0]=diff.append([np.nan])
    for j in range(1,len(l)):
        diff_val=l[j]-l[j-1]
        diff.append(diff_val)
    return Series(diff)

#Second order differencing
def second_order_diff(l):
    diff=[]
    diff[0]=diff.append([np.nan])
    diff[1]=diff.append([np.nan])
    for j in range(2,len(l)):
        diff_val=l[j]-2*l[j-1]+l[j-2]
        diff.append(diff_val)
    return Series(diff)

#Third order differencing
def third_order_diff(l):
    diff=[]
    diff[0]=diff.append([np.nan])
    diff[1]=diff.append([np.nan])
    diff[2]=diff.append([np.nan])
    for j in range(3,len(l)+2):
        diff_val=l[j]-l[j-1]
        diff.append((diff_val))
    return Series(diff)

#Function to calculate Corrrelation coefficient
def correlation_coefficient_cal(x,y):
    n=len(x)
    sum1=0
    sum2=0
    for i in x:
        sum1+=i

    meanx=sum1/n

    for j in y:
        sum2+=j

    meany=sum2/n
    diff=0
    deno1=0
    deno2=0
    for i,j in zip(x,y):
        diff+=(i-meanx)*(j-meany)
        deno1+=(i-meanx)**2
        deno2+=(j-meany)**2

    den=(deno1)**0.5
    den1=(deno2)**0.5
    r=diff/(den*den1)
    return r

#Auto correlation Function
def ACF(x,lags):
    acfmean=np.mean(x)
    T=len(x)
    list1=[]
    prod1=0
    deno1=0
    for t in range(0, T):
        deno1 += (x[t] - acfmean) ** 2

    for l in range(0,lags):
        for t in range(l,T):
            prod1+=(x[t]-acfmean)*(x[t-l]-acfmean)
        acf=float(prod1/deno1)
        prod1=0
        list1.append(acf)
    print("ACF:",list1)
    return list1


#Function to estimate variance
def estimated_var(x,k):
    sum=0
    for i in range(0,len(x)):
        sum+=i**2
    deno=1/(len(x)-k-1)
    var=np.sqrt((deno*sum))
    return var

#Function to calculate moving average
def movingaverage(data):
    m=int(input("Enter m:"))
    if m%2!=0:
       j = 0
       ma_list =[]
       while (j+m)!=len(data)+1:
           sum = 0
           mean=0
           for i in range(j,j+m):
               sum+=data[i]
           mean=sum/m
           ma_list.append(mean)
           j+=1
           k = int((m - 1) / 2)
       return ma_list,k

    elif m%2==0:
        fold=int(input("Folding order:"))
        if fold%2!=0:
            print("Invalid fold value(Fold should be even)")
        else:
            j=0
            ma_list = []
            while (j + m) != len(data) + 1:
                sum = 0
                mean = 0
                for i in range(j, j + m):
                    sum += data[i]
                mean = sum / m
                ma_list.append(mean)
                j += 1
            j=0
            final=[]
            while (j + fold) != len(ma_list) + 1:
                sum = 0
                mean = 0
                for i in range(j, j + fold):
                    sum += ma_list[i]
                mean = sum / fold
                final.append(mean)
                j += 1
            k=int(m/2)
            return final,k


#Function to generate arma process
def armaprocess_GPAC():
    T=int(input("Enter number of samples"))
    mean=int(input("Enter the mean of white noise"))
    var=int(input("Enter variance of white noise"))
    na=int(input("Enter AR process order"))
    nb=int(input("Enter MA process order"))
    naparam=[0]*na
    nbparam=[0]*nb
    for i in range(0,na):
        naparam[i]=float(input(f"Enter the coefficient of AR:a{i+1}"))
    for i in range(0,nb):
        nbparam[i]=float(input(f"Enter the coefficient of MA:b{i+1}"))
    ar=np.r_[1,naparam]
    ma=np.r_[1,nbparam]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    if mean==0:
        y=arma_process.generate_sample(T)
    else:
        mean_y = mean* (1 + np.sum(nbparam)) / (1 + np.sum(naparam))
        y = arma_process.generate_sample(T, scale=np.sqrt(var) + mean_y)
    return y

#Calculate Qvalue in the GPAC
def qvalue_cal(y,an,bn):
    deno=[]
    k=an
    j=bn
    for a in range(k):
        deno.append([])
        for b in range(k):
            deno[a].append(y[np.abs(j+b)])
        j=j-1
    ddeno=round(np.linalg.det(deno),5)
    j=bn
    num=deno[:k-1]
    num.append([])
    for a in range(k):
        num[k-1].append(y[j+a+1])
    dnum=round(np.linalg.det(num),5)
    if ddeno==0:
        return float('inf')
    else:
        qval=dnum/ddeno
        return round(qval,4)

#Function to align Q-values in GPAC
def GPAC(y,k,j):
    q=[]
    for b in range(j):
        q.append([])
        for a in range(1,k+1):
            q[b].append(qvalue_cal(y,a,b))
    gpac=np.array(q).reshape(j,k)
    gpactable=pd.DataFrame(gpac)
    c=np.arange(1,k+1)
    gpactable.columns=c
    print(gpactable)
    sns.heatmap(gpactable,annot=True)
    plt.xlabel('k')
    plt.ylabel('j')
    plt.title('GPAC table')
    plt.show()

#ACF plot
def acf_plot(lags,acf,samples):
    x = np.arange(0,lags)
    m = 1.96 / np.sqrt(len(samples))
    plt.stem(x,acf, linefmt='r-', markerfmt='bo', basefmt='b-')
    plt.stem(-1 *x,acf,linefmt='r-',markerfmt='bo', basefmt='b-')
    plt.title("ACF")
    plt.axhspan(-m,m,alpha=.2,color='yellow')
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    plt.show()

#Professor's ACF_PACF plot
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags):
 acf = sm.tsa.stattools.acf(y, nlags=lags)
 pacf = sm.tsa.stattools.pacf(y, nlags=lags)
 fig = plt.figure()
 plt.subplot(211)
 plt.title('ACF/PACF of the raw data')
 plot_acf(y, ax=plt.gca(), lags=lags)
 plt.subplot(212)
 plot_pacf(y, ax=plt.gca(), lags=lags)
 fig.tight_layout(pad=3)
 plt.show()

#LM algorithm
def WN(teta,na,y):
    numerator=[1]+list(teta[na:])
    denominator=[1]+list(teta[:na])
    if len(numerator)!=len(denominator):
        while len(numerator)<len(denominator):
            numerator.append(0)
        while len(denominator)<len(numerator):
            denominator.append(0)
    system=(denominator,numerator,1)
    t,e=signal.dlsim(system,y)
    e=[i[0] for i in e]
    return np.array(e)

def step0(na,nb):
    teta_o=np.zeros(shape=(na+nb,1))
    return teta_o.flatten()

def step1(delta,na,nb,teta,y):
    e_teta=WN(teta,na,y)
    SSE_O=np.dot(e_teta.T,e_teta)
    X=[]
    for i in range(na+nb):
        teta_delta = teta.copy()
        teta_delta[i]=teta[i]+delta
        en=WN(teta_delta,na,y)
        Xi=(e_teta-en)/delta
        X.append(Xi)
    Xfinal=np.transpose(X)
    A=np.dot(Xfinal.T,Xfinal)
    G=np.dot(Xfinal.T,e_teta)
    return A,G,SSE_O

def step2(A,G,mu,na,nb,teta,y):
    n=na+nb
    I=np.identity(n)
    dteta1=A+(mu*I)
    dteta_inv=np.linalg.inv(dteta1)
    delta_teta=np.dot(dteta_inv,G)
    teta_new=teta+delta_teta
    e=WN(teta_new,na,y)
    SSE_new=np.dot(e.T,e)
    if np.isnan(SSE_new):
        SSE_new=10**10
    return SSE_new,delta_teta,teta_new

def step3(max_iter,mu,delta,epsilon,mu_max,na,nb,y):
    num_iter=0
    teta=step0(na,nb)
    SSE=[]
    while num_iter<max_iter:
        A,G,SSE_O=step1(delta,na,nb,teta,y)
        if num_iter == 0:
            SSE.append(SSE_O)
        SSE_new,delta_teta,teta_new=step2(A,G,mu,na,nb,teta,y)
        SSE.append(SSE_new)
        if SSE_new<SSE_O:
            if np.linalg.norm(delta_teta)<epsilon:
                teta_hat=teta_new
                var=SSE_new/(len(y)-A.shape[0])
                A_inv=np.linalg.inv(A)
                cov=var*A_inv
                return SSE,cov,teta_hat,var
            else:
                teta=teta_new
                mu=mu/10
        while SSE_new>=SSE_O:
            mu=mu*10
            if mu>mu_max:
               print('Mu\'s maximum limit is exceeded')
               return None,None,None,None
            SSE_new, delta_teta, teta_new = step2(A, G, mu, na,nb, teta, y)
        num_iter+=1
        teta = teta_new
        if num_iter>max_iter:
            print('Maximum iterations exceeded')
            return None,None,None,None


with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
#Confidence interval
def conf_int(cov,params,na,nb):
    print("Confidence Interval:")
    for i in range(na):
        pos=params[i]+2*np.sqrt(cov[i][i])
        neg=params[i]-2*np.sqrt(cov[i][i])
        print(neg,f'<a{i+1}<',pos)
    for i in range(nb):
        pos=params[na+i]+2*np.sqrt(cov[na+i][na+i])
        neg=params[na+i]-2*np.sqrt(cov[na+i][na+i])
        print(neg,f'<b{i+1}<',pos)

#zero-poles cancellation
def zero_poles(params,na):
    y_den=[1]+list(params[:na])
    e_num=[1]+list(params[na:])
    zeros=np.roots(e_num)
    poles=np.roots(y_den)
    print("The roots of numerator are",zeros)
    print("The roots of denominator are",poles)

#Plot of SSE
def plotSSE(SSE):
    iter=np.arange(0,len(SSE))
    plt.plot(iter,SSE,label='SSE')
    plt.xlabel('Number of iterations')
    plt.ylabel('SSE')
    plt.title('SSE vs. #of Iterations')
    plt.legend()
    plt.show()

#Plot one step ahead prediction plot
def onestepplot(y,y_hat):
    plt.plot(y,label='Actual/Train')
    plt.plot(y_hat,label='one step predictions')
    plt.title('Plot of Actual/Train vs. One step prediction')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.show()

#Chi-square test
from scipy.stats import chi2
def chi_test(na,nb,lags,Q,e):
    DOF=lags-na-nb
    alpha=0.01
    chi_critical=chi2.ppf(1-alpha,DOF)
    if Q<chi_critical:
        print('The residuals are white')
    else:
        print('The residual is not white')
    lbvalue,pvalue=sm.stats.acorr_ljungbox(e,lags=[lags])
    print('From acorr_ljungbox test')
    print(lbvalue)
    print(pvalue)

#differencing Professor's code
def difference(y,interval=1):
    diff=[]
    for i in range(interval,len(y)):
        value=y[i]-y[i-interval]
        diff.append(value)
    return diff

#SARIMA simulation
def sarima_model():
    T=int(input('Enter number of samples'))
    mean=eval(input('Enter mean of white nosie'))
    var=eval(input('Enter variance of white noise'))
    na = int(input("Enter AR process order"))
    nb = int(input("Enter MA process order"))
    naparam = [0] * na
    nbparam = [0] * nb
    for i in range(0, na):
        naparam[i] = float(input(f"Enter the coefficient of AR:a{i + 1}"))
    for i in range(0, nb):
        nbparam[i] = float(input(f"Enter the coefficient of MA:b{i + 1}"))
    while len(naparam) < len(nbparam):
        naparam.append(0)
    while len(nbparam) < len(naparam):
        nbparam.append(0)
    ar = np.r_[1, naparam]
    ma = np.r_[1, nbparam]
    e=np.random.normal(mean,np.sqrt(var),T)
    system=(ma,ar,1)
    t,process=signal.dlsim(system,e)
    return process

#Base models
#average
def avg_one(x):
    train=[]
    for i in range(0,len(x)):
        mean=np.mean(x[0:i])
        train.append(mean)
    return train
def avg_hstep(train,test):
    forecast=np.mean(train)
    pred=[]
    for i in range(len(test)):
        pred.append(forecast)
    return pred


