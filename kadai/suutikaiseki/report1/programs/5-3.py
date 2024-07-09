import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy import linalg
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
import matplotlib.ticker as ptick

# def sin(deg):
#     return math.sin(math.radians(deg))

# def cos(deg):
#     return math.cos(math.radians(deg))

def error(a,myu,sigma):
    rnd = np.random.normal(myu,sigma)
    #print(rnd)
    return a+rnd

def normalizeCO(x,y,f):
    x = x
    y = y
    f0 = f
    xx = x**2
    yy = y**2
    xy = x*y
    f0x = f0*x
    f0y = f0*y
    f0f0 = f0**2


    cov = np.matrix([[xx,  xy,     0,   f0x,     0,    0], \
                   [xy,  xx+yy, xy,   f0y,   f0x,    0], \
                   [0,   xy,    yy,     0,   f0y,    0], \
                   [f0x, f0y,   0,    f0f0,    0,    0], \
                   [0,   f0x,   f0y,  0,     f0f0,   0], \
                   [0,   0,     0,    0,     0,      0]])

    #cov = cov
    return 4*cov

def Mmat(xi,Vo,u):
    A = xi.reshape(6,1)
    M_1=np.dot(A,A.T)
    M_2=np.dot(u.T,np.dot(Vo,u))
    M=M_1/M_2
    return M


def OLS(xi,N):
    M=0
    for i in  range(len(xi)):
        A = xi[i].reshape(6,1)
        #print(A)
        M+=np.dot(A,A.T)
    M=M/N
    value, vector = eigs(M, 1, which="SM")
    if vector.sum()<0:
        vector=-vector
    return np.array(vector,dtype=np.float)

def MLE(xi, N, Vo):
    u = np.matrix(np.ones(6)).T
    W = 1
    M = np.zeros((6, 6), dtype=np.float)
    L = np.zeros((6, 6), dtype=np.float)
    count = 0
    threshold = 1e-6  # 収束判定の閾値

    while True:
        M.fill(0)
        L.fill(0)

        for i in range(len(xi)):
            A = xi[i].reshape(6, 1)
            M_1 = np.dot(A, A.T).astype(np.float)
            M_2 = np.dot(u.T, np.dot(Vo[i], u)).astype(np.float) 
            M += M_1 / M_2

            L_11 = np.dot(xi[i], u) ** 2
            L_11 = L_11[0, 0] * Vo[i]
            L_11 = L_11.astype(np.float)  # L_11をnp.floatにキャスト
            L += L_11 / (M_2 ** 2)

        J = (M - L)
        u_old = u
        value, vector = eigs(J, 1, which="SM")
        u = np.matrix(vector)
        if u.sum() < 0:
            u = -u

        # 収束判定
        if np.linalg.norm(np.abs(u) - np.abs(u_old)) < threshold:
            break

        count += 1
        if count > 100:  # 最大反復回数を設定して無限ループを防止
            #print("Maximum iterations reached.")
            break

    return np.array(vector, dtype=np.float)

def RMS(data,Pu):
    #data = data/np.linalg.norm(data)
    #true = true/np.linalg.norm(true)
    RSA=0
    #vec=(1/np.linalg.norm(data))*data
    rsa = np.dot(Pu,data)
    RSA = np.dot(rsa.T,rsa)
    #RSA_value =np.sqrt(RSA/len(data))
    return RSA

def KCR(X,Y,u,f,sigma):
    M_bar=np.zeros((6, 6))
    for k in range(N):
        xi_bar=(np.array([math.pow(X[k],2),2*X[k]*Y[k],math.pow(Y[k],2),2*X[k]*f,2*Y[k]*f,math.pow(f,2)]))
        Vo_bar=((sigma**2)*normalizeCO(X[k],Y[k],f))
        M_bar+=Mmat(xi_bar,Vo_bar,u)
    val, vec =np.linalg.eig(M_bar)
    val = np.sort(val)[::-1]
    kcr=0
    for j in range(0,5):
        kcr+=1/val[j]
    return np.sqrt(kcr)

def main(N):
    pi = math.pi
    myu = 0
    f=1
    sigma_max=20
    xaxis= np.linspace(0.1, 2, sigma_max-1)

    true = np.array([1/np.power(300,2),0,1/np.power(200,2),0,0,-1]).T
    true = ((true/np.linalg.norm(true))).reshape(6,1)
    Pu=np.eye(6)- np.outer(true.T, true)
    #print(Pu)

    Ols=[]
    Mle=[]
    X = []
    Y = []
    ols=[]
    mle=[]
    
    M_bar=0
    
    D_kcr=[]
    

    for i in  range(0,N):
            sheta = -(pi/4)+((11*pi)/(12*N)*i)
            x = 300*math.cos(sheta)
            y = 200*math.sin(sheta)    
            X.append(x)
            Y.append(y)
    loop=1000

    for sigma in range(1,sigma_max,1):
        sigma = sigma*0.1
        print("********************"+str(sigma)+"**********************")
        #xaxis.append(sigma)
        X_e = []
        Y_e = []
        xi=[]
        Vo=[]
        ols=0
        mle=0




        for i in  range(loop):
            xi=[]
            Vo=[]
            for j in  range(0,N):
                X_e=error(X[j],myu,sigma)
                Y_e=error(Y[j],myu,sigma)
                xi .append( np.array([X_e**2,
                2 * X_e * Y_e,
                Y_e**2,
                2 * X_e * f,
                2 * Y_e * f,
                f**2])) 
                Vo.append(normalizeCO(X_e,Y_e,f))
                
            print(i)
            u_ols=(OLS(xi,N))
            #u_mle=(MLE(xi,N,Vo))
            ols+=RMS(u_ols,Pu)
            #mle+=RMS(u_mle,Pu)
            #print("ols:",ols)
            #print("mle:",mle)
            
            

        ols_rms=np.sqrt(ols/loop)
        #mle_rms=np.sqrt(mle/loop)
        Ols.append(ols_rms.reshape(1))
        #Mle.append(mle_rms.reshape(1))
        #print(M_bar)

        
        kcr=KCR(X,Y,true,f,sigma)
        D_kcr.append(kcr)

        #break

 
    print("ols:",Ols)
    print("************************")
    print("mle:",Mle)
    print("************************")
    print("kcr:",D_kcr)


    # fig, ax = plt.subplots()
    # plt.plot(xaxis,Ols,color="Red",label="OLS")
    # plt.plot(xaxis,Mle,color="Blue",label="Mle")
    # plt.xlabel('standard deviation')
    # plt.ylabel('RMS-error')
    # plt.legend()
    # ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
    # ax.ticklabel_format(style="sci", axis="y", scilimits=(-1,-6))
    # plt.grid() #グリッド
    # plt.savefig('kadai4.png')


    #plot RMS
    # fig, ax = plt.subplots()
    # plt.plot(xaxis,Ols,color="Red",label="LSM")
    # plt.plot(xaxis,Mle,color="Blue",label="Mle")
    # plt.plot(xaxis,D_kcr,color="Green",label="KCR")
    # plt.xlabel('standard deviation')
    # plt.ylabel('RMS-error')
    # ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
    # ax.ticklabel_format(style="sci", axis="y", scilimits=(-1,-6))
    # plt.grid() #グリッド
    # plt.legend()

    # plt.savefig("kadai5.png")

    fig, ax = plt.subplots()
    plt.plot(xaxis,Ols,color="Red",label="OLS")
    plt.plot(xaxis,D_kcr,color="Green",label="KCR")
    plt.xlabel('standard deviation')
    plt.ylabel('RMS-error')
    plt.legend()
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-1,-6))
    plt.grid() #グリッド
    plt.savefig('test.png')


if __name__ == "__main__":
    args = sys.argv
    if len(args)!=2:
        print('Arguments are ',len(args))
    else:
        N = int(sys.argv[1])
        main(N)