import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy import linalg
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

# def sin(deg):
#     return math.sin(math.radians(deg))

# def cos(deg):
#     return math.cos(math.radians(deg))

def error(a,myu,sigma):
    rnd = np.random.normal(myu,sigma)
    #print(rnd)
    return a+rnd

def normalizeCO(x,y,f):
    V = np.array([[np.power(x, 2), x * y, 0, f * x, 0, 0],
              [x * y, np.power(x, 2) + np.power(y, 2), x * y, f * y, f * x, 0],
              [0, x * y, np.power(y, 2), 0, f * y, 0],
              [f * x, f * y, 0, np.power(f, 2), 0, 0],
              [0, f * x, f * y, 0, np.power(f, 2), 0],
              [0, 0, 0, 0, 0, 0]])
    return 4*V

def Mmat(xi,N):
    for i in range(0,N-1):
        print(xi[i])
        A = xi[i].reshape(6,1)
        M+=np.dot(A,A.T)
    M=M/N
    return M


def OLS(xi,N):
    M=0
    for i in  range(len(xi)):
        A = xi[i].reshape(6,1)
        #print(A)
        M+=np.dot(A,A.T)
    M=M/N
    value, vector = eigs(M, 1, which="SM")
    #print(value, vector)
    # index = np.argsort(value)
    # value = value[index]
    # vector = vector[:, index]
    return np.array(vector,dtype=np.float)

def MLE(xi,N,Vo):
    u=np.matrix(np.ones(6)).T
    W=1
    M=np.matrix(np.zeros((6, 6)),dtype=np.float)
    L=np.matrix(np.zeros((6, 6)),dtype=np.float)
    #print(N,len(xi),len(Vo))
    while True:
        for i in range(len(xi)):
            A = xi[i].reshape(6,1)
            Mo=np.dot(A,A.T)
            MLu=np.dot(u.T,np.dot(Vo[i],u))
            np.add(M,Mo/MLu)


            Lo=np.dot(xi[i],u)**2
            #print(Lo)
            #print(Vo[i])
            Lo=Lo[0,0] * Vo[i]
            np.add(L,Lo/(MLu**2))
        J=(M-L)/N
        u_old=u
        value, vector = eigs(J, 1, which="SM")
        u=np.matrix(vector)
        #print(u)
        if u.sum()<0:
            u=-u
        if np.linalg.norm(np.abs(u) - np.abs(u_old)) < 1:
            break
    # index = np.argsort(value)
    # value = value[index]
    # vector = vector[:, index]
    return np.array(vector,dtype=np.float)

def RMS(data,Pu):
    #data = data/np.linalg.norm(data)
    #true = true/np.linalg.norm(true)
    RSA=0
    vec=(1/np.linalg.norm(data))*data
    rsa = np.dot(Pu,vec)
    RSA = np.dot(rsa.T,rsa)
    #RSA_value =np.sqrt(RSA/len(data))
    return RSA

def KCR(xi,Vo,u):
    A = xi.reshape(6,1)
    M_1=np.dot(A,A.T)
    M_2=np.dot(u.T,np.dot(Vo,u))
    M=M_1/M_2
    return M

def main(N,out_name):
    pi = math.pi
    myu = 0
    f=1
    sigma_max=20

    true = np.array([1/np.power(300,2),0,1/np.power(200,2),0,0,-1]).T
    true = (true/np.linalg.norm(true)) 
    Pu=np.eye(6)- np.outer(true.T, true)
    #print(Pu)

    Ols=[]
    Mle=[]
    X = []
    Y = []
    ols=[]
    mle=[]
    xaxis= np.linspace(0.1, 2, sigma_max-1)
    M_bar=0
    kcr=0
    D_kcr=[]
    

    for i in  range(0,N):
            sheta = -(pi/4)+((11*pi)/(12*N)*i)
            x = 300*math.cos(sheta)
            y = 200*math.sin(sheta)    
            X.append(x)
            Y.append(y)

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
        loop=1000

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
            #print(xi)
            u_ols=(OLS(xi,N))
            u_mle=(MLE(xi,N,Vo))
            ols+=RMS(u_ols,Pu)
            mle+=RMS(u_mle,Pu)
            #print("ols:",ols)
            #print("mle:",mle)
            # xi_bar=np.array([math.pow(X[i],2),2*X[i]*Y[i],math.pow(Y[i],2),2*X[i]*f,2*Y[i]*f,math.pow(f,2)])
            # Vo_bar=normalizeCO(X[i],Y[i],f)*(sigma**2)
            # M_bar+=KCR(xi_bar,Vo_bar,true)

        ols_rms=np.sqrt(ols/loop)
        mle_rms=np.sqrt(mle/loop)
        Ols.append(ols_rms.reshape(1))
        Mle.append(mle_rms.reshape(1))
        #print(M_bar)

        # val, vec =np.linalg.eig(M_bar)
        # val = np.sort(val)[::-1]
        # for j in range(0,5):
        #     kcr+=1/val[j]
        # D_kcr.append(sigma*np.sqrt(kcr))

        #break


    #print("ols:",ols)
    # print("mle:",mle)  
    print("ols:",Ols)
    print("************************")
    print("mle:",Mle)
    print("************************")
    print("kcr:",D_kcr)
    #print(len(Mle))
    #print(len(xaxis))
    # print(Ols)
    # print(Mle)
    # print(xaxis)

    plt.plot(xaxis,Ols,color="Red",label="OLS")
    #plt.plot(xaxis,Mle,color="Blue",label="Mle")
    #plt.plot(xaxis,D_kcr,color="Green",label="KCR")
    plt.grid() #グリッド
    plt.legend()
    plt.xlabel('standard deviation')
    plt.ylabel('RMS-error')
    plt.savefig(out_name)
    plt.show()
    
if __name__ == "__main__":
    args = sys.argv
    if len(args)!=3:
        print('Arguments are ',len(args))
    else:
        N = int(sys.argv[1])
        out_name = str(sys.argv[2])
        main(N,out_name)