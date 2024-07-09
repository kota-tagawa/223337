import matplotlib.pyplot as plt
import numpy as np
import math
import sys

def random(a,myu,sigma):
    rnd = np.random.normal(myu,sigma)
    return a+rnd


def main(N,out_name):
    pi = math.pi
    myu = 0
    sigma = 3.0 

    X = []
    Y = []
    X_e = []
    Y_e = []
    
    for i in  range(0,N-1):
        sheta = -(pi/4)+((11*pi)/(12*N)*i)
        x = 300*math.cos(sheta)
        y = 200*math.sin(sheta)    
        X.append(x)
        Y.append(y)
        X_e.append(random(x,myu,sigma))
        Y_e.append(random(y,myu,sigma))

    plt.plot(X_e,Y_e,'.',color="Blue",label="random")
    plt.plot(X,Y,'.',color="Red",label="true")
    # plt.xlim(min(X)+10,max(X)+10) #x軸の範囲を1~5に指定
    # plt.ylim(min(Y)+10,max(Y)+10) #x軸の範囲を1~5に指定
    plt.grid() #グリッド
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("out/"+out_name)
    plt.show()
    
if __name__ == "__main__":
    args = sys.argv
    if len(args)!=3:
        print('Arguments are ',len(args))
    else:
        N = int(sys.argv[1])
        out_name = str(sys.argv[2])
        main(N,out_name)