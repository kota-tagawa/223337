import matplotlib.pyplot as plt
import numpy as np
import math
import sys

def main(N,out_name):
    pi = math.pi
    A = []
    X = []
    Y = []
    for i in  range(0,N-1):
        theta = -(pi/4)+((11*pi)/(12*N)*i)
        x = 300*math.cos(theta)
        y = 200*math.sin(theta)    
        A.append([x,y])
        X.append(x)
        Y.append(y)
    
    plt.plot(X,Y,'.',color="Red")
    plt.xlim(min(X)+10,max(X)+10) #x軸の範囲を1~5に指定
    plt.ylim(min(Y)+10,max(Y)+10) #x軸の範囲を1~5に指定
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
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