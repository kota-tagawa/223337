import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from scipy import linalg
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

def error(a, myu, sigma):
    rnd = np.random.normal(myu, sigma)
    return a + rnd

def normalizeCO(x, y, f):
    V = np.array([[math.pow(x, 2), x * y, 0, f * x, 0, 0],
                  [x * y, math.pow(x, 2) + math.pow(y, 2), x * y, f * y, f * x, 0],
                  [0, x * y, math.pow(y, 2), 0, f * y, 0],
                  [f * x, f * y, 0, math.pow(f, 2), 0, 0],
                  [0, f * x, f * y, 0, math.pow(f, 2), 0],
                  [0, 0, 0, 0, 0, 0]])
    return 4 * V

def Mmat(xi, N):
    M = np.zeros((6, 6))  # Mの初期化を追加
    for i in range(0, N-1):
        A = xi[i].reshape(6, 1)
        M += np.dot(A, A.T)
    M = M / N
    return M

def OLS(xi, N):
    A = xi.reshape(6, 1)
    M = np.dot(A, A.T) / N
    value, vector = eigs(M, 1, which="SM")
    vector = vector.real  # 複素数の実数部分を取り出す
    return vector[:, 0]  # 形状を修正

def MLE(xi, N, Vo):
    u = np.matrix(np.ones(6)).T
    W = 1
    M = 0
    L = 0
    A = xi.reshape(6, 1)
    M_1 = np.dot(A, A.T)
    M_2 = np.dot(u.T, np.dot(Vo, u))
    M = M_1 / M_2

    L_11 = np.power(A.T, u)
    L_1 = L_11[0][0] * Vo
    L_2 = np.power(np.dot(u.T, np.dot(Vo, u)), 2)
    L = L_1 / L_2
    J = (M - L) / N
    value, vector = eigs(J, 1, which="SM")
    vector = vector.real  # 複素数の実数部分を取り出す
    return vector[:, 0]  # 形状を修正

def RMS(data, Pu, N):
    RSA = 0
    for i in range(0, N-1):
        vec = (1 / np.linalg.norm(data[i])) * data[i]
        rsa = np.dot(Pu, vec)
        RSA += rsa ** 2
    RSA_value = np.sqrt(RSA / (N-1))
    return RSA_value

def main(N, out_name):
    pi = math.pi
    myu = 0
    f = 1
    sigma_max = 30

    true = np.array([1 / np.power(300, 2), 0, 1 / np.power(200, 2), 0, 0, -1]).T
    true = (1 / np.linalg.norm(true) * true)
    Pu = np.eye(6) - np.outer(true, true.T)

    Ols = []
    Mle = []
    X = []
    Y = []
    xaxis = np.linspace(0.1, 2, sigma_max-1)

    for i in range(0, N-1):
        sheta = -(pi/4) + ((11 * pi) / (12 * N) * i)
        x = 300 * math.cos(sheta)
        y = 200 * math.sin(sheta)
        X.append(x)
        Y.append(y)

    for sigma in range(1, sigma_max, 1):
        sigma = sigma * 0.1
        ols = []
        mle = []

        for i in range(0, N-1):
            X_e = error(X[i], myu, sigma)
            Y_e = error(Y[i], myu, sigma)
            xi = np.array([math.pow(X_e, 2), 2 * X_e * Y_e, math.pow(Y_e, 2), 2 * X_e * f, 2 * Y_e * f, math.pow(f, 2)])
            Vo = normalizeCO(X_e, Y_e, f)
            ols.append(OLS(xi, N))
            mle.append(MLE(xi, N, Vo))

        Ols.append(RMS(np.array(ols), Pu, N))
        Mle.append(RMS(np.array(mle), Pu, N))

    plt.plot(xaxis, Ols, color="Red", label="OLS")
    plt.plot(xaxis, Mle, color="Blue", label="MLE")
    plt.grid()
    plt.legend()
    plt.xlabel('Standard Deviation (σ)')
    plt.ylabel('RMS Error')
    plt.savefig(out_name)
    plt.show()

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print('Arguments are ', len(args))
    else:
        N = int(args[1])
        out_name = str(args[2])
        main(N, out_name)
