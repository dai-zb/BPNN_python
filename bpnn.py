import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  # 多层线性回归


class BpNet(object):
    def __init__(self, net, alpha=0.1):
        self.net = net
        self.alpha = alpha  # 学习系数
        self.layer_num = len(net)
        self.weight = []  # 权重
        self.threshold = []  # 阈值
        self.h = []  # 节点值
        self.E_H = []  # 节点值 偏差值
        # self.E_I = []
        # self.delta_weight = []
        # self.delta_threshold = []

    def init_net(self):
        w_temp2 = np.zeros((self.net[0], 1))
        self.h.append(np.mat(w_temp2))
        self.E_H.append(np.mat(w_temp2))
        for i in range(1, self.layer_num):
            # w_temp0 = np.ones((net_nodes[i], net_nodes[i - 1]))
            # w_temp1 = np.ones((net_nodes[i], 1))
            prng = np.random.RandomState(14)  # 定义局部种子
            w_temp0 = prng.rand(net_nodes[i], net_nodes[i - 1])
            w_temp1 = prng.rand(net_nodes[i], 1)
            w_temp2 = np.zeros((net_nodes[i], 1))
            self.weight.append(np.mat(w_temp0))
            # self.delta_weight.append(np.mat(w_temp0))
            self.threshold.append(np.mat(w_temp1))
            # self.delta_threshold.append(np.mat(w_temp1))
            self.h.append(np.mat(w_temp2))
            self.E_H.append(np.mat(w_temp2))
            # self.E_I.append(np.mat(w_temp2))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def net_forward(self, h_begin):
        self.h[0] = np.mat(h_begin).T
        for i in range(1, self.layer_num):
            self.h[i] = np.dot(self.weight[i-1], self.h[i-1]) + self.threshold[i-1]
            self.h[i] = self.sigmoid(self.h[i])

    def net_backward(self, h_end):
        self.E_H[-1] = self.h[-1] - np.mat(h_end).T
        for i in range(self.layer_num - 1, 0, -1):
            # self.E_I[i-1] = np.multiply(self.E_H[i], np.ones(self.E_H[i].shape) - self.E_H[i])
            E_I = np.multiply(self.E_H[i], np.ones(self.h[i].shape) - self.h[i])
            # self.delta_weight[i-1] = np.dot(self.E_I[i-1], self.h[i-1].T)
            delta_weight = np.dot(E_I, self.h[i-1].T)
            # self.delta_threshold[i-1] = self.E_I[i-1]
            delta_threshold = E_I
            # self.E_H[i-1] = np.dot(self.weight[i-1].T, self.E_I[i-1])
            self.E_H[i-1] = np.dot(self.weight[i-1].T, E_I)
            self.weight[i-1] = self.weight[i-1] - self.alpha*delta_weight
            self.threshold[i-1] = self.threshold[i-1] - self.alpha*delta_threshold
        pass


if __name__ == '__main__':
    # net_nodes = [2, 4, 5, 4]  # 网络结构:输入 隐藏1 隐藏2 输出
    # bp = BpNet(net_nodes, 0.3)
    # bp.init_net()
    # for j in range(0, 1000):
    #     h0 = [2, 2]
    #     bp.net_forward(h0)
    #     h3 = [0.8, 0.6, 0.4, 0.2]
    #     bp.net_backward(h3)
    #     print(np.max(bp.E_H[-1]))
    # pass
    # h0 = [2, 5]
    # bp.net_forward(h0)
    # print(bp.h[-1])
    iris = datasets.load_iris()  # 导入数据集
    X = iris.data  # 获得其特征向量
    y = iris.target  # 获得样本label
    y = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    net_nodes = [4, 15, 3]  # 网络结构:输入 隐藏1 隐藏2 输出
    bp = BpNet(net_nodes, 0.5)
    bp.init_net()

    # for n in range(4):
    #     for i in range(len(y_train)):
    #         bp.net_forward(X_train[i])
    #         bp.net_backward(y_train[i])
    #         print(np.max(bp.E_H[-1]))
    #
    # T = 0
    # L = []
    # for j in range(len(y_test)):
    #     bp.net_forward(X_test[j])
    #     e = abs(bp.h[-1][0, 0] - y_test[j])
    #     L.append(e)
    #     if e < 0.25:
    #         T += 1
    #         print("T "+str(e))
    #     else:
    #         print("F "+str(e))
    # print(T/len(y_test))
    # print(sum(L)/len(L))

    for c in range(1, 100):
        for i in range(len(y_train)):
            bp.net_forward(X_train[i])
            y_tr = np.mat(np.zeros((1, 3)))
            y_tr[0, y_train[i]] = 1
            bp.net_backward(y_tr)
            # print(bp.h[-1])

        T = 0
        L = []
        E = []
        E1 = []
        I = []
        for j in range(len(y_test)):
            bp.net_forward(X_test[j])
            # print(bp.h[-1])
            p = np.argmax(bp.h[-1])  # get the index of max in the a
            if y_test[j] == p:
                T += 1
                L.append(bp.h[-1][p, 0]/sum(bp.h[-1]))
            else:
                E.append(y_test[j])
                E1.append(p)
                I.append(j)
        print(str(c)+" "+str(100*T/len(y_test))+" "+str(sum(L)/len(L)))
        print(str(E))
        print(str(E1))
        print(str(I))

    clf = MLPRegressor(solver='lbfgs',activation= 'logistic', hidden_layer_sizes=(15,), random_state=1)
    clf.fit(X_train, y_train)
    y_p = clf.predict(X_test)

    T = 0
    L = []
    E = []
    E1 = []
    I = []
    for j in range(len(y_test)):
        e = abs(y_test[j]-y_p[j])
        if e < 0.5:
            T += 1
            L.append(e)
        else:
            E.append(y_test[j])
            E1.append(p)
            I.append(j)
    print("sk-learn")
    print(str(100 * T / len(y_test)) + " " + str(sum(L) / len(L)))
    print(str(E))
    print(str(E1))
    print(str(I))

    pass
