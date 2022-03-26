import torch
def gini_score_fast(X, y, num_bins=10):
    gamma = torch.nn.functional.one_hot(y)
    G_F = torch.zeros((X.shape[1], ), dtype=torch.float64).t()
    for j in range(X.shape[1]):
        column = X[:, j]
        n = column.shape[0]
        hist, edges = torch.histogram(column, bins=num_bins)
        column.unsqueeze_(1)
        U_i = torch.zeros((len(edges)-1), gamma.shape[1])
        for i in range(U_i.shape[1]):
            count, _ = torch.histogram(column[gamma[:, i]==1], bins=edges)
            p_k_o = count / hist
            p_k_o = torch.nan_to_num(p_k_o)
            U_i[:, i] = p_k_o
        G_U = 1-torch.sum((torch.square(U_i)), axis=1)
        G_U = (hist/n) * G_U
        G_F[j] = torch.sum(G_U, axis=0)
    return G_F



def gini_score_fast_old(X, y, num_bins=10):
    # for older version
    # pytorch 1.8.1 and cuda 10.1
    gamma = torch.nn.functional.one_hot(y)
    G_F = torch.zeros((X.shape[1], ), dtype=torch.float64).t()
    for j in range(X.shape[1]):
        column = X[:, j]
        n = column.shape[0]
        min = torch.min(column)
        max = torch.max(column)
        # step = (max-min)/num_bins
        # edges = torch.empty(num_bins+1)
        # interval = torch.arange( min, max,  step)
        # print(edges.shape)
        # edges[:len(interval)] = interval
        # edges[-1] = max
        # print(edges)
        # hist, edges = torch.histogram(column, bins=num_bins)
        hist = torch.histc(column, bins=num_bins, min=min, max=max)
        # print(hist.shape)
        column.unsqueeze_(1)
        U_i = torch.zeros((num_bins), gamma.shape[1])
        for i in range(U_i.shape[1]):
            count = torch.histc(column[gamma[:, i]==1], bins=num_bins, min=min, max=max)
            p_k_o = count / hist
            p_k_o = torch.nan_to_num(p_k_o)
            U_i[:, i] = p_k_o
        G_U = 1-torch.sum((torch.square(U_i)), axis=1)
        G_U = (hist/n) * G_U
        G_F[j] = torch.sum(G_U, axis=0)
    return G_F

if __name__ == '__main__':
    x = torch.randn(100, 20)
    y = torch.ones(100, dtype=int)
    g_score = gini_score_fast_old(x, y)
    print(g_score.shape)