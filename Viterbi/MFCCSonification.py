import librosa
from Viterbi import *
from sklearn.decomposition import PCA

if __name__ == '__main__':
    hop = 512
    x, sr = librosa.load("lincoln1.mp3")
    T = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop, htk=True).T

    pca = PCA(n_components=2)
    S = pca.fit_transform(T)    
    
    t = np.linspace(0, 1, 1000)
    Y = get2DFigure8(t)
    
    Y = Y*25
    Y[:,0] += 35
    Y[:,0] *= 1.5
    Y[:,1] -= 25
    
    plt.scatter(S[:, 0], S[:, 1])
    plt.scatter(Y[:, 0], Y[:, 1], c = t)
    plt.show()
    
    P = np.zeros(S.shape[0])
    X = do_viterbi(S, Y, P)
    G = S[X, :]
    plt.scatter(G[:, 0], G[:, 1], c = np.arange(G.shape[0]))
    plt.show()
    
    H = X*hop
    SL = 500
    PA = []
    for i in range(0,len(H),1):
        for j in range(0,SL,1):
            q = x[H[i]+j]
            PA.append(q)