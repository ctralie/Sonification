import librosa
from Viterbi import *
from sklearn.decomposition import PCA

if __name__ == '__main__':
    hop = 512
    x, sr = librosa.load("lincoln1.mp3")
    S = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop, htk=True).T

    pca = PCA(n_components=2)
    S = pca.fit_transform(S)    
    
    t = np.linspace(0, 1, 10000)
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