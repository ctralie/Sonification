import librosa
from Viterbi import *
from sklearn.decomposition import PCA

if __name__ == '__main__':
    hop = 512
    x, sr = librosa.load("lincoln1.mp3")
    S = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop, htk=True).T

    pca = PCA(n_components=2)
    S = pca.fit_transform(S)

    plt.scatter(S[:, 0], S[:, 1])
    plt.show()