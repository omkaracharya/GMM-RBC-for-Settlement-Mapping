import os

import numpy as np
from PIL import Image
from sklearn import mixture
from sklearn.externals import joblib


def train_model():
    if not os.path.exists('models/gmm_model.pkl'):
        print("Training Model...")
        img = Image.open("images/city1.png")
        pix = img.load()
        row = img.size[0]
        col = img.size[1]

        counter = 0
        data = np.zeros([row * col, 3])

        for c in range(col):
            for r in range(row):
                data[counter] = list(pix[r, c])[:-1]
                counter += 1

        g = mixture.GaussianMixture(n_components=6)
        g.fit(data)
        print("Model trained.")

        joblib.dump(g, 'models/gmm_model.pkl')
        print("Model saved.")


def load_model():
    g = joblib.load('models/gmm_model.pkl')
    print("Model Loaded.")
    return g


def test_model(g):
    for file in os.listdir("images/"):
        print("images/" + file)
        img = Image.open("images/" + file)
        pix = img.load()
        row = img.size[0]
        col = img.size[1]

        counter = 0
        data = np.zeros([row * col, 3])

        for c in range(col):
            for r in range(row):
                data[counter] = list(pix[r, c])[0:3]
                counter += 1

        components = g.predict(data)

        # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (255, 255, 0), (0, 0, 0)]
        output = [tuple(np.asarray(g.means_[t]).astype(int)) for t in components]

        img2 = Image.new('RGB', [row, col])
        img2.putdata(output)
        img2.save("output/" + file)


def main():
    train_model()
    g = load_model()
    test_model(g)


if __name__ == '__main__':
    main()
