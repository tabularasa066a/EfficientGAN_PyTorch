"""参考
https://qiita.com/taku-buntu/items/0093a68bfae0b0ff879d
"""
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam  # ubuntuだと動かない？
from keras.utils import np_utils
import tensorflow as tf
import tensorflow
# from tensorflow.compat.v1.keras.backend import set_session
# from keras.backend import tensorflow_backend

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import rarfile as rar
from pathlib import Path

np.random.seed(0)
np.random.RandomState(0)
# tf.set_random_seed(0) #古い
tf.random.set_seed(0)

## 以下は最新tfでは不要になったかも
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
# session = tf.Session(config=config)
# tensorflow_backend.set_session(session)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[0]))
else:
    print("Not enough GPU hardware devices available")

## 学習用画像ファイルへのパス
# root_dir = str(Path('kill_me_baby_datasets').resolve())
root_dir = str(Path('DAGM2007/Normal').resolve())  # DAGM画像用

## kill_me_baby_datasets/.DS_Storeファイルが存在する場合、削除する(エラーで落ちるので)
DS_Store_isExists = os.path.exists(os.path.join(root_dir, '.DS_Store'))
if DS_Store_isExists:
    os.remove(os.path.join(root_dir, '.DS_Store'))

IMAGE_CHANNELS = 1  # 画像のチャンネル数(1: グレースケール, 3: color)
IMAGE_SIZE = 256  # 正方形画像の辺長さ
IMAGE_SIZE_HALF = int(IMAGE_SIZE / 2)
IMAGE_SIZE_QUARTED = int(IMAGE_SIZE / 4)
IMAGE_SIZE_DOBLE = int(IMAGE_SIZE * 2)

# IMAGES_4_TRAIN = 300  # 訓練に使う画像の枚数

class DCGAN():
    def __init__(self):

        self.class_names = os.listdir(root_dir)
        # class_names = ['botsu', 'yasuna&sonya&agiri', 'others', 'yasuna&sonya', 'yasuna&agiri', 'yasuna', 'sonya', 'agiri']が出力

        self.shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)  # サイズとチャンネル数(カラーなので3)
        self.z_dim = 100

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        ## generatorは単独では学習を行わない(combinedとして学習を行う)のでcompile不要
        # self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        z = Input(shape=(self.z_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        noise_shape = (self.z_dim,)

        model = Sequential()

        model.add(Dense(IMAGE_SIZE * IMAGE_SIZE_QUARTED * IMAGE_SIZE_QUARTED, activation="relu", input_shape=noise_shape))
        model.add(Reshape((IMAGE_SIZE_QUARTED, IMAGE_SIZE_QUARTED, IMAGE_SIZE)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(IMAGE_SIZE, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(IMAGE_SIZE_HALF, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(IMAGE_CHANNELS, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        img_shape = self.shape

        model = Sequential()

        model.add(Conv2D(IMAGE_SIZE_QUARTED, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(IMAGE_SIZE_HALF, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(IMAGE_SIZE, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(IMAGE_SIZE_DOBLE, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)  # Model(input, output)

    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])

        return model

    def train(self, iterations, batch_size=IMAGE_SIZE, save_interval=50, model_interval=1000, check_noise=None, r=5, c=5):
        ## r, c: 何枚の画像を一度に貼るか
        X_train = self.load_imgs()  # 今回は画像生成なので、キャラ名に対するラベル(labels)は不要

        half_batch = int(batch_size / 2)

        # 各画素の範囲を [0, 255] から [-1, 1] に変換する
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        for iteration in range(iterations):

            # ------------------
            # Training Discriminator
            # -----------------
            idx = np.random.randint(0, X_train.shape[0], half_batch)

            imgs = X_train[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))

            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            # Training Generator
            # -----------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))

            ## Generatorの学習では常に１(本物)というラベル付けを行う必要がある(成功体験を積ませる)
            # np.onesはbatck_size分の長さからなる1で構成された配列を作る
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))  # (input, label)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            model_dir = Path('ganmodels')
            model_dir.mkdir(exist_ok=True)
            if iteration % save_interval == 0:
                self.save_imgs(iteration, check_noise, r, c)
                start = np.expand_dims(check_noise[0], axis=0)
                end = np.expand_dims(check_noise[1], axis=0)
                resultImage = self.visualizeInterpolation(start=start, end=end)
                cv2.imwrite("images/latent/" + "latent_{}.png".format(iteration), resultImage)
                if iteration % model_interval == 0:
                    self.generator.save(str(model_dir)+"/dcgan-{}-iter.h5".format(iteration))

    def save_imgs(self, iteration, check_noise, r, c):
        noise = check_noise
        gen_imgs = self.generator.predict(noise)

        if IMAGE_CHANNELS == 1:  # グレースケールのとき
            gen_imgs = np.squeeze(gen_imgs)  # グレー画像用の配列の形にreshape

        # 0-1 rescale
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if IMAGE_CHANNELS == 1:
                    axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
                elif IMAGE_CHANNELS == 3:
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        # fig.savefig('images/gen_imgs/kill_me_%d.png' % iteration)
        fig.savefig('images/gen_imgs/generated_img_%d.png' % iteration)  # DAGM画像用

        plt.close()

    ## 画像をロードしてimagesに格納
    def load_imgs(self):
        img_paths = []
        labels = []
        images = []

        ## DAGM画像ファイル読み取り用
        for cl_name in self.class_names:
            img_names = os.listdir(os.path.join(root_dir, cl_name))  # 画像ファイルの名前を全て取得しimg_namesに格納
            for img_name in img_names:
                img_paths.append(os.path.abspath(os.path.join(root_dir, cl_name, img_name)))
            for img_path in img_paths:
                if IMAGE_CHANNELS == 1:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # グレースケール読み込み
                elif IMAGE_CHANNELS == 3:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
                images.append(img)

        # for cl_name in self.class_names:
        #     ## まずは画像のファイル名をimg_pathsに格納していく
        #     img_names = os.listdir(os.path.join(root_dir, cl_name))
        #     for img_name in img_names:
        #         img_paths.append(os.path.abspath(os.path.join(root_dir, cl_name, img_name)))

        #     ## 画像を読み込み、imagesに格納
        #     for img_path in img_paths:
        #         img = cv2.imread(img_path)
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #         images.append(img)  # データを配列に格納

        return np.array(images)

    def visualizeInterpolation(self, start, end, save=True, nbSteps=10):
        print("Generating interpolations...")

        steps = nbSteps
        latentStart = start
        latentEnd = end

        startImg = self.generator.predict(latentStart)
        endImg = self.generator.predict(latentEnd)

        vectors = []

        alphaValues = np.linspace(0, 1, steps)
        for alpha in alphaValues:
            vector = latentStart * (1 - alpha) + latentEnd * alpha
            vectors.append(vector)

        vectors = np.array(vectors)

        resultLatent = None
        resultImage = None

        for i, vec in enumerate(vectors):
            gen_img = np.squeeze(self.generator.predict(vec), axis=0)
            gen_img = (0.5 * gen_img + 0.5) * 255
            if IMAGE_CHANNELS == 1:
                gen_img = np.squeeze(gen_img)  # グレー画像用の配列の形にreshape
                interpolatedImage = gen_img
            elif IMAGE_CHANNELS == 3:
                interpolatedImage = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
            interpolatedImage = interpolatedImage.astype(np.uint8)
            resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage, interpolatedImage])

        return resultImage


if __name__ == '__main__':
    # datarar = rar.RarFile('kill_me_baby_datasets.rar')
    # datarar.extractall()  #rarファイルの解凍

    dcgan = DCGAN()
    r, c = 5, 5
    check_noise = np.random.uniform(-1, 1, (r * c, 100))
    dcgan.train(iterations=200000, batch_size=IMAGE_SIZE_QUARTED, save_interval=1000,model_interval=5000, check_noise=check_noise, r=r,c=c)
    # dcgan.train(iterations=10000, batch_size=32, save_interval=100,model_interval=500, check_noise=check_noise, r=r,c=c)

