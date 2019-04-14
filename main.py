import numpy as np
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import *
from Load_samples import *
from Global_parameters import *
import tensorflow as tf
tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)


def Rate_func(temp):
    h, Vrf, Vn_input = temp
    Vn_input = tf.cast(Vn_input, tf.complex64)
    h = tf.cast(h, tf.complex64)
    num = list(h.shape)[0]
    l = 1 / tf.norm(Vrf, axis=[1, 2])
    v = tf.einsum('ijk, i->ijk', Vrf, l)
    hv = tf.matmul(h, v)
    hv_h = tf.transpose(hv, perm=[0, 2, 1], conjugate=True)
    t = tf.squeeze(P / Vn_input)
    temp2 = tf.einsum('ijk, i->ijk', tf.matmul(hv, hv_h), t)
    temp = tf.cast(
        tf.eye(Ns, batch_shape=[num]), tf.complex64) + temp2
    temp = tf.linalg.slogdet(temp)
    _, rate = tf.cast(temp, tf.float32) / tf.log(2.0)
    return -tf.cast(rate, tf.float32)


def just_ypred(y_true, y_pred):
    return y_pred


def trans_Vrf(temp):
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return vrf


path = "./dataset_path"
H, Hest = est_load(path)
H = H.astype('complex64')
Hest = Hest.astype('complex64')
print('load sample size:', H.shape[0])
H_est = np.concatenate([np.expand_dims(np.real(Hest), 1),
                        np.expand_dims(np.imag(Hest), 1)], 1)
# H_2 = np.expand_dims(H_2, 1)
H = np.squeeze(H)
# SNR = np.ones([H.shape[0], 1]) * 0
SNR = np.random.randint(-10, 10, [H.shape[0], 1])
Vn = P / (np.power(10, SNR / 10) * np.ones([H.shape[0], 1]))
CSI = Input(name='CSI', shape=(H_est.shape[1:4]), dtype=tf.float32)
est_CSI = Input(name='est_CSI', shape=(H.shape[1],), dtype=tf.complex64)
Vn_input = Input(shape=(1,), dtype=tf.float32)

temp = BatchNormalization()(CSI)
temp = Conv2D(
    filters=128,
    kernel_size=(
        2,
        3),
    activation='relu',
    data_format="channels_first")(temp)
temp = BatchNormalization()(temp)
temp = Conv2D(
    filters=64,
    kernel_size=(
        1,
        3),
    activation='relu',
    data_format="channels_first")(temp)


temp = Flatten()(temp)
temp = concatenate([temp, Vn_input])
temp = BatchNormalization()(temp)
out = Dense(Nt * Nrf)(temp)
out = Reshape((Nt, Nrf))(out)
out = Lambda(trans_Vrf, dtype=tf.complex64, output_shape=(Nt,))(out)
rate = Lambda(Rate_func, dtype=tf.float32,
              output_shape=(1,))([est_CSI, out, Vn_input])
model = Model(inputs=[CSI, est_CSI, Vn_input], outputs=rate)
model.compile(
    optimizer=tf.train.AdamOptimizer(
        learning_rate=0.001),
    loss=just_ypred)
model.summary()


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))


lh = LossHistory()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    './temp_trained.h5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='min',
    save_weights_only=True)

history = model.fit(x=[H_est,
                       H,
                       Vn],
                    y=H,
                    batch_size=512,
                    epochs=1000,
                    verbose=2,
                    validation_split=0.1,
                    callbacks=[checkpoint,
                               lh])
print(lh.losses_val)

# for testing

model.load_weights('./temp_trained.h5')
sio.savemat('1.mat', {'conv': lh.losses_val})
rate = []
for snr in range(-20, 2, 2):
    SNR = np.ones([H.shape[0], 1]) * snr
    Vn = P / (np.power(10, SNR / 10) * np.ones([H.shape[0], 1]))
    y = model.evaluate(x=[H_est, H, Vn], y=H, batch_size=10000)
    print(snr, y)
    rate.append(-y)
