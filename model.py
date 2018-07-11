import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.utils import plot_model

from config import img_rows, img_cols, num_classes, channel, kernel


def build_model():
    # Encoder
    image_encoder = VGG19(input_shape=(img_rows, img_cols, channel), include_top=False, weights='imagenet',
                         pooling='None')
    # for layer in image_encoder.layers:
    #    layer.trainable = False
    input_tensor = image_encoder.inputs
    x = image_encoder.layers[-1].output

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (kernel, kernel), activation='elu', padding='same', name='deconv5_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (kernel, kernel), activation='elu', padding='same', name='deconv5_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (kernel, kernel), activation='elu', padding='same', name='deconv5_3',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (kernel, kernel), activation='elu', padding='same', name='deconv4_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (kernel, kernel), activation='elu', padding='same', name='deconv4_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (kernel, kernel), activation='elu', padding='same', name='deconv4_3',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (kernel, kernel), activation='elu', padding='same', name='deconv3_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (kernel, kernel), activation='elu', padding='same', name='deconv3_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (kernel, kernel), activation='elu', padding='same', name='deconv3_3',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (kernel, kernel), activation='elu', padding='same', name='deconv2_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernel, kernel), activation='elu', padding='same', name='deconv2_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (kernel, kernel), activation='elu', padding='same', name='deconv1_1',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernel, kernel), activation='elu', padding='same', name='deconv1_2',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='valid', name='pred',
                     kernel_initializer='he_normal',
                     bias_initializer='zeros')(x)

    model = Model(inputs=input_tensor, outputs=outputs, name="SegNet")
    return model


if __name__ == '__main__':
    encoder_decoder = build_model()
    # input_layer = model.get_layer('input')
    print(encoder_decoder.summary())
    plot_model(encoder_decoder, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
