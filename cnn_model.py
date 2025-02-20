from keras.models import Model
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout, concatenate, BatchNormalization
from keras.regularizers import l2


class NewsCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        convs = []
        for filter_size in [3, 4, 5]:
            conv = Conv1D(256, filter_size, activation='relu',
                          kernel_regularizer=l2(0.001))(inputs)
            conv = BatchNormalization()(conv)
            pool = GlobalMaxPooling1D()(conv)
            convs.append(pool)

        merged = concatenate(convs, axis=1)
        dropout1 = Dropout(0.5)(merged)
        dense = Dense(128, activation='relu',
                      kernel_regularizer=l2(0.001))(dropout1)
        dropout2 = Dropout(0.5)(dense)
        outputs = Dense(self.num_classes, activation='softmax')(dropout2)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
