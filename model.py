"""
  This file used to build our model
"""
import pickle
try:
    import tensorflow as tf
    from tensorflow.keras import Model, optimizers, callbacks, layers
except ImportError:
    import keras
    from keras import Model, optimizers, callbacks, layers
    tf = keras

x_train = pickle.load(open('dir../after_aug/x_train', 'rb'))
y_train = pickle.load(open('dir../after_aug/y_train', 'rb'))
x_val = pickle.load(open('dir../after_aug/x_val', 'rb'))
y_val = pickle.load(open('dir../after_aug/y_val', 'rb'))
x_test = pickle.load(open('dir../after_aug/x_test', 'rb'))
y_test = pickle.load(open('dir../after_aug/y_test', 'rb'))

base_model = tf.keras.applications.Xception(include_top=False,
                                           weights='imagenet',
                                           input_tensor=None,
                                           input_shape=(299, 299, 3),
                                           pooling=None,
                                           classifier_activation="softmax",
                                           )
base_model.trainable = False

inputs = tf.keras.Input(shape=(299, 299, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(6, activation='softmax')(x)           # final layer

model = Model(inputs, outputs)

opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
acc_checkpoint = callbacks.ModelCheckpoint("dir../first_train/max_acc", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
loss_checkpoint = callbacks.ModelCheckpoint("dir../first_train/min_loss", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [acc_checkpoint, loss_checkpoint]

hist = model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val), batch_size=32, callbacks=callbacks_list)

with open('dir../first/hist', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)


base_model.trainable = True

opt = optimizers.Adam(learning_rate=0.00001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

acc_checkpoint = callbacks.ModelCheckpoint("dir../second_train/max_acc", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
loss_checkpoint = callbacks.ModelCheckpoint("dir../second_train/min_loss", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [acc_checkpoint, loss_checkpoint]

hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=32, callbacks=callbacks_list)

with open('dir../second/hist', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
