from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Single fully-connected neural layer as encoder and decoder
df = pd.read_csv('my_small.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
original_dim= df.shape[1]-2
print(original_dim)
input_shape = (original_dim, )
intermediate_dim = int(original_dim/2)

use_regularizer = True
my_regularizer = None
my_epochs = 50

if use_regularizer:
    # add a sparsity constraint on the encoded representations
    # note use of 10e-5 leads to blurred results
    my_regularizer = regularizers.l1(10e-8)
    # and a larger number of epochs as the added regularization the model
    # is less likely to overfit and can be trained longer
    my_epochs = 100
    features_path = 'sparse_autoe_features.pickle'
    labels_path = 'sparse_autoe_labels.pickle'

# this is the size of our encoded representations
encoding_dim = int(original_dim/2)   # 32 floats -> compression factor 24.5, assuming the input is 784 floats

# this is our input placeholder; 784 = 28 x 28
input_img = Input(shape=input_shape)

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(original_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# Separate Decoder model

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Train to reconstruct MNIST digits

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# prepare input data
scaler = MinMaxScaler()
df_values = df.drop(columns=['word', 'tag'], axis=1)
df_norm = scaler.fit_transform(df_values)
tags=df['tag'].unique()
le = LabelEncoder()
le.fit(tags)
#print(le.classes_)
encoded_tags=le.transform(tags)
print(tags)
print(encoded_tags)
# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
x_train, x_test, y_train, y_test = train_test_split(df_norm, df_norm,
                                                    test_size=0.33, random_state=42)

# Train autoencoder for 50 epochs

autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test),
                verbose=2)

# after 50/100 epochs the autoencoder seems to reach a stable train/test lost value

# Visualize the reconstructed encoded representations

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)

decoded_imgs = decoder.predict(encoded_imgs)

e_df=encoder.predict(df_norm)


plt.figure(figsize=(8, 6), dpi=100 )
plt.scatter(e_df[:, 0], e_df[:, 1], c=le.transform(df['tag']))
plt.title('Autoencoder')
plt.colorbar()
plt.show()
K.clear_session()
