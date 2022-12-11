# red-greeNN-blue

## Running the model with the custom Multinomial Cross-Entropy loss function

_Note: the steps written here are with the context of the reduced CNN model, however, the steps would be exactly the same if the complete original CNN model (in `cnn_model.py`) was being used; we would just create an instance of the `CNNModel` class instead_.

First, you must get the data you want to use to train your model.
If, for example, you are using all of the inputs that were preprocessed by
`preprocessing.py`, this would look something like this:

```python
X = np.load("preprocessed_data/inputs.npy") # shape: (dataset_size, 150, 150)
y = np.load("preprocessed_data/labels.npy") # shape: (dataset_size, 150, 150, 3)
```

Next, you need to remove the _L_ channel values from `y` and add an extra axis
to the end of `X` (if the shapes match what we have above):

```python
X = tf.expand_dims(x, 3) # shape: (dataset_size, 150, 150, 1)
y = y[:,:,:,1:] # shape: (dataset_size, 150, 150, 2)
```

Next, you must instantiate an instance of the loss class (note: this takes a while since there are quite a few pre-emptive computations that take place here):

```python
loss_class = MultinomialCrossEntropyLoss(y)
```

Now, you can get, compile, and fit your model:

```python
model_class = ReducedCNNModel()
model = model_class.get_cnn_colorizer_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=loss_class.loss, run_eagerly=True)
model.fit(X_mini, y_mini, verbose=1, batch_size=10, epochs=5)
```

Once this is done, you can get colorized versions of new images like this:
```python
vals = np.load('bin_to_ab_array.npy') # Note: this np.load line does not need to be run repeatedly for each image you predict; just once before you predict on new images
image = some_image # assuming this is some black-and-white image with shape (1, 150, 150, 1)
output_img = model(image)
output_img = tf.reshape(output_img, shape=(150 * 150, model_class.classes_count))
h_result = model_class.h(output_img, vals)
h_result = tf.expand_dims(tf.reshape(h_result, shape=(150, 150, 2)), axis=0)
colorized_img = tf.concat([image, h_result], 3)
```

In order to display this image, you can run:

```python
def display_img(img):
    final_output_img_rgb = (color.lab2rgb(img) * 255).astype(np.uint8)
    pli_img = Image.fromarray(final_output_img_rgb)
    display(pli_img.convert('RGB') if pli_img.mode != 'RGB' else pli_img)

display_img(tf.squeeze(colorized_img))
```

<br>

## Running the model with Mean Squared Error (MSE) loss

First, you must get the data you want to use to train your model.
If, for example, you are using all of the inputs that were preprocessed by
`preprocessing.py`, this would look something like this:

```python
X = np.load("preprocessed_data/inputs.npy") # shape: (dataset_size, 150, 150)
y = np.load("preprocessed_data/labels.npy") # shape: (dataset_size, 150, 150, 3)
```

Next, you need to remove the _L_ channel values from `y` and add an extra axis
to the end of `X` (if the shapes match what we have above):

```python
X = tf.expand_dims(x, 3) # shape: (dataset_size, 150, 150, 1)
y = y[:,:,:,1:] # shape: (dataset_size, 150, 150, 2)
```

Now, you can get, compile, and fit your model:

```python
model_class = ReducedCNNModelMSE()
model = model_class.get_cnn_colorizer_model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=model_class.mse_loss)
model.fit(X, y, verbose=1, batch_size=100, epochs=5)
```

Once this is done, you can get colorized versions of new images like this:

```python
image = some_image # assuming this is some black-and-white image with shape (1, 150, 150, 1)
output_img = model(image)
colorized_img = tf.concat([image, output_img], 3)
```

In order to display this image, you can run:

```python
def display_img(img):
    final_output_img_rgb = (color.lab2rgb(img) * 255).astype(np.uint8)
    pli_img = Image.fromarray(final_output_img_rgb)
    display(pli_img.convert('RGB') if pli_img.mode != 'RGB' else pli_img)

display_img(tf.squeeze(colorized_img))
```
