# red-greeNN-blue

## Running the MSE model

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
