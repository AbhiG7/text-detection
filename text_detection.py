import tensorflow as tf

mnist = tf.keras.datasets.mnist      # Getting the mnist dataset (huge dataset of written numbers)
(train_data, train_label), (test_data, test_label) = mnist.load_data()      # Splitting dataset into training and testing data

# Normalizing the data to make it easier and faster to compute
train_data = tf.keras.utils.normalize(train_data, axis=1)
test_data = tf.keras.utils.normalize(test_data, axis=1)

# Creating a basic feedforward model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),             # The input layer that flattens out the 28x28 matrix for each number
    tf.keras.layers.Dense(units=128, activation='relu'),      # A layer where all the neurons are connected between previous and next layers, more units = more neurons and more complex
    tf.keras.layers.Dense(units=128, activation='relu'),      # Second hidden layer that connects to one above it
    tf.keras.layers.Dense(units=10, activation='softmax')    # Output layer that has 10 neurons (one for each number), softmax will scale down all activations of neurons such that all add up to 1 and gives the probability of getting a number
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # Creates the model given everything above

model.fit(train_data, train_label, epochs=6)     # Trains the model created above, Epochs = how many times are we gonna run the model with the same data

# Evaluates the loss and accuracy of the model as we go and prints it out
loss, accuracy = model.evaluate(test_data, test_label)
print(accuracy)
print(loss)

model.save('digits_detect.model')       # Saves the model so I don't have to rerun it everytime I wanna use it