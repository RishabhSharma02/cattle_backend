import os
import subprocess
import cv2
import random
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import layers, metrics
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input


def get_encoder(input_shape):
    """ Returns the image encoding model """
    encode_model = Sequential([
        layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = (128, 128, 3)),
        layers.Conv2D(32, kernel_size = 3, activation='relu'),
        layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size = 3, activation='relu'),
        layers.Conv2D(64, kernel_size = 3, activation='relu'),
        layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'),
        layers.Conv2D(128, kernel_size = 4, activation='relu'),    
    layers.GlobalAveragePooling2D(),    
    layers.Dense(250)
    ], name="Encode_Model")    
    return encode_model

class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # Calculating Euclidean distance
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
    
class TripletDistanceLayer(layers.Layer):
    # A layer to compute the mean of cosine and Euclidean distances
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # Calculating cosine distance
    def cosine_distance(self, anchor, positive, negative):
        ap_cosine_distance = tf.keras.losses.cosine_similarity(anchor, positive, axis=-1)
        an_cosine_distance = tf.keras.losses.cosine_similarity(anchor, negative, axis=-1)
        return (ap_cosine_distance, an_cosine_distance)
    
    # Calculating the mean of cosine and Euclidean distances
    def call(self, anchor, positive, negative):
        ap_euclidean_distance, an_euclidean_distance = DistanceLayer()(anchor, positive, negative)
        ap_cosine_distance, an_cosine_distance = self.cosine_distance(anchor, positive, negative)
        ap_mean_distance = tf.reduce_mean([ap_euclidean_distance, ap_cosine_distance], axis=0)
        an_mean_distance = tf.reduce_mean([an_euclidean_distance, an_cosine_distance], axis=0)
        return (ap_mean_distance, an_mean_distance)

def get_siamese_network(input_shape=(128, 128, 3)):
    encoder = get_encoder(input_shape)
    
    # Input Layers for the images
    anchor_input = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")
    
    # A layer to compute the mean of cosine and Euclidean distances
    distances = TripletDistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )
    
    # Creating the Model
    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="Siamese_Network"
    )
    return siamese_network

siamese_network = get_siamese_network()

class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        
        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")
        self.acc_tracker = metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            acc = self._compute_acc(data)
            
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        acc = self._compute_acc(data)
        
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin,0.0)
        return loss
    
    def _compute_acc(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        accuracy = tf.reduce_mean(tf.cast(ap_distance < an_distance, dtype=tf.float32))
        return accuracy

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker, self.acc_tracker]
    
siamese_triplet = SiameseModel(siamese_network)

siamese_triplet.load_weights('trained_ML_files/siamese_model-final')  #All siamsese trained models are kept here "trained_ML"

def extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder

encoder = extract_encoder(siamese_triplet)    

def classify_images(face_list1, face_list2, threshold=9):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction

def run_verification(path_anchor, path_positive, path_negative):
    # Code Block 2
    def read_image_test(path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        return image

    # Code Block 4
    input_shape = (128, 128, 3)
    encoder = get_encoder(input_shape)

    # Code Block 7
    siamese_model = SiameseModel(siamese_network)

    # Code Block 8
    anchor = []
    positive = []
    negative = []

    anchor.append(read_image_test(path_anchor))
    positive.append(read_image_test(path_positive))
    negative.append(read_image_test(path_negative))

    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)

    anchor = preprocess_input(anchor)
    positive = preprocess_input(positive)
    negative = preprocess_input(negative)

    # Code Block 13
    pred1 = classify_images(anchor, positive)
    pred2 = classify_images(anchor, negative)

    if pred1 == 0:
        return "Verified"
    else:
        return "Not verified"


def process_image(image_path):
    if image_path.lower().endswith((".jpg", ".png")):
        command = f"python filenew.py --weights yolo_points_muzzle.pt --source {image_path}"
        subprocess.run(command, shell=True)
    else:
        print(f"Invalid image file: {image_path}")


# Your existing code here

app = FastAPI()

@app.post("/verify")
async def verify_images(anchor: UploadFile, positive: UploadFile):
    try:
        # Save uploaded images to temporary files
        anchor_path = "temp_anchor.jpg"
        positive_path = "temp_positive.jpg"
        negative_path = 'dummy_image/IMG_0161.JPG'

        with open(anchor_path, "wb") as anchor_file:
            anchor_file.write(anchor.file.read())

        with open(positive_path, "wb") as positive_file:
            positive_file.write(positive.file.read())

        #perform detection and cropping
        process_image(anchor_path)
        process_image(positive_path)

        new_anchor_path = "test_images/temp_anchor_crop.jpg"
        new_positive_path = "test_images/temp_positive_crop.jpg" 

        # Perform verification
        result = run_verification(new_anchor_path, new_positive_path, negative_path)

        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)