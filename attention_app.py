import numpy as np
from tensorflow import keras
import cv2
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

attention_class = {0: 'Attentive', 1: 'Inattentive'}

# load model

resnet_model = keras.models.load_model('resnet.h5')

# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

code_block_1 = '''

# denotes number of input nodes based on number of features in dataset
n_features = X_train.shape[1] 

# model has been tuned (see model tuning section for parameters tuned)
def create_model(n_features):
  model = Sequential()
  adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.add(Dense(100, activation='relu', input_shape=(n_features,)))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  
  model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

  return model

attention_model = create_model(n_features)

'''

code_block_2 = '''

def create_res_model():
  base_model = keras.applications.ResNet50V2(
      weights='imagenet',
      input_shape=(224, 224, 3),
      include_top=False
  )
  
  inputs = keras.Input(shape=(224, 224, 3))
  base_model.trainable = False

  x = base_model(inputs, training=False)
  x = keras.layers.GlobalAveragePooling2D()(x)
  outputs = keras.layers.Dense(1, activation='sigmoid')(x)

  model = keras.Model(inputs, outputs)

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  return model

res_model = create_res_model()

'''

facemesh_mapping_pic = Image.open('facemesh_mapping.jpg')
conv_diagram_pic = Image.open('conv_diagram.jpg')
three_strategies_pic = Image.open('three_strategies.jpg')
tuning_results_pic = Image.open('tuning_results.jpg')
resnet_results_pic = Image.open('resnet_results.jpg')
approach_1_pic = Image.open('approach_1.jpg')
approach_2_pic = Image.open('approach_2.jpg')

class Resnetvideo(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img_resized = img_resized.reshape(1,224,224,3)

        prediction = resnet_model.predict(img_resized)[0]
        result = int(np.round(prediction))
        output = attention_class[result]

        cv2.putText(img, output, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

def main():
    st.title("Attention Span Detection System in Real-Time using FaceMesh and ResNet in Python")
    st.caption("Authors: Tan Jun Jie and Justin Quek")
    st.caption("November 1, 2021")

    st.header("Motivation")
    st.write("With the advent of online learning, teachers and educators face a new challenge "
             "in monitoring students’ learning in online classes or home-based learning, as compared "
             "to physical classes. Computer vision could serve as an effective decision tool, helping "
             "to identify students who are consistently unmotivated.")
    st.write("In this post, we will explore how we can leverage a common and effective machine learning "
             "technique known as transfer learning to build an image recognition model capable of detecting "
             "attentive and inattentive behaviours.")

    st.header("What is transfer learning?")
    st.write("Training a machine learning model from scratch can be a long and arduous task. Often, this involves "
             "collecting massive amounts of data and running multiple trials with various hyperparameters to "
             "test which combination maximises accuracy and minimises loss for the task at hand.")
    st.write("With transfer learning, we are essentially truncating the model building process. We are utilising "
             "pre-existing models that others have trained and fine-tuned, and repurposing it to suit our task.  ")
    st.write("Transfer learning is typically used when the dataset we have is small, while the dataset that "
             "the pre-trained model was trained on is significantly larger. The task that we are trying to tackle "
             "should ideally be aligned with the task that the pre-trained model was trained on.")
    st.write("For instance, if we want to build a model that can classify dogs and cats (an image "
             "classification problem), then we should use a pre-trained model that was trained on images rather than "
             "one that was trained on speech for example. ")

    st.header("Transfer learning approaches")
    st.write("To start off, we should first be aware that a pre-trained model consists of two main parts: the "
             "convolutional base and the classifier, as shown below:")

    st.image(conv_diagram_pic, width=150)

    st.write("Repurposing the pre-trained model to suit the needs of our data may involve retraining one or both "
             "parts of the model. This can be done via 3 key strategies:")

    st.subheader("Train the entire model")
    st.write("This involves keeping the general architecture of the model, but training it according to our dataset. "
             "This process involves re-training the model from scratch, which means that whatever weights that have "
             "been previously tuned would be reset. ")
    st.write("This strategy naturally requires a large dataset and consequently, large computing power.")

    st.subheader("Train some layers and leave others frozen")
    st.write("Lower layers of a neural network typically extract general features which are problem independent, "
             "while higher layers typically extract specific features which are problem dependent.")
    st.write("This strategy involves balancing between training and freezing layers. If your dataset is small with a "
             "large number of parameters, you can freeze more layers to avoid overfitting. ")
    st.write("Conversely, if you have a large dataset with fewer parameters, you can afford to train more layers since "
             "overfitting is not an issue.")

    st.subheader("Freeze the entire convolutional base")
    st.write("The idea is to freeze the convolutional base in its original form to retain the weights that were "
             "previously tuned as we want to preserve the “learning” that was already done. ")
    st.write("The pre-trained model will be used as a fixed feature extraction mechanism, and a new classifier layer "
             "would be trained to suit our task. This strategy can be useful when short on computational power or when "
             "using a small dataset.")

    st.image(three_strategies_pic,
             caption="Source: https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751")

    st.header("Building the attention detection model")
    st.write("For our attention span detection model, we will be adopting Strategy 3 (freezing of convolutional base), "
             "due to our relatively small dataset of 1021 pictures and low computing power.")
    st.write("Using a dataset of 1021 images (512 inattentive images labelled ‘0’, 509 attentive images labelled ‘1’). "
             "The labels for the images were stored in a .csv file, and the dataset was split into training (60%), "
             "validation (20%) and testing (20%) datasets.")
    st.write("We will be utilising 2 pre-existing models for our task, FaceMesh and ResNet50V2 and covering 2 possible "
             "approaches to creating an attention span detection model:")
    st.write("__FaceMesh__: https://google.github.io/mediapipe/solutions/face_mesh.html")
    st.write("__ResNet50V2__: https://keras.io/api/applications/resnet/#resnet50v2-function")

    st.subheader("Approach 1: Feature extraction with FaceMesh ")

    st.image(approach_1_pic, width=400)

    st.write("In the first approach, FaceMesh will be used as a feature extractor. FaceMesh predicts the 3D "
             "coordinates of 468 landmarks on a human face, which amounts to a total of 1404 data points when "
             "accounting for x, y and z coordinates. This allows us to more accurately capture subtleties in human "
             "facial expressions. An example of feature expression on an image input can be found below:")

    st.image(facemesh_mapping_pic)

    st.write("After extracting the features from each image using FaceMesh, we built a simple convolutional neural "
             "network (CNN) classifier and fed the features in as input.")

    st.code(code_block_1, language="python")

    st.write("The resultant model had a decent accuracy in the range of 0.75-0.80. We then proceeded to fine-tune "
             "the model according to various parameters, such as the type of optimiser (Adam vs SGD), learning rates "
             "(0.001, 0.01, 1), number of hidden layers (2, 3, 4, 5) and number of nodes in each layer (50, 100, 150, 200). "
             "A summary of the comparative accuracy and loss scores is shown below.")

    st.image(tuning_results_pic)

    st.write("Finally, we concluded the most optimal set of parameters to be a model that utilised the Adam optimiser, "
             "along with a learning of 0.001, 3 hidden layers, with 150 nodes in each layer.")

    st.subheader("Approach 2: Image classification with ResNet50V2")

    st.image(approach_2_pic, width=400)

    st.write("In the second approach, the pre-trained model ResNet50V2 was used for our image classification task. "
             "ResNet50V2 is an artificial neural network with 50 layers. It is trained on the ImageNet dataset and "
             "consists of 1000 classes.")
    st.write("ResNet50V2 was first imported from Keras and the output layer of the ResNet50V2 model was removed. "
             "The weights of the ResNet50V2 model were frozen and a new output layer for our binary classification "
             "task was added. ")

    st.code(code_block_2, language='python')

    st.write("The resultant mode had an accuracy in the range of 0.85-0.95.")

    st.image(resnet_results_pic)

    st.header("Try out a demo!")
    st.write("Here we have a demo of a real time attention detection system using ResNet50V2 that will give a live "
             "prediction of whether you are attentive or not attentive.")
    st.write("Some things you can consider trying: shifting your body to one side, yawning, closing your eyes. ")

    webrtc_streamer(key="example", video_processor_factory=Resnetvideo)

    st.header("Conclusion")
    st.write("To further improve on our model, a broader and more diverse dataset could be used for training, so as "
             "to attain a better representation of any target population.")
    st.write("The inattentive dataset only consisted of two poses, namely yawning and closing of eyes. In the future, "
             "a greater number of poses could be used in the inattentive dataset, though we should caveat that the "
             "limitation of using FaceMesh would be that the poses used are limited to those which are frontal facing, "
             "where a person’s face can be detected by the camera.")
    st.write("Poses such as looking downwards or tilting one’s head to the sides would fare well during feature "
             "extraction and the landmarks obtained would be messy and inaccurate.")
    st.write("In addition, a time-based scoring system could also be implemented to monitor one’s attentiveness over "
             "a fixed period of time.")

if __name__ == "__main__":
    main()



