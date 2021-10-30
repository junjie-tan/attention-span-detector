import numpy as np
from tensorflow import keras
from PIL import Image

attention_class = {0: 'Attentive', 1: 'Inattentive'}

# load model

resnet_model = keras.models.load_model(r'C:\Users\tjunj\PycharmProjects\AI_Case_Studies\resnet.h5')

img_path = r'C:\Users\tjunj\PycharmProjects\AI_Case_Studies\sample2.jpg'
img = Image.open(img_path)

#img_resized = tf.keras.applications.resnet50.preprocess_input(img)

img = img.resize((224, 224))
img_array = np.asarray(img)
img_resized = img_array.reshape(1,224,224,3)

prediction = resnet_model.predict(img_resized)[0]
output = int(np.round(prediction))

print(type(output))
print(prediction)
print(attention_class[output])
