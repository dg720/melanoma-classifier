import tensorflow as tf
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

model_file = "C:/Data-Sets/Skin-Lesion/best.keras"
model = tf.keras.models.load_model(model_file)
# print(model.summary())

input_shape = (64, 64)
batch = 32
categories = ["MEL", "NV", "BCC"]


def prepareImage(img):
    resized = cv2.resize(img, input_shape, interpolation=cv2.INTER_AREA)
    # imgResult = np.expand_dims(resized, axis=0)  # becomes (1, 64, 64, 3)
    imgResult = resized / 255.0
    return imgResult


def get_predictions(input_img):
    processed_img = np.expand_dims(prepareImage(input_img), axis=0)
    resultArray = model.predict(processed_img, batch_size=32)
    # answers = np.argmax(resultArray, axis=1)
    # text = categories[answers[0]]
    return resultArray[0]


# load the google image
# imgPath = "C:/Users/dhruv/Documents/melanoma_classifier/data/external/Basal-cell-carcinoma.jpg"
imgPath = "C:/Users/dhruv/Documents/melanoma_classifier/data/external/3.bmp"
img = cv2.imread(imgPath)

imgForModel = prepareImage(img)
get_predictions(imgForModel)

# run the prediction
resultArray = model.predict(imgForModel, batch_size=batch, verbose=1)
answers = np.argmax(resultArray, axis=1)

# print(answers)

text = categories[answers[0]]
print("Predicted answer is : " + text)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text, (0, 20), font, 0.5, (209, 19, 77), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

im2 = imgForModel.copy()
im2[:, :, 0] = imgForModel[:, :, 2]
im2[:, :, 2] = imgForModel[:, :, 0]
plt.figure(figsize=(8, 8))
plt.imshow(im2)
plt.show()

# LIME explainer
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    imgForModel,  # Input image
    get_predictions,  # Prediction function
    top_labels=3,  # Number of top classes to explain
    hide_color=0,  # Color to use for occluded segments
    num_samples=1000,  # Number of samples for LIME to generate
)

# Visualize explanation for the top predicted class
top_label = explanation.top_labels[0]
temp, mask = explanation.get_image_and_mask(
    label=top_label,
    positive_only=True,  # Show only features that positively contributed
    num_features=10,  # Number of superpixels to highlight
    hide_rest=False,  # Whether to hide non-contributing areas
)

plt.figure(figsize=(8, 8))
plt.imshow(mark_boundaries(temp, mask))
plt.title(f"Explanation for class: {top_label}")
plt.show()

# display LIME explanation
# temp, mask = lime_image.ImageExplanation.get_image_and_mask(lime_image.ImageExplanation.top_labels[0], positive_only=True, num_features=3,hide_rest=False)
