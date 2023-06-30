### Introduction

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D** **`[x, y]` **or 3D*`[x, y, visible]` coordinates.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

### YOLOV8 Pose

How to use YOLOv8 pretrained Pose models?

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg') 

# Extract keypoint
result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
```


### Exploring Ouput Keypoint![](https://cdn-images-1.medium.com/max/800/1*PM5Q-58eNOWdoLogVKCGnQ.png)

source : https://learnopencv.com/wp-content/uploads/2021/05/fix-overlay-issue.jpg

In the output of YOLOv8 pose estimation, there are no keypoint names. Here’s sample output

![](https://cdn-images-1.medium.com/max/800/1*Om_wkVg8tv0ou1BN1tQl_Q.png)

To obtain the x, y coordinates by calling the keypoint name, you can create a Pydantic class with a “keypoint” attribute where the keys represent the keypoint names, and the values indicate the index of the keypoint in the YOLOv8 output.

```python
from pydantic import BaseModel

class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

# example 
get_keypoint = GetKeypoint()
nose_x, nose_y = result_yolov8[get_keypoint.NOSE]
left_eye_x, left_eye_y = keypoint[get_keypoint.LEFT_EYE]
```


### Generate Dataset Keypoint

To classify keypoints, you need to create a keypoint dataset. If you are using images from a public dataset on Kaggle [yoga-pose-classification](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification). This dataset have 5 classes Downdog, Goddess, Plank, Tree, Warrior2. I will run pose estimation YoloV8 on each image and extract the output. I extracted the keypoints for each body part to obtain the x, y coordinates, and then I saved them in CSV format.

![](https://cdn-images-1.medium.com/max/800/1*SBXgggGPWHPnoVCz9pLV6Q.png)
column of dataset

![](https://cdn-images-1.medium.com/max/800/1*KMwo_Htgmmi0DAJFLRZY3g.png)
sample dataset


### Train Classification

Let’s proceed with training a multi-class classification model for keypoints using the PyTorch library for neural networks.

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
  
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

hidden_size = 256
model = NeuralNet(X_train.shape[1], hidden_size, len(class_weights))
```


The neural network architecture consists of two linear layers and a ReLU activation function:

* `self.l1 = nn.Linear(input_size, hidden_size)`: The first linear layer, which takes the input features and maps them to the hidden layer.
* `self.relu = nn.ReLU()`: The activation function, which applies element-wise rectified linear unit (ReLU) activation to introduce non-linearity.
* `self.l2 = nn.Linear(hidden_size, num_classes)`: The second linear layer, which maps the hidden layer to the output classes.

`forward(self, x)` This method defines the forward pass of the neural network. It takes an input tensor** **`x` and returns the output tensor. The forward pass involves passing the input through the defined layers in sequence and returning the final output.

```python
learning_rate = 0.01
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


In this code,`learning_rate` is set to 0.01, which controls the step size during optimization. The `CrossEntropyLoss`criterion is used for multi-class classification, and the `weight` parameter is set to the class weights converted to a PyTorch tensor. This allows for handling class imbalance if present in the dataset.

The optimizer is defined as the Adam optimizer, which is a popular optimization algorithm for neural networks. It takes** `model.parameters()` as the input, which specifies the parameters of the model to be optimized. The **`lr` parameter sets the learning rate for the optimizer.


### Training Keypoint Result

The results are quite good for a simple Neural Network and the given dataset size, with an accuracy above 90%.

![](https://cdn-images-1.medium.com/max/800/1*2cgx6lQExwpRBL8FuWknoQ.png)
