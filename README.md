## Federated Learning Experiment Report

### 1. Introduction

**Federated Learning (FL)** is a distributed machine learning approach where multiple clients collaboratively train a model under the orchestration of a central server, while keeping their data localized. This method enhances data privacy and security by minimizing data exchange.

**FedAvg** (Federated Averaging) is a foundational algorithm in FL where each client trains a local model on its data and sends the model updates to the central server. The server then averages these updates to produce a global model.

**FedProx** (Federated Proximal) is an extension of FedAvg designed to handle system heterogeneity. It introduces a proximal term in the local objective to restrict the local model updates, ensuring they do not deviate significantly from the global model.

### 2. Dataset and Data Partitioning

**CIFAR-10 Dataset**: CIFAR-10 is a widely used dataset for image classification tasks, consisting of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

**Data Partitioning Strategy**:
- **IID (Independent and Identically Distributed)**: Data is randomly partitioned among clients, ensuring each client has a similar distribution of classes and has same number of samples.
- **Non-IID (Non-Independent and Identically Distributed)**: Data is partitioned such that each client receives data from only a few classes, creating a skewed distribution.

### 3. Model Architecture

The CNN architecture used for the image classification task is as follows:  

**Input:**
    Input image size: 32x32 pixels with 3 channels (RGB).  
**Layers:**  
    - Convolutional Layer 1:  
        Input channels: 3 (RGB)  
        Output channels: 64  
        Kernel size: 3x3  
        Padding: 1
        Batch Normalization  
    - Convolutional Layer 2:  
        Input channels: 64  
        Output channels: 128  
        Kernel size: 3x3  
        Padding: 1  
        Batch Normalization  
    - Convolutional Layer 3:  
        Input channels: 128  
        Output channels: 256  
        Kernel size: 3x3  
        Padding: 1  
        Batch Normalization  
    - Convolutional Layer 4:  
        Input channels: 256  
        Output channels: 512  
        Kernel size: 3x3  
        Padding: 1  
        Batch Normalization  
    - Max Pooling Layer:  
        Kernel size: 2x2
        Stride: 2  
    - Fully Connected Layer 1:  
        Input features: 51222  
        Output features: 1024  
        Batch Normalization  
    - Fully Connected Layer 2:  
        Input features: 1024  
        Output features: 512  
        Batch Normalization  
**Output Layer (Fully Connected):**  
    Input features: 512  
    Output classes: 10  
**Activation Functions:**  
    ReLU activation functions are commonly used after each layer except for the output layer.  
**Dropout:**  
    Dropout with a probability of 0.5 is applied after the last fully connected layer.
This architecture consists of multiple convolutional layers followed by max pooling, batch normalization, and fully connected layers. The model aims to extract features from the input images and learn a representation that can be used for image classification into 10 classes.  

### 4. Federated Learning Setup

**Framework**: PyTorch 

**Number of Clients**: 10

**Communication Rounds**: 200

**Hyperparameters**:
- Batch size: 32
- Learning rate: 0.01

### 5. Results

The obtained test accuracy for FedAvg and FedProx under both IID and Non-IID scenarios is presented below. 

**Visualization**: Plots showing test accuracy over communication rounds.
![image](https://github.com/user-attachments/assets/149c272b-b27b-4911-a9ed-f09b27fe0ab9)




### 6. Analysis and Discussion
The plot illustrates the performance of two federated learning algorithms, FedAvg and FedProx, under both IID (Independent and Identically Distributed) and Non-IID (Non-Independent and Identically Distributed) data conditions over 200 training rounds.

**FedAvg Performance:**
    - IID Case: The FedAvg algorithm shows a steady increase in accuracy, reaching a plateau around 90%. This suggests that FedAvg effectively leverages the IID data distribution, allowing it to learn from the global model efficiently.
    - Non-IID Case: The accuracy of FedAvg under Non-IID conditions is significantly lower, fluctuating around 30-40%. This indicates that the algorithm struggles to generalize when data is not uniformly distributed among clients.  
**FedProx Performance:**
    - IID Case: FedProx also performs well with IID data, achieving comparable results to FedAvg, but with slightly more stability in the later rounds. This may be due to its ability to handle local model divergence better.
    - Non-IID Case: Like FedAvg, FedProx shows reduced accuracy in Non-IID scenarios, but it maintains a somewhat higher level of accuracy compared to FedAvg, oscillating around 40-50%. This suggests that FedProx's additional regularization term helps mitigate the challenges posed by the Non-IID distribution.  
**Comparison Between Algorithms:**
    In IID settings, both algorithms converge similarly, highlighting their effectiveness in well-distributed data scenarios.
    In contrast, under Non-IID conditions, FedProx outperforms FedAvg, indicating that the additional complexity introduced by FedProx is beneficial in handling non-uniform data distributions.  
**Implications:**
    - Model Selection: The choice of algorithm is crucial depending on the data distribution. For scenarios where data is likely to be Non-IID, FedProx may be a more suitable choice due to its robustness.
    - Further Research: Investigating hybrid approaches or tuning the parameters of FedProx could lead to improved performance across both IID and Non-IID settings.  
**Conclusion:**
    The results underscore the importance of considering data distribution in federated learning applications. While both FedAvg and FedProx demonstrate strengths, FedProx shows a notable advantage in handling challenges posed by Non-IID data distributions, making it a viable option for real-world federated learning scenarios.

### 7. Instructions to Run the Code

**Clone the Repository**:
```bash
git clone https://github.com/huyenlili/Federated-Learning.git
```
**Access the Notebook**:
Opens your the repository was cloned, navigate to the directory where the Jupyter Notebook file (file.ipynb) is located and open it by clicking on it.

**Run the Code Cells**:
Execute the code cells in the Jupyter Notebook 

**Note**: Ensure you have a CUDA-enabled GPU for faster training.

