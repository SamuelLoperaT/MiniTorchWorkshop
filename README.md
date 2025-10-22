# 🧠 MiniTorch Workshop

> A hands-on deep learning framework built from scratch — implementing the core components of a neural network and applying them to a real-world Kaggle competition.

---

## 📘 Overview

The **MiniTorch Workshop** is an educational project focused on understanding the fundamentals behind modern deep learning frameworks such as PyTorch and TensorFlow.  
Through this workshop, I built a simplified neural network library from first principles — implementing the mathematical and programmatic foundations of layers, activations, loss functions, and optimization.

This repository contains:
- My implementation of the MiniTorch framework in individual classes (`Batchorm.py`, `Dropout.py`, `Linear.py`,`Net.py`,`CrossEntropyFromLogits.py`)
- The accompanying Jupyter Notebook (`Reporte y Entrenamiento.ipynb`)
- A Kaggle notebook applying the framework to a real classification problem

---

## 🧩 Project Structure
```
MiniTorchWorkshop
│
├── Reporte y Entrenamiento.ipynb # Step-by-step workshop notebook
├── Dropout.py # Custom neural network library
├── Linear.py # Custom neural network library
├── Batchorm.py # Custom neural network library
├── CrossEntropyFromLogits.py # Custom neural network library
├── Net.py # Custom neural network library
├── AICompetition.ipynb # Kaggle competition notebook
└── README.md # This file
```

---

## 🚀 Learning Objectives

1. **Implement Core Neural Network Components**
   - Forward and backward passes for:
     - Linear (fully connected) layers  
     - Activation functions (e.g., ReLU)  
     - Loss functions (e.g., Cross-Entropy)
2. **Add Regularization Techniques**
   - Batch Normalization  
   - Dropout
3. **Train and Evaluate on MNIST**
   - Build models using the MiniTorch framework  
   - Visualize training/validation performance
4. **Apply the Framework in a Kaggle Competition**
   - Import the custom library  
   - Design, train, and compare network architectures  
   - Submit predictions and analyze results

---

## 🧠 What I Learned

- **Forward/Backward Propagation** — manually computing gradients and understanding the mechanics behind backprop.  
- **Network Architecture Design** — exploring different model configurations and hyperparameters.  
- **Regularization** — implementing BatchNorm and Dropout to improve generalization.  
- **Practical ML Workflow** — training models, monitoring convergence, and validating results.  
- **Model Evaluation** — interpreting metrics and analyzing the effect of architectural choices.

---

## 🏗️ Technologies & Tools

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python |
| Core Libraries | NumPy, Matplotlib |
| Environment | Jupyter Notebook |
| Dataset | Intel Image Classification |
| External Platform | [Kaggle Competition](https://www.kaggle.com/code/juanmartinezv4399/ai-competition01) |

---

## 📊 Results Summary

| Metric | Description |
|--------|-------------|
| **Best Model** | Custom 3-layer neural network with BatchNorm + Dropout |
| **Dataset** | Intel Image Classification |
| **Accuracy** | 0.47 |
| **Observations** | BatchNorm improved convergence speed and stability; Dropout reduced overfitting |

Results considered reasonable, since the dataset could use a different arquitechture, such as a CNN.
---

## 🧰 How to Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/SamuelLoperaT/MiniTorchWorkshop.git
   cd MiniTorchWorkshop```

2. **Install Dependencies**

    ```bash
    pip install -f requirements.txt
    ```
3. **Run the Workshop Notebook**
    ```bash
    jupyter notebook 'Reporte y Entrenamiento.ipynb'
    ```
4. **Train your Model**
    - Follow the notebook steps to implement layers and activations.
    - Run experiments and evaluate your model on MNIST.
## 🧩 Key Files
| File                      | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| `Reporte y Entrenamiento.ipynb` | Step-by-step guide for implementing and testing components |
| `BatchNorm.py`            | BatchNorm Layer Implementation              |
| `Dropout.py`            | Dropout Layer Implementation              |
| `Linear.py`            | Linear Layer Implementation              |
| `Net.py`            | Neural Network Wrapper and data structures            |
| `CrossEntropyFromLogits.py`            | Cross Entropy cost function and softmax layer           |
| `README.md`               | Project documentation                                      |

## 🏁 Outcome

By completing this workshop, I developed:

- A working understanding of neural network internals

- Experience with gradient-based optimization

- Skills in model regularization and evaluation

- A fully functional deep learning framework from scratch

## 💬 Author

**Samuel Lopera Torres**

📧 [samuel69lopera@gmail.com](samuel69lopera@gmail.com)

🔗 [LinkedIn](https://www.linkedin.com/in/samuel-lopera-torres-2b8400212) | [GitHub](https://github.com/SamuelLoperaT)
