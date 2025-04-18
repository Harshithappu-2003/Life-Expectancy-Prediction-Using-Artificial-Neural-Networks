### ❓ Problem Statement

Life expectancy is a key indicator of a country’s healthcare quality and overall well-being. Accurately predicting it based on health, demographic, and economic features can help governments and organizations make data-driven policy decisions. The problem addressed in this project is how to use historical and statistical data to effectively predict the life expectancy of a country.

* * *

### 🔍 Approach / Methodology

*   **Model Type**: Artificial Neural Network (ANN)
    
*   **Architecture**: A simple feedforward neural network built using TensorFlow/Keras.
    
    *   Input Layer: Based on the number of features
        
    *   Hidden Layers: Dense layers with ReLU activation
        
    *   Output Layer: Single neuron for regression
        
*   **Data Handling**:
    
    *   Cleaned missing values
        
    *   Encoded categorical variables
        
    *   Normalized feature data
        
*   **Training**: Model was trained using `mean_squared_error` as the loss function and the `adam` optimizer.
    
*   **Validation**: Used a train-test split (80-20) to evaluate the model performance.
    

* * *

### 🛠 Tech Stack

*   **Languages**: Python
    
*   **Libraries**:
    
    *   `Pandas` & `NumPy` – Data manipulation
        
    *   `Matplotlib` & `Seaborn` – Data visualization
        
    *   `Scikit-learn` – Data preprocessing and evaluation
        
    *   `TensorFlow` & `Keras` – Building and training the ANN model
        

* * *

### ⚙️ Installation & Usage

To run this project locally:

1.  **Clone the repository:**
    
    bash
    
    CopyEdit
    
    `git clone <repo-url> cd life-expectancy-regression-with-ann`
    
2.  **Install dependencies:**
    
    bash
    
    CopyEdit
    
    `pip install -r requirements.txt`
    
    _(or manually install `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`)_
    
3.  **Run the Jupyter Notebook:**
    
    bash
    
    CopyEdit
    
    `jupyter notebook life-expectancy-regression-with-ann.ipynb`
    

* * *

### 📈 Results / Output

*   **Evaluation Metric**: Mean Squared Error (MSE) and R² score
    
    *   Example Results:
        
        *   **MSE**: ~12.58
            
        *   **R² Score**: ~0.97 (high correlation and low error)
            
*   **Visuals**:
    
    *   Training loss curve
        
    *   Actual vs Predicted life expectancy plot
        

* * *

### ⚠️ Limitations & Future Work

*   **Limitations**:
    
    *   Data contains missing values that were imputed
        
    *   Assumes a static model—doesn't account for time-series trends
        
    *   Simpler ANN architecture might not capture all complex patterns
        
*   **Future Improvements**:
    
    *   Use advanced models like RNNs or LSTMs for temporal data analysis
        
    *   Add more recent or dynamic datasets (e.g., post-COVID stats)
        
    *   Hyperparameter tuning and model ensembling
        
    *   Deploy as a web-based tool using Flask/Streamlit
