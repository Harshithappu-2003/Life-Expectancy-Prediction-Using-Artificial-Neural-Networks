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
            
        *   **R² Score**: ~0.9108733809426091 (high correlation and low error)
            
*   **Visuals**:
    
    *   Training loss curve
        
    *   Actual vs Predicted life expectancy plot
        

* * *

### Screenshots
![2](https://github.com/user-attachments/assets/e963c759-a6a0-4ea2-9304-f2e8f849a996)
![3](https://github.com/user-attachments/assets/67edee80-93cc-409c-a699-c90f9d62e511)
![4](https://github.com/user-attachments/assets/0ade154a-3d69-4e05-bdc2-a4e30a1a7e78)
![5](https://github.com/user-attachments/assets/c22f8599-a095-4907-af76-4c5476110413)
![6](https://github.com/user-attachments/assets/88de876b-851f-4525-b4c4-90ca86fc5fcf)
![7](https://github.com/user-attachments/assets/2c92fc06-7934-4242-bfb3-1959fb35ffca)
![8](https://github.com/user-attachments/assets/ff20ffb9-6cd5-45ca-a5c3-14d304eedb5e)
![9](https://github.com/user-attachments/assets/9410f364-c40c-4a5d-a409-d2af734a4233)
![10](https://github.com/user-attachments/assets/17cb9e71-5b0b-4aed-81eb-0bf03549aee4)
![11](https://github.com/user-attachments/assets/987b346d-37fa-453c-ad76-56d35965df96)
![12](https://github.com/user-attachments/assets/f4eefcc8-51af-455f-967c-a0f7884049d6)



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
