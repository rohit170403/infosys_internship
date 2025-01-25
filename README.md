### **Task 1: Cancer Patient Data Analysis and Prediction**

This task involves analyzing cancer patient data to predict the risk level (Low, Medium, High) using patient features. It employs statistical techniques, dimensionality reduction, and machine learning models like K-Nearest Neighbors (KNN) and Random Forest.

---

#### **Data Exploration and Preprocessing**

1. **Data Loading:**
   - The dataset is loaded from a Google Sheet named **'cancer patient data sets'** (Sheet1).
   - The Google Colab platform is used for data processing, ensuring easy access to cloud storage and data visualization tools.

2. **Data Cleaning and Preparation:**
   - The categorical **'Level'** column is encoded into numerical values: 
     - Low → 1
     - Medium → 2
     - High → 3
   - The **'Patient Id'** column is dropped as it does not contribute to prediction.
   - Numerical conversion is applied to relevant columns, ensuring data consistency and handling any missing or malformed data.

3. **Exploratory Data Analysis (EDA):**
   - **Correlation Analysis:**
     - The correlation of each numerical feature with the target column (**'Level'**) is calculated.
     - Features strongly correlated with **'Level'** are identified as potentially important for prediction.
     - Results are visualized using barplots for better understanding.
   - **Dimensionality Reduction with SVD:**
     - Singular Value Decomposition (SVD) reduces the dataset into two components for visual exploration.
     - Visualizations are created to observe how data points cluster based on the **'Level'** variable.
   - **Feature Distribution:**
     - Histograms for each feature display their distributions.
   - **Line Plots:**
     - The relationships between features and the target variable are plotted to uncover patterns or trends.

---

#### **K-Nearest Neighbors (KNN) Classification**

1. **Data Splitting and Scaling:**
   - The dataset is split into training (80%) and testing (20%) subsets.
   - Standardization is performed using **StandardScaler** to normalize feature values, ensuring fair distance calculations for KNN.

2. **Hyperparameter Tuning:**
   - A range of **K values** (number of neighbors) is tested.
   - Training and testing error rates are plotted for each value of **K**, helping identify the optimal value that minimizes errors.

3. **Model Training and Evaluation:**
   - A KNN classifier is trained using the optimal **K** determined earlier.
   - The model is evaluated using:
     - **Accuracy score**: Measures the percentage of correctly classified instances.
     - **Confusion matrix**: Provides detailed insights into true positives, false positives, etc.
     - **Classification report**: Includes precision, recall, and F1-score metrics.

---

#### **Random Forest Classification**

1. **Model Training:**
   - A Random Forest Classifier is trained with Out-of-Bag (OOB) scoring enabled.
   - OOB scoring provides an internal validation estimate, reducing the need for separate validation data.

2. **Hyperparameter Tuning:**
   - A **grid search** is conducted to optimize hyperparameters:
     - `max_depth`: Controls tree depth.
     - `min_samples_leaf`: Minimum samples per leaf node.
     - `n_estimators`: Number of trees in the forest.
   - A **4-fold cross-validation** ensures robustness in hyperparameter selection.

3. **Model Interpretation:**
   - **Decision Trees**:
     - The best-performing Random Forest model's decision trees are visualized to understand decision-making pathways.
   - **Feature Importances**:
     - The model's calculated feature importances highlight the most influential predictors in the dataset.

---

### **Task 2: Medical Text Analysis and Clustering**

This task analyzes medical text data for insights and groups similar cases through clustering techniques.

---

#### **Data Preprocessing**

1. **Text Cleaning:**
   - Common preprocessing steps include:
     - Removing stop words (common English words with little semantic value).
     - **Stemming**: Reducing words to root forms using Porter Stemmer.
     - **Lemmatization**: Extracting base forms of words using WordNet Lemmatizer.
     - **Spell Checking**: Fixing typographical errors using the `autocorrect` library.

---

#### **Exploratory Data Analysis (EDA)**

1. **Frequent Medical Terms:**
   - Word counts identify common terms in the medical text.

2. **Retained vs. Omitted Words:**
   - Visualizes the differences between the original text and cleaned text to assess preprocessing effectiveness.

3. **Demographic Analysis:**
   - Extracts and visualizes **age** and **gender** distributions from the data.

4. **Length Analysis:**
   - Explores the distribution of text lengths in various columns, offering insights into data consistency.

5. **Frequent Words and Bigrams:**
   - Top frequent words and common two-word combinations (bigrams) are identified and visualized.

---

#### **Dashboard**

- A comprehensive dashboard is created using **Matplotlib** to include:
  - Frequent medical terms.
  - Retained vs. omitted words.
  - Age and gender distributions.
  - Text length distributions for 'data' and 'conversation' columns.
  - Top 50 frequent words and bigrams.

---

#### **Clustering and Dimensionality Reduction**

1. **Feature Vectorization:**
   - Text data is converted into numerical features using **TF-IDF Vectorization**.

2. **Clustering:**
   - **K-Means Clustering** groups similar medical cases based on textual patterns.

3. **Dimensionality Reduction:**
   - **t-SNE Visualization** creates a 2D representation of clusters for interpretation.
   - **PCA** improves clustering performance by reducing feature matrix dimensions.

---

#### **Symptom Extraction and Analysis**

1. **NLP Model for Symptom Extraction:**
   - A domain-specific NLP model (**SciSpaCy**) extracts medical terms and symptoms from text.

2. **Symptom Lexicon Matching:**
   - Extracted terms are matched against a predefined **symptom lexicon** for accuracy.

3. **TF-IDF Vectorization of Symptoms:**
   - A TF-IDF matrix is generated specifically for the symptoms, aiding downstream tasks like clustering or classification.

---

#### **Future Improvements**

1. **Symptom Lexicon Expansion:**
   - Enhance the lexicon with additional medical terms for broader coverage.

2. **Algorithm Experimentation:**
   - Test alternative clustering techniques (e.g., DBSCAN, hierarchical clustering).

3. **Performance Metrics:**
   - Use specialized metrics to evaluate model efficacy in clustering and prediction.

This project integrates statistical analysis, machine learning, and natural language processing to gain actionable insights from cancer patient data and medical text, providing a strong foundation for further research and development.




colab files of all team members 
shanmuga kumar :- https://colab.research.google.com/drive/1B6FRD7hBeqwkg2Sq_EP1plq70AoWptZt?usp=sharing

sharma thangam :- https://colab.research.google.com/drive/1lzvKIb-5TYtM9Usiv42PuV7QMaeFVCWw?usp=sharing
