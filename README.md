# Heart Attack Prediction
Synthesize Decision Tree algorithms and their variations to predict the likelihood of developing heart disease.

## 1. Dataset
In this project, I will apply basic machine learning methods to predict whether a person is likely to develop heart disease based on the **Cleveland Heart Disease** dataset from the **UCI Machine Learning Repository**. 
The **Cleveland** dataset comprises 14 features, including:
- age
- gender
- chest pain type
- resting blood pressure
- serum cholesterol levels
- fasting blood sugar
- resting ECG results
- maximum heart rate achieved
- exercise-induced angina
- ST depression induced by exercise relative to rest
- peak exercise ST segment
- the number of major vessels (0-3) colored by fluoroscopy
- thalassemia
- a diagnosis of heart disease (0 representing patients without the disease, and 1, 2, 3, 4 representing patients with varying degrees of heart disease).

The Cleveland dataset consists of **303 samples**, with these **14 features** depicted in here.

<img width="1000" alt="dataset" src="https://github.com/duongngockhanh/heart-attack-prediction/assets/87640587/ca42df6f-bb8a-4acb-84e2-e966d3291a80">

## 2. Algorithms
I have experimented with training on **9 machine learning algorithms**, which include:

1. K Nearest Neighbors
2. Support Vector Machine
3. Naive Bayes Classifier
4. Decision Tree
5. Random Forest
6. AdaBoost
7. GradientBoost
8. XGBoost
9. Stacking

The obtained results regarding the **accuracy** of each model after evaluation on the validation dataset can be seen in the figure below.

<img width="1000" alt="comparison" src="https://github.com/duongngockhanh/heart-attack-prediction/assets/87640587/4124f139-f609-45b3-97d0-3aee80717a5e">

## 3. Stacking
Furthermore, I would like to provide a more detailed description of the **Stacking algorithm** I used. 

In this approach, I employed **6 models** to extract the information, which was then utilized as input for the **XGBoost algorithm**.

<img width="1000" alt="stacking" src="https://github.com/duongngockhanh/heart-attack-prediction/assets/87640587/02b7c28a-48a6-4555-bde8-6ed6413e4985">
