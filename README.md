# Parkinson’s Disease Early Detection and Progression Analysis with Machine Learning

Our team utilized data from PPMI (Parkinson's Progression Markers Initiative) with machine learning to find ways to improve early detection of Parkinson's Disease. The goal is to determine the key features in Parkinson's Disease Development.

## Repository Appendix
- **Notebooks** folder contains all the notebooks for data processing/cleaning, unsupervised learning, and supervised learning
- **Datasets** folder contains original datasets we obtained from PPMI website, processed dataset, cleaned datasets, and training set
- **Images** folder contains all the images we see in the README file

## Introduction
Parkinson's disease is estimated to affect over 14.2 million people by 2040 due to increasing longevity and declining birth rate. The disease progresses slowly, but early detection is critical in mitigatign its symptoms and improving the quality of life. 

## Data Source
We got our training data from [PPMI website](http://ppmi-info.org), which includes measurements from questionnaires. 

## Process
1. Data cleaning and transformation
2. Data exploration: PCA
3. Feature Engineering: Importance Score
4. Unsupervised Learning: K-Means Clustering, SGDC (Stochastic Gradient Descendent Classifier), Agglomerative Hierarchical Clustering
5. Supervised Learning: Logistic Regression, Random Forest, Neural Networks

## Unsupervised Learning Results
Our first attempt is dimensionality reduction with PCA. The bar plot below shows the variance changes iterate through each principal component. We selected the first 4 compoenent to pass through K-Means clustering model because they contain 95% of the variance in the data.
image: ![](Images/pca.png)

The plot below on the left shows the K-Mean Clustering result with 4 components and cluster size set to 3. Inner-cluster distances were small. No seperation between clusters were seen. The plot on the right is the UMAP result with 2 components. 
image: ![](Images/umap.png)

The next approach is SGDC (Stochastic Gradient Descendent Classifier) with learning rate set to “optimal” with log loss function, due to the high dimensionality of training data. Sigmoid function was added to the last layer due to the binary nature of the target values (Positive vs Negative for Parkinson's). A large amount of lower surface area on the lost/cost function curve. The path traced by the classifier converges to the local minimum without any oscillation which means the learning rate is optimal in this case.
image: ![](Images/gradient_descent.png)

The plot below shows the result of the third approach with Agglomerative Hierarchical Clustering set to 3 clusters. The dendrogram shows a large separation after the first merge of clusters, then the difference between the clusters decreases as the clusters reach the stopping criteria. 
image: ![](Images/dendrogram.png)

### Unsupervised Learning Conclusion
We could assume the SDGC model is the most suitable for us to find important features that will help in supervised learning. We found the model performs well with Rigidity vs Tremor score values. 

## Supervised Learning Results
Logistic Regression to predict if a particular patient was ‘Healthy Control’ or ‘Parkinson’s Disease’. The model utilizes the logistic sigmoid function to manipulate the output into a probability value that can be mapped into the available class choices. We applied 5-fold cross validation. It resulted in a mean ‘roc_auc’ scoring value of ~0.684. Precision and recall, being 0.68 and 0.44 respectively. From image titled *ROC Curve - Logistic Regression*, the model has a ~73% probability of correctly choosing between a positive and a negative class. In the right image titled *Calibration Curve - Logistic Regression*, shows that overall our baseline model follows the perfect calibration curve well, a sign that our logistic regression model is decent as a supervised model in this case. image: ![](Images/supervised_1.png)

Second model is Random Forest model, works by combining the results of multiple decision trees. Applying the same preprocessing pipeline and 5-fold cross-validation with ‘roc_auc’ scoring produced a mean value of ~0.684, better than the logistic regression baseline model. From the plot titled *ROC Curve-Random Forest*, we see a greater area for lower false positive rate. In terms of its calibration curve shown in plot on the right titled *Calibration Curve - Random Forest* displays more variance than what the logistic regression model produces.
image: ![](Images/supervised_2.png)

Third machine learning model is neural networks, which is suitable for the high-complexity nature of the questionnaire data (normalized with MinMaxScaler). The baseline neural network architecture contains three hidden layers using ReLU activation functions and a Dropout layer to prevent overfitting. The output layer composed of a sigmoid activation since this is a binary classification problem. In the compiling phase, the Adam optimizer was used and the binary crossentropy as the loss function, using accuracy as the evaluation metrics. The baseline neural network resulted in a test accuracy of ~0.67 using 50 epochs and a batch size of 32.


### Supervised Learning Conclusion
Random Forest model performs better at choosing between the classes for lower false positive rates than Logistic Regression Model. Might be due to the fact Random Forest is an ensemble model.
Random Forest model is good at identifying true positives while keeping false positives low. 

## Evaluation
![](Images/evaluation_1.png)
![](Images/evaluation_2.png)

