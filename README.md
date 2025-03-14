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
image: ![](Images/supervised_1.png)

image: ![](Images/supervised_2.png)
image: ![](Images/supervised_3.png)

## Evaluation
image: ![](Images/evaluation_1.png)
image: ![](Images/evaluation_2.png)

