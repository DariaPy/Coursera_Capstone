# Project goal
Current repository presents the materials and results submitted for the course "IBM Applied Data Science: Capstone project"

The goal of this project is to build several machine learning models to be able to predict severity of consequences for motor vehicles collisions in Seattle using the dataset of previously registered cases and their attributes. This project is based on City of Seattle data on traffic crash, also called a motor vehicle collisions, or car accidents. Motor vehicle collisions occurs when a vehicle collides with another vehicle, pedestrian or cyclist, animal or stationary obstruction.

# Data sources
The dataset includes all types of collisions and is hosted by the City of Seattle at the open data platform. 
The CSV file about collisions can be obtained via link http://data-seattlecitygis.opendata.arcgis.com/datasets/5b5c745e0f1f48e7a53acec63a0022ab_0.csv 
Description of different attributes of the collision cases will be used to build machine learning models and predict severity of accidents. 
The file of that description (Collisions_OD.pdf) is available via link  https://www.seattle.gov/Documents/Departments/SDOT/GIS/Collisions_OD.pdf. 
The dataset timeframe: 2004 to Present (September 30, 2020).
Update Cycle: Weekly.

# Project plan
The general plan for this project analysis is:
•	Load, read and inspect the dataset.
•	Fix missing data and type mismatch.
•	Investigate relations between accidents severity and attributes.
•	Pick relevant attributes and build ML models.
•	Assess and compare models’ performance.
•	Improve models and describe the final model with recommendations.

# Data Cleaning
Initial dataset included 221,738 observations and 40 attributes.
Cleaned dataset included 169,089 observations and 22 attributes.

# Machine Learning modeling
I picked 5 machine learning techniques for the current classification problem:
•	Decision Tree.
•	Random Forest.
•	Logistic Regression.
•	Naive Bayes.
•	k-Nearest Neighbors.
To train the models I used 18 attributes for 135,271 observations of accidents. 
To test the models I used 18 attributes for 33,818 observations of accidents.

# Libraries and methods:
•	Numpy - to work with arrays.
•	Pandas - to work with tabular data.
•	Datetime - to extract the data from timestamp.
•	Plotly, Matplotlib, Seaborn - to plot the data.
•	Scikit-learn - to build and assess ML models.

# Results
Among the individual models, the Decision Tree model performed the best, not only because it showed 100% accuracy, but mostly because it predicted the most severe cases correctly (minority cases, class 3). Random Forest performed the same, because it is simply a collection of decision trees whose results are aggregated into one final result. Random Forest could be useful in case Decision tree would not do the trick; but in this particular project Decision Tree classification model works perfectly fine on its own. 

Naive Bayes prediction model showed 0.990 accuracy score, and although it had some errors in class 1 and 2, it was the second after the best result. This model was able to give correct prediction for all positive samples of the most severe cases (minority cases, class 3). 

Logistic Regression prediction model showed 0.994 accuracy score, but its recall for severity class 3 was only 0.68. That means the classifier was able to find only 68% of the positive samples of  class 3, therefore it’s not good enough.

k-Nearest Neighbors prediction model showed 0.916 accuracy score and 0.925 F1 score for k=17, but that could be the problem for the minority of cases, accidents of class 3. Due to unbalanced dataset kNN model is not able to learn predicting class 3 correctly (recall for the class 3 is only 0.01, and F1 score is 0.03). kNN model performance was the worst in this case study, despite high overall accuracy.
