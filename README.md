# The MiLe End Hums and Whistles Machine Learning Classification Mini-Project 

I have built a machine learning model to classify hum and whistles of 2 iconic songs (details of these eight songs as shown below). For example, take an input of a Potter audio segment of humming or whistling and predicts its song label (lables are these 2 types of songs)

* Harry Potter theme song (Potter) https://youtu.be/Htaj3o3JD8I?t=0
* The Imperial March (StarWars) https://youtu.be/s3SZ5sIMY6o?t=9


# Dataset
As students of Queen Mary University of London, we have created approximately 15 seconds audio segments of hum and whistle of these 2 songs. 

As mentioned in section 3 ML pipeline, the data involved is devided into three parts:
* training data to train the model
* validating data to tune the model 
* testing data to test the final result of the model.

Experiment data: Train data and validate data is four features extracted from 50 recordings of Potter and StarWars, and splitting them into a ratio of 7:3 to use as training and validating purpose.
Test data is the four features extracted from another 50 recordings of Potter and 50 recordings of StarWars.
After extracting features for the experiment data, I have produced various 2-D and 3D scatter plots to view them. And I also applied PCA to reduce to two PCs to visualize the data. Then I see there seems to have some outliers which are quite far away from the majority of the data. So I have applied Mahalanobis distance to exclude the top five items according to their outlier scores.

And I have also excluded the top five items in the final test dataset according to their outlier scores by using Mahalanobis distance again.


# Machine Learning pipeline
My machine learning pipeline involves these several parts: its input, its transformation stage, actual machine learning models used, and finally its output.

Input: I have used 50 recordings of Potter and 50 recordings StarWars from Part1 folders as the input for training and validating of the model purpose.

Transformation stages: since what we are dealing here is audio recordings, unfortunately the features and labels for the ML modelling are not very straightforward. Therefore a series of data pre-processing steps are involved to get the features and lables of training data for the later modelling.

1)The audio segments of Potter Part1 and StarWars Part1 are stored online and in zip file, I have downloaded them to Google drive.

2) Unzip them to specified directory.

3) I have printed file paths and realized that some of the recordings are wrongly named which is affecting extracting the correct lables of my training and validating data later on, thus I have applied some rules to rename them into the right format.

4) I used the function getXy() to extract four features 'power','pitch_mean', 'pitch_std' and 'voiced_fr' of each recording and the corresponding labels of the data.

5) The top five outliers have been removed based on Mahalanobis distance.

6) Split the data into a ratio of 7:3 for training and validating purposes.

Modelling: I have used KNN with k=3 as my model method with 5-fold cross validation to train and validate the model in order to get a more general training result.

Output: the final output of the model is to predict if a recording is from Potter or StarWars. If the label predicted is 1 it means it's from Potter, if the label predicted is 0 then it means it's from StarWars.

# Transformation stage

# Modelling
Transformations are described in detail in previous section 'Machine Learning pipeline'.

#Methodology
After preprocessing the data, I have visualised the features on the scatter plots, I see that the data is not easy to be seen as two clear classifications. So I have been carried out various ML models to explore which one is the best to use. And the model performanced is assessed based on it's accuracy and f1-score. I trained the model with train data and test them with validate data. After trying with the following models, I will decide the best model to use. After that, I will use the final test data to test the model.

1.KNN
2.SVM
3.Logistic Regression (and with polynomial features)
4.Naive Bayes Classifier
5.Random Forest Tree


