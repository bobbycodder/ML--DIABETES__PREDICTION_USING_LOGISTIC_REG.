# ML--DIABETES__PREDICTION_USING_LOGISTIC_REG.
I have created **Machine Learning Model With Linear Regression** for ** DIABETES Predictions.**


 <img src="https://www.genengnews.com/wp-content/uploads/2020/08/Aug28_2020_Getty_1213259073_DiabetesTestingEquipment-scaled-e1598623827835.jpg" width="400" height="200">

In this I've used Python’s Famous libraries like [Plotly](https://www.geeksforgeeks.org/getting-started-with-plotly-python/#:~:text=The%20Plotly%20Python%20library%20is%20an%20interactive%20open%2Dsource%20library.&text=plotly%20graph%20objects%20are%20a,histograms%2C%20pie%20charts%2C%20etc.), [Pandas](https://pandas.pydata.org/), [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/), [Scipy](https://www.scipy.org/), and [Sklearn](https://scikit-learn.org/) for **Data analysis and Model Development.**

I've created [Linear Regression Model](https://www.geeksforgeeks.org/ml-logostic-regression/) and performed it onto different companies car price data.

I've used [Jupyter Notebook](https://jupyter.org/) for coding!

I obtained the Dataset from [Kaggle](https://www.kaggle.com/kandij/diabetes-dataset).

  
<img src="https://www.cdc.gov/pcd/issues/2017/images/16_0244_01.gif" width="400" height="200">  

# DIABETES_ PREDICTION 

__The data was collected and made available by “National Institute of Diabetes and Digestive and Kidney Diseases” as part of the Pima Indians Diabetes Database. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here belong to the Pima Indian heritage (subgroup of Native Americans), and are females of ages 21 and above.__

__We’ll be using Python and some of its popular data science related packages. First of all, we will import pandas to read our data from a CSV file and manipulate it for further use. We will also use numpy to convert out data into a format suitable to feed our classification model. We’ll use seaborn and matplotlib for visualizations. We will then import Logistic Regression algorithm from sklearn. This algorithm will help us build our classification model. Lastly, we will use joblib available in sklearn to save our model for future use.__



# What I have done?

**I have done:**
  * Data Cleaning and Analysis,
  * Data Exploration,
  * Build a function to COUNT PLOT of any ALL FEATURES 
  * Build a Logistic  regression model for predicting diabetes with some features
  * Then predicting model 
  * Defining confussion matrix and classification report 
  * ROC curve


<img src="data:image/png;base64,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" width="400" height="200">

# My Model is predcting with 80% ACCURACY

***Follow along Jupyter Notebook{Predicting Stock Prices with Linear Regression.ipynb} for more!!***

 ***I hope you liked this project on Stock Price prediction using Python with machine learning by implementing the Linear Regression Model***
