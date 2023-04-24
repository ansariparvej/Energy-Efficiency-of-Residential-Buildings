# Energy-Efficiency-of-Residential-Buildings

### Problem Statement

With record high and low temperatures across the globe, it is becoming increasing important to be efficient when it comes to heating and cooling our buildings. Whether you are trying to reduce the cost of your energy bill or you're trying to reduce your carbon footprint, improving the energy efficacy of your building can both save you some money and even help the environment. We will be looking at a data set that can help us with both!

### Goal:

This is a Multi Regression problem, in which the related inputs are given as inputs to the model and the respective model returns estimated multi-outputs in the form of Heating Load and Cooling Load which are required for energy efficient residential buildings.

### Dataset Source:

https://archive.ics.uci.edu/ml/datasets/Energy+efficiency

                          OR

https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset


### Proposed Solution Approach:

First of all, Exploratory Data Analysis (EDA), Feature Engineering (FE) and Feature Selection (FS) [if required] using various python based libraries [pandas, numpy etc.] on downloaded data set from the above mentioned link will be performed. Visualization tools [matplotlib, seaborn etc.] will aid to get a better understanding of the data that we are working with. Afterwards, distinct regression models wiil be created. Finally, We will evaluate these models using distinct perfomance metrics plus will try to get best Hyper prameters using Grid Search CV apporach and will select the best performing(most suitable) model for this specific dataset for predictions of heating load as well as cooling load of residential buildings."

### Tech Stack Used

1. Python 
2. Pycharm as IDE & Jupyter Notebook for Analysis
3. Machine learning algorithms 
4. MongoDB
5. FastAPI [Back End]
6. Streamlit [UI Interface/Front End]
7. Docker [Docker Desctop and Docker hub]

### Most Suitable Machine Learning Model [As per given Dataset] 

XGboost Multi-Output Regressor Model [Testing Accuracy: 99.67%]

### Infrastructure Required for Deployment.

1. Docker hub
2. AWS Elastic Beanstalk (EB)

## How to run?

Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, Elastic Beanstalk (EB).

## Data Collections
![image](https://user-images.githubusercontent.com/57321948/193536736-5ccff349-d1fb-486e-b920-02ad7974d089.png)


## Project Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)


## Deployment Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)


### Step 1 - Clone the repository
```bash
git clone https://github.com/ansariparvej/Energy-Efficiency-of-Residential-Buildings.git
```

### Step 2 - Create a conda environment after opening the repository

```bash
conda create -n Energy_Efficiency_of_Residential_Buildings python=3.8.16 -y
```

```bash
conda activate Energy_Efficiency_of_Residential_Buildings
```

### Step 3 - Install the requirements
```bash
pip3 install -r requirements.txt
```

### Step 4 - Run the application server [Using Fast API]
```bash
python main.py
```

### Step 5 - Train application 
```bash
http://localhost:8080/train

```

### Step 6 - Run the application server [Using Fast API]
```bash
streamlit run app.py
```

### Step 7 - Prediction application [Using Streamlit/UI]
```bash
http://localhost:8501

```

## For Application Deployment, REFER: >> Dployment_Steps.pdf file.


Application Link:

http://energyefficiencyofresidentialbuildin-e-2.eba-vypd6sib.ap-south-1.elasticbeanstalk.com/

