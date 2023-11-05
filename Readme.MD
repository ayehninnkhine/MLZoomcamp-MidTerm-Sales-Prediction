# *Sales Prediction on Advertising Expenditures*  

## **About the project**   

The objective of the project is to predict the sales based on advertising expenses.
* 'TV' - expenses on TV advertising
* 'Radio' - expenses on Radio advertising
* 'Newspaper' - expenses on Newspaper advertising
* 'Sales' - target variable: Sales
   
Developed ML model is realized as a web service and deployed to the Google Cloud Platform.    
   
## **Structure of the repository**    
   
The repository contains the next files and folders:

* `README.md` - project documentation    
* `advertising.csv` - advertising dataset (https://www.kaggle.com/datasets/bumba5341/advertisingcsv/data)
* `EDA and Model Training.ipynb` - a notebook with exploratory data analysis and model training
* `train.py` - a python script to train the model with Random Forest Regressor
* `model.bin` - a stored Random Forest Regressor model  
* `Pipfile` and `Pipfile.lock` - files with virtual environment for project
* `predict.py` - a python script to create a web service on the base of developed ML model    
* `Dockerfile` - to containerize the developed ML model
* `predict_local.py` - a python file to test and work with the locally deployed model
* `predict_cloud.py` - a python file to test and work with the model, deployed to Google Cloud Platform    

## **Virtual environment**   
   
Virtual environment of the project is provided by `Pipfile` and `Pipfile.lock`. These files contain all information about libraries and dependencies for the project. To create a virtual environment with libraries and dependencies required for the project, one should install `pipenv` library:  
   
`pip install pipenv`   
   
Then it's necessary to clone this repository from GitHub, open a terminal in the folder with this repository, and run the following commands:   
   
`pipenv install`   # to install project virtual environment
`pipenv shell`     # to activate virtual environment


## **Running a web service in a local server**   
      
Developed final model is implemented in a web service. To run it, it's necessary to install `Docker`, create a container (which contains all system dependencies, libraries, scripts and others) and run it.   
   
`Docker` may be installed from the official site https://www.docker.com/

File `Dockerfile` of the current repository (or cloned to your PC) contains all specifications to a container to be built: python, virtual environment, scripts and model file etc. To build a container one should start a `Docker`, open a terminal or command window and enter the next command:   
   
`docker build -t advertising .` (For Linux)

`docker buildx build --platform linux/amd64 -t advertising .` (For Apple M1)
   
Once docker container is built, you can run it with the next command:   
   
`docker run -it --rm -p 9696:9696 advertising:latest`
   
A script `predict_local.py` sends to the local server a condo with following features in json format:   

data = {
    'TV': 180.8,
    'Radio': 10.8,
    'Newspaper': 58.4,
}

The result of script's work should be as follows:   

{
 "predicted_advertising": 2.9011847847125116
}


## **Running a web service in a cloud**   
   
The web service is also deployed to Google Cloud Platform (Cloud Run) with the next commands:   
      
`gcloud config set project <project_ID>`  # create a project  (In this project, the project ID is galvanic-circle-401908)
   
`docker images`   # obtain a list of docker images to get exactly the name of needed image   
   
`docker tag advertising:latest gcr.io/galvanic-circle-401908/advertising_server`  # create a tag to image   
   
`docker push gcr.io/galvanic-circle-401908/advertising_server` # push image to Google Container Registry   
   
`gcloud run deploy advertising --image gcr.io/galvanic-circle-401908/advertising_server --port 9696 --platform managed --region us-central1`    # deploy image   

Web service API is available on URL: https://advertising-ogkgw3gxza-uc.a.run.app/predict
   
A script `predict_cloud.py` sends to Google cloud server a condo with the following features in json format:   

data = {
    'TV': 180.8,
    'Radio': 10.8,
    'Newspaper': 58.4,
}

The result of script's work should be as follows:   

{
 "predicted_advertising": 2.9011847847125116
}


## **Acknowledgement**   
   
I would like to thank DataTalksClub: Machine Learning Zoomcamp team for your insightful lectures. 
