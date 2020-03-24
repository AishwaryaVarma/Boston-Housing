# Boston-Housing
Boston Housing using sagemaker



Predicting Boston Housing Prices
Updating a model using SageMaker¶
Deep Learning Nanodegree Program | Deployment

Training two different models and using SageMaker to switch a deployed endpoint from using one model to the other. One of the benefits of using SageMaker to do this is that we can make the change without interrupting service. What this means is that we can continue sending data to the endpoint and at no point will that endpoint disappear.

General Outline
Typically, when using a notebook instance with SageMaker, you will proceed through the following steps. Of course, not every step will need to be done with each project. Also, there is quite a lot of room for variation in many of the steps, as you will see throughout these lessons.

Download or otherwise retrieve the data.
Process / Prepare the data.
Upload the processed data to S3.
Train a chosen model.
Test the trained model (typically using a batch transform job).
Deploy the trained model.
Use the deployed model.
In this notebook we will be skipping step 5, testing the model. In addition, we will perform steps 4, 6 and 7 multiple times with different models.

Step 0: Setting up the notebook
Begin by setting up all of the necessary bits required to run our notebook. To start that means loading all of the Python modules we will need.

In [ ]:
%matplotlib inline

import os

import numpy as np
import pandas as pd

from pprint import pprint
import matplotlib.pyplot as plt
from time import gmtime, strftime

from sklearn.datasets import load_boston
import sklearn.model_selection
In addition to the modules above, we need to import the various bits of SageMaker that we will be using.

In [ ]:
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# This is an object that represents the SageMaker session that we are currently operating in. This
# object contains some useful information that we will need to access later such as our region.
session = sagemaker.Session()

# This is an object that represents the IAM role that we are currently assigned. When we construct
# and launch the training job later we will need to tell it what IAM role it should have. Since our
# use case is relatively simple we will simply assign the training job the role we currently have.
role = get_execution_role()
Step 1: Downloading the data
Fortunately, this dataset can be retrieved using sklearn and so this step is relatively straightforward.

In [ ]:
boston = load_boston()
Step 2: Preparing and splitting the data
Given that this is clean tabular data, we don't need to do any processing. However, we do need to split the rows in the dataset up into train, test and validation sets.

In [ ]:
# First we package up the input data and the target variable (the median value) as pandas dataframes. This
# will make saving the data to a file a little easier later on.

X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

# We split the dataset into 2/3 training and 1/3 testing sets.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_bos_pd, Y_bos_pd, test_size=0.33)

# Then we split the training set further into 2/3 training and 1/3 validation sets.
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.33)
Step 3: Uploading the training and validation files to S3
When a training job is constructed using SageMaker, a container is executed which performs the training operation. This container is given access to data that is stored in S3. This means that we need to upload the data we want to use for training to S3. We can use the SageMaker API to do this and hide some of the details.

Save the data locally
First we need to create the train and validation csv files which we will then upload to S3.

In [ ]:
# This is our local data directory. We need to make sure that it exists.
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
In [ ]:
# We use pandas to save our train and validation data to csv files. Note that we make sure not to include header
# information or an index as this is required by the built in algorithms provided by Amazon. Also, it is assumed
# that the first entry in each row is the target variable.

pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
Upload to S3
Since we are currently running inside of a SageMaker session, we can use the object which represents this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with data uploaded from some other notebook or project.

In [ ]:
prefix = 'boston-update-endpoints'

val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
Step 4 (A): Train the XGBoost model
Now that we have the training and validation data uploaded to S3, we can construct our XGBoost model and train it. We will be making use of the high level SageMaker API to do this which will make the resulting code a little easier to read at the cost of some flexibility.

To construct an estimator, the object which we wish to train, we need to provide the location of a container which contains the training code. Since we are using a built in algorithm this container is provided by Amazon. However, the full name of the container is a bit lengthy and depends on the region that we are operating in. Fortunately, SageMaker provides a useful utility method called get_image_uri that constructs the image name for us.

To use the get_image_uri method we need to provide it with our current region, which can be obtained from the session object, and the name of the algorithm we wish to use. In this notebook we will be using XGBoost however you could try another algorithm if you wish. The list of built in algorithms can be found in the list of Common Parameters.

In [ ]:
# As stated above, we use this utility method to construct the image name for the training container.
xgb_container = get_image_uri(session.boto_region_name, 'xgboost')

# Now that we know which container to use, we can construct the estimator object.
xgb = sagemaker.estimator.Estimator(xgb_container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session
Before asking SageMaker to begin the training job, we should probably set any model specific hyperparameters. There are quite a few that can be set when using the XGBoost algorithm, below are just a few of them. If you would like to change the hyperparameters below or modify additional ones you can find additional information on the XGBoost hyperparameter page

In [ ]:
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)
Now that we have our estimator object completely set up, it is time to train it. To do this we make sure that SageMaker knows our input data is in csv format and then execute the fit method.

In [ ]:
# This is a wrapper around the location of our train and validation data, to make sure that SageMaker
# knows our data is in csv format.
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='text/csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='text/csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
Step 5: Test the trained model
We will be skipping this step for now.

Step 6 (A): Deploy the trained model
Even though we used the high level approach to construct and train the XGBoost model, we will be using the lower level approach to deploy it. One of the reasons for this is so that we have additional control over how the endpoint is constructed. This will be a little more clear later on when construct more advanced endpoints.

Build the model
Of course, before we can deploy the model, we need to first create it. The fit method that we used earlier created some model artifacts and we can use these to construct a model object.

In [ ]:
# Remember that a model needs to have a unique name
xgb_model_name = "boston-update-xgboost-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the xgboost container that we used for training
# can also be used for inference and the model artifacts come from the previous call to fit.
xgb_primary_container = {
    "Image": xgb_container,
    "ModelDataUrl": xgb.model_data
}

# And lastly we construct the SageMaker model
xgb_model_info = session.sagemaker_client.create_model(
                                ModelName = xgb_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = xgb_primary_container)
Create the endpoint configuration
Once we have a model we can start putting together the endpoint. Recall that to do this we need to first create an endpoint configuration, essentially the blueprint that SageMaker will use to build the endpoint itself.

In [ ]:
# As before, we need to give our endpoint configuration a name which should be unique
xgb_endpoint_config_name = "boston-update-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
xgb_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = xgb_endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": xgb_model_name,
                                "VariantName": "XGB-Model"
                            }])
Deploy the endpoint
Now that the endpoint configuration has been created, we can ask SageMaker to build our endpoint.

Note: This is a friendly (repeated) reminder that you are about to deploy an endpoint. Make sure that you shut it down once you've finished with it!

In [ ]:
# Again, we need a unique name for our endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = xgb_endpoint_config_name)
In [ ]:
endpoint_dec = session.wait_for_endpoint(endpoint_name)
Step 7 (A): Use the model
Now that our model is trained and deployed we can send some test data to it and evaluate the results.

In [ ]:
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[0])))
In [ ]:
pprint(response)
In [ ]:
result = response['Body'].read().decode("utf-8")
In [ ]:
pprint(result)
In [ ]:
Y_test.values[0]
Shut down the endpoint
Now that we know that the XGBoost endpoint works, we can shut it down. We will make use of it again later.

In [ ]:
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
Step 4 (B): Train the Linear model
Suppose we are working in an environment where the XGBoost model that we trained earlier is becoming too costly. Perhaps the number of calls to our endpoint has increased and the length of time it takes to perform inference with the XGBoost model is becoming problematic.

A possible solution might be to train a simpler model to see if it performs nearly as well. In our case, we will construct a linear model. The process of doing this is the same as for creating the XGBoost model that we created earlier, although there are different hyperparameters that we need to set.

In [ ]:
# Similar to the XGBoost model, we will use the utility method to construct the image name for the training container.
linear_container = get_image_uri(session.boto_region_name, 'linear-learner')

# Now that we know which container to use, we can construct the estimator object.
linear = sagemaker.estimator.Estimator(linear_container, # The name of the training container
                                        role,      # The IAM role to use (our current role in this case)
                                        train_instance_count=1, # The number of instances to use for training
                                        train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                        output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                            # Where to save the output (the model artifacts)
                                        sagemaker_session=session) # The current SageMaker session
Before asking SageMaker to train our model, we need to set some hyperparameters. In this case we will be using a linear model so the number of hyperparameters we need to set is much fewer. For more details see the Linear model hyperparameter page

In [ ]:
linear.set_hyperparameters(feature_dim=13, # Our data has 13 feature columns
                           predictor_type='regressor', # We wish to create a regression model
                           mini_batch_size=200) # Here we set how many samples to look at in each iteration
Now that the hyperparameters have been set, we can ask SageMaker to fit the linear model to our data.

In [ ]:
linear.fit({'train': s3_input_train, 'validation': s3_input_validation})
Step 6 (B): Deploy the trained model
Similar to the XGBoost model, now that we've fit the model we need to deploy it. Also like the XGBoost model, we will use the lower level approach so that we have more control over the endpoint that gets created.

Build the model
Of course, before we can deploy the model, we need to first create it. The fit method that we used earlier created some model artifacts and we can use these to construct a model object.

In [ ]:
# First, we create a unique model name
linear_model_name = "boston-update-linear-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the linear-learner container that we used for training
# can also be used for inference.
linear_primary_container = {
    "Image": linear_container,
    "ModelDataUrl": linear.model_data
}

# And lastly we construct the SageMaker model
linear_model_info = session.sagemaker_client.create_model(
                                ModelName = linear_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = linear_primary_container)
Create the endpoint configuration
Once we have the model we can start putting together the endpoint by creating an endpoint configuration.

In [ ]:
# As before, we need to give our endpoint configuration a name which should be unique
linear_endpoint_config_name = "boston-linear-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
linear_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = linear_endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": linear_model_name,
                                "VariantName": "Linear-Model"
                            }])
Deploy the endpoint
Now that the endpoint configuration has been created, we can ask SageMaker to build our endpoint.

Note: This is a friendly (repeated) reminder that you are about to deploy an endpoint. Make sure that you shut it down once you've finished with it!

In [ ]:
# Again, we need a unique name for our endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = linear_endpoint_config_name)
In [ ]:
endpoint_dec = session.wait_for_endpoint(endpoint_name)
Step 7 (B): Use the model
Just like with the XGBoost model, we will send some data to our endpoint to make sure that it is working properly. An important note is that the output format for the linear model is different from the XGBoost model.

In [ ]:
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[0])))
In [ ]:
pprint(response)
In [ ]:
result = response['Body'].read().decode("utf-8")
In [ ]:
pprint(result)
In [ ]:
Y_test.values[0]
Shut down the endpoint
Now that we know that the Linear model's endpoint works, we can shut it down.

In [ ]:
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
Step 6 (C): Deploy a combined model
So far we've constructed two separate models which we could deploy and use. Before we talk about how we can change a deployed endpoint from one configuration to another, let's consider a slightly different situation. Suppose that before we switch from using only the XGBoost model to only the Linear model, we first want to do something like an A-B test, where we send some of the incoming data to the XGBoost model and some of the data to the Linear model.

Fortunately, SageMaker provides this functionality. And to actually get SageMaker to do this for us is not too different from deploying a model in the way that we've already done. The only difference is that we need to list more than one model in the production variants parameter of the endpoint configuration.

A reasonable question to ask is, how much data is sent to each of the models that I list in the production variants parameter? The answer is that it depends on the weight set for each model.

Suppose that we have $k$ models listed in the production variants and that each model $i$ is assigned the weight $w_i$. Then each model $i$ will receive $w_i / W$ of the traffic where $W = \sum_{i} w_i$.

In our case, since we have two models, the linear model and the XGBoost model, and each model has weight 1, we see that each model will get 1 / (1 + 1) = 1/2 of the data sent to the endpoint.

In [ ]:
# As before, we need to give our endpoint configuration a name which should be unique
combined_endpoint_config_name = "boston-combined-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
combined_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = combined_endpoint_config_name,
                            ProductionVariants = [
                                { # First we include the linear model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": linear_model_name,
                                    "VariantName": "Linear-Model"
                                }, { # And next we include the xgb model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": xgb_model_name,
                                    "VariantName": "XGB-Model"
                                }])
Now that we've created the endpoint configuration, we can ask SageMaker to construct the endpoint.

Note: This is a friendly (repeated) reminder that you are about to deploy an endpoint. Make sure that you shut it down once you've finished with it!

In [ ]:
# Again, we need a unique name for our endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = combined_endpoint_config_name)
In [ ]:
endpoint_dec = session.wait_for_endpoint(endpoint_name)
Step 7 (C): Use the model
Now that we've constructed an endpoint which sends data to both the XGBoost model and the linear model we can send some data to the endpoint and see what sort of results we get back.

In [ ]:
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[0])))
pprint(response)
Since looking at a single response doesn't give us a clear look at what is happening, we can instead take a look at a few different responses to our endpoint

In [ ]:
for rec in range(10):
    response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[rec])))
    pprint(response)
    result = response['Body'].read().decode("utf-8")
    print(result)
    print(Y_test.values[rec])
If at some point we aren't sure about the properties of a deployed endpoint, we can use the describe_endpoint function to get SageMaker to return a description of the deployed endpoint.

In [ ]:
pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
Updating an Endpoint
Now suppose that we've done our A-B test and the new linear model is working well enough. What we'd like to do now is to switch our endpoint from sending data to both the XGBoost model and the linear model to sending data only to the linear model.

Of course, we don't really want to shut down the endpoint to do this as doing so would interrupt service to whoever depends on our endpoint. Instead, we can ask SageMaker to update an endpoint to a new endpoint configuration.

What is actually happening is that SageMaker will set up a new endpoint with the new characteristics. Once this new endpoint is running, SageMaker will switch the old endpoint so that it now points at the newly deployed model, making sure that this happens seamlessly in the background.

In [ ]:
session.sagemaker_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=linear_endpoint_config_name)
To get a glimpse at what is going on, we can ask SageMaker to describe our in-use endpoint now, before the update process has completed. When we do so, we can see that the in-use endpoint still has the same characteristics it had before.

In [ ]:
pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
If we now wait for the update process to complete, and then ask SageMaker to describe the endpoint, it will return the characteristics of the new endpoint configuration.

In [ ]:
endpoint_dec = session.wait_for_endpoint(endpoint_name)
In [ ]:
pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
Shut down the endpoint
Now that we've finished, we need to make sure to shut down the endpoint.

In [ ]:
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
Optional: Clean up
The default notebook instance on SageMaker doesn't have a lot of excess disk space available. As you continue to complete and execute notebooks you will eventually fill up this disk space, leading to errors which can be difficult to diagnose. Once you are completely finished using a notebook it is a good idea to remove the files that you created along the way. Of course, you can do this from the terminal or from the notebook hub if you would like. The cell below contains some commands to clean up the created files from within the notebook.

In [ ]:
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
In [ ]:

