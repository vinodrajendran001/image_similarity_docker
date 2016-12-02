Prerequistes:

--> Install Docker and Docker-compose
--> mac users remove the .DS_store file present inside the subfolders esp. "static".
--> *"static" folder should contain only images

How to run image_similarity CAE/VAE application

step 1: Go to "models" folder

step 2: copy anyone model file e.g. model_cae.h5 and place it inside the "app" folder.

step 3. Rename the copied model file inside "app" folder to "model.h5"
	e.g. "model_cae.h5" to "model.h5" 

step 4: open the terminal

step 5: In terminal, inside the image_similarity run the following command

	~/image_similarity_cnn$ docker-compose build

	~/image_similarity_cnn$ docker-compose up

Thats it!

step 5: Go to browser and in the url type '0.0.0.0' 

	--file upload page opens
	--choose the image file (.jpg or .png) and click upload

voila...top 3 results will be displayed.


Note: Any changes done to "app" folder, then all commands in step 5 has to be executed

--------------------------------------------------------------------------------------------------
To verify storage in the database
--------------------------------------------------------------------------------------------------

1. open a new terminal
2. execute the command to open the mongodb shell
	$ docker exec -it imagesimilaritycnn_db_1 mongo
3. switch to "caevae" database by executing the following command
	> use caevae
4. To view the information stored(filename, features, datetime and top 3 results for uploaded images are displayed)
	> db.caevae.find()

-------------------------------------------------------------------------------------------
For re-training the model (optional):
-------------------------------------------------------------------------------------------

step 1: Go to "training" folder

step 2: Run the required alogorithm file by changing the parameters
	e.g. ~/image_similarity_cnn/training$ python cae_imgsim.py

step 3: Copy the generated model file e.g. model_cae.h5 and place it inside "app folder" by renaming it to "model.h5"	

step 4: Repeat step 5.
