Prerequistes:

--> Install Docker and Docker-compose
--> mac users remove the .DS_store file present inside the subfolders esp. "static".
--> *"static" folder should contain only images

How to run Knn-SURF application

step 1: open the terminal

step 2: In terminal, inside the "image_similarity_surf_knn" run the following command

	~/image_similarity_surf_knn$ docker-compose build

	~/image_similarity_surf_knn$ docker-compose up

Thats it!

step 3: Go to browser and in the url type '0.0.0.0' 

	--file upload page opens
	--choose the image file (.jpg or .png) and click upload

voila...top 3 results will be displayed.

Note: Any changes done to "app" folder, then all commands in step 4 has to be re-executed


--------------------------------------------------------------------------------------------------
To verify storage in the database
--------------------------------------------------------------------------------------------------

1. open a new terminal
2. execute the command to open the mongodb shell
	$ docker exec -it imagesimilaritysurfknn_db_1 mongo
3. switch to "surfknn" database by executing the following command
	> use surfknn
4. To view the information stored(filename, features, datetime and top 3 results for uploaded images are displayed)
	> db.surfknn.find()
