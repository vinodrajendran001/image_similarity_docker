Prerequistes:

--> Install Docker and Docker-compose
--> mac users remove the .DS_store file present inside the subfolders esp. "static".
--> *"static" folder should contain only images

How to run SIFT/SSIM application

step 1: open the terminal

step 2: By default this program uses 'SIFT' algorithm

step 3: If you want to use 'SSIM' algorithm, go to app/app.py. In app.py, change the line -- algorithm = 'SIFT' to algorithm = 'SSIM' under constant definitions section.

Note: step 3 is optional

step 4: In terminal, inside the "image_similarity_SIFT_SSIM" run the following command

	~/image_similarity_SIFT_SSIM$ docker-compose build

	~/image_similarity_SIFT_SSIM$ docker-compose up

Thats it!

step 5: Go to browser and in the url type '0.0.0.0' 

	--file upload page opens
	--choose the image file (.jpg or .png) and click upload

voila...top 3 results will be displayed.


Note: Any changes done to "app" folder, then all commands in step 4 has to be re-executed

--------------------------------------------------------------------------------------------------
To verify storage in the database
--------------------------------------------------------------------------------------------------

1. open a new terminal
2. execute the command to open the mongodb shell
	$ docker exec -it imagesimilaritysiftssim_db_1 mongo
3. switch to "siftssim" database by executing the following command
	> use siftssim
4. To view the information stored(filename, features, datetime and top 3 results for uploaded images are displayed)
	> db.siftssim.find()
