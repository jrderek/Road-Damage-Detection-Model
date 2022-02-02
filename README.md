# Road-Damage-Detection-Model


Problem Explanation

Roads are one of the most crucial parts of a social as well as economic development of any developing as well as developed country. But as we are well aware that the maintenance of the same by governmental organisations such as municipalities is a big challenge, many researchers are already indulged in finding multiple ways for developing an efficient and apt way for helping the municipalities. If regular inspection of the road conditions are not maintained then the condition of roads will worsen gradually due to many factors such as weather, traffic, aging, poor choice of materials etc.
Some agencies deploy road survey vehicles which consist of multiple expensive sensors and high resolution cameras. There are some experienced road managers who supervise and perform visual inspection of roads. But these methods are of course really time consuming and expensive. Even after the completion of inspection, these agencies struggle to maintain accurate and updated databases of recorded structural damages.
This poor management leads to unorganised and inappropriate resource allocation for road maintenance.
So we need something which is inexpensive, fast and organised solution for such road damage detections. Nowadays we are very fortunate that almost everyone carries a camera based smartphone. So with the advent of Object Detection techniques in AI, people have started launching challenges and research in this domain and municipalities in Japan have already started using such smartphone based AI techniques to perform road damage inspection. So this case study is an attempt to use some state of the art techniques to build a model which will try to detect multiple types of road damages such as potholes, alligator cracks, etc using artificial intelligence tools.

Understanding the data

The dataset can be downloaded from: https://github.com/sekilab/RoadDamageDetector/
The training dataset contains Japan/India/Czech images and annotations. The format of annotations is the same as pascal VOC. The dataset proposed in this study considers data collected from multiple countries- Japan, India and Czech Republic (partially from Slovakia).

What is pascal VOC format?
It’s a type of image data format which contains an XML file for each image unlike the coco format which contains json file. The XML file contains all the information related to the object class found in the image along with it’s bounding box coordinates, etc.

![image](https://user-images.githubusercontent.com/96236642/152174621-75076462-b8ef-41e3-948e-a3617849053c.png)

A typical xml file for an image.

The image data contains three folders: Czech, India and Japan. Each folder contains two subfolders: Annotations and Images. Subfolder “Annotations” contains all the XML files for all the images present subfolder “images” with the same filename but the extensions.

![image](https://user-images.githubusercontent.com/96236642/152174700-b88bc4b9-f7da-45ad-a73b-f1ad7b917451.png)

Flow of Image data directory

The total number of images is increased to 26620, almost thrice the prevailing 2018 dataset. New images were collected from India and Czech Republic (partially from Slovakia) to make the data more heterogeneous and train robust algorithms. Although there are many types of road damage categories, the dataset focuses on following types:
D00: Linear crack, longitudinal, wheel mark part
D10: Linear crack, lateral, equal interval
D20: Alligator crack
D40: Potholes, Rutting, Dump, Separation
D43: Cross Walk Blur
D44: White Line Blur
D50: Utility hole (maintenance hatch)

Refer: (PDF) Generative adversarial network for road damage detection (researchgate.net)

![image](https://user-images.githubusercontent.com/96236642/152174800-d59b6aa0-9a33-4b39-a7f1-08129cdf1ec7.png)

Note that the standards related to evaluations of Road Marking deterioration such as Crosswalk or White Line Blur differ significantly across several countries. Thus, these categories were excluded from the study so that generalized models can be trained applicable for monitoring road conditions in more than one country.

Exploratory Data Analysis
Now, this dataset is in Pascal VOC format which is an XML file. For EDA I will convert them into csv first. 

So we create a csv file for each country data and hence we get three csv files. Then we concatenate all three into one to get the main csv file. So our final dataframe looks like this:

![image](https://user-images.githubusercontent.com/96236642/152174892-6d65ff52-6580-4c6f-95b9-af6c9278617b.png)


As you can see, for each image we have the dimension of image, location of bounding box and the damage type class.
Now let’s have a look at class distribution:

![image](https://user-images.githubusercontent.com/96236642/152174943-e8b3aa89-6fbb-42a0-a438-147729fd4ead.png)


There can be seen 10 different classes of damages.

The data is not balanced as class D43 (Cross Walk Blur), D01 (Longitudinal Construction Joint Part Crack), D11 (Lateral Construction Joint Part Crack) and D0w0 is really scarce whereas D20 (Alligator Crack) is in abundance.
So you may think of dropping all the entries with D43, D01, D11 and D0w0. In my case, I dropped these 4 classes entries and made it a six class labeled problem. After removing let’s see the area distribution of classes:

![image](https://user-images.githubusercontent.com/96236642/152175015-a065bab2-109e-4262-9ae4-7f56c80f1964.png)


Let’s the see the area density curve:
![image](https://user-images.githubusercontent.com/96236642/152175059-d8be0d0c-9abc-4300-81b5-49ced934497e.png)


Now let’s see what is our goal here. We have an image of a road, we want to detect a bound box which bounds the damage region on the road and detects the type of damage. So I took the images and drew the given bounding box in training data just for understanding.

![image](https://user-images.githubusercontent.com/96236642/152175101-2cb7fa0f-6df2-4258-9614-c096fd7aeba4.png)

![image](https://user-images.githubusercontent.com/96236642/152175167-007672a4-0f19-4c31-9301-742d4b9f1df9.png)

Czech:


India:

![image](https://user-images.githubusercontent.com/96236642/152175241-3daa8d97-2d0a-45af-8a27-f50e701d0882.png)

![image](https://user-images.githubusercontent.com/96236642/152175301-caeef8a8-d388-4b6b-a504-01ba596849d9.png)


Japan:

![image](https://user-images.githubusercontent.com/96236642/152175350-fb03fd64-0cbd-4704-b6f5-a74896c01475.png)

![image](https://user-images.githubusercontent.com/96236642/152175388-f27e0b31-dd75-49da-b26d-ddbdcb662a4a.png)



Now, let’s proceed forward and see some existing approaches.
Existing Approaches
Paper 1 : https://arxiv.org/pdf/2008.13101v1.pdf
Paper 2 : https://arxiv.org/ftp/arxiv/papers/1909/1909.08991.pdf

I tried three models for this problem:
YoloV3
YoloV5
EfficientDet3

The results from all three were very different. I will explain “How to custom train the model” for all three and share my observations.
Let me walk you through each solution.

YoloV3

Preparing the dataset:
First and foremost, create just one folder of images rather than three different folders for each country. Get the images corresponding to the 6 classes which we have decided to go with and delete rest images. To do so we need to keep the images present in our csv file which we have already created.
Step 1: Create a txt file for images of interest.

Step 2: Create a txt file for unwanted images.

Step 3: Delete the unwanted images.

Step 4: Now that you have all the image of interest in one folder, we will create another folder named “labels” in which we we will create a .txt file for each image which will contain the object class and coordinates of object in the image following this syntax: <object-class> <x_center> <y_center> <width> <height>.

  Step 5: Now that you have two folders (“images” and “labels”), you need to download the “Darknet” from here. You can git clone it also.

  Step 6: In Darknet folder, you will see a “Makefile”. Edit that Makefile and set GPU = 1 ,CUDNN=1 and OPENCV=1.

  Step 7: Download the darknet pre-trained model which is trained on Imagenet and place it in “Darknet” folder.

  Step 8: Now move the “images” and “labels” folder you created in the “Darknet/data” folder.

  Step 9: Go to “darknet/cfg” folder and create a copy of “yolov3.cfg” and rename it with “yolov3_custom_train.cfg”. Now do the changes in this “yolov3_custom_train.cfg” file:
In line 8 & 9, change the “width” and “height” as per your image or your choice.
In line 20, you can play with “max_batches”.
In line 603, 689 and 776, you can set “filters” using the formula (filters = (classes + 5)*3).
In line 610, 696 and 783, set “classes” equal to 6, since we have 6 different classes.

  Step 10: Since I trained on Google colab ,I wanted to save the weights after every 100 epochs and hence I changed line 138 in “detector.c” in “darknet/examples” folder as:
if(i%1000==0 || (i < 1000 && i%100 == 0)){
Step 11: Now it’s time to split the dataset into train and validation set. For this we need two .txt files “train.txt” and “val.txt” which is supposed to carry the directory paths for train images and validation images respectively.



Step 12: Now create “yolo.names” file with name of the classes in it, in “darknet/data”.
                        
                         ![image](https://user-images.githubusercontent.com/96236642/152175651-d182a2fd-7c5c-473f-9f06-0525a446ff70.png)


Step 13: Create “yolo.data” in “darknet/data”.

                         ![image](https://user-images.githubusercontent.com/96236642/152175733-f0d84478-bcc6-4c89-a24a-5e8b798b5f30.png)

                         
Training yoloV3:
Now that you have your darknet folder ready, zip it and upload it on google drive. Now open google colab and change the runtime to GPU. For training instruction you can check out my colab notebook for yoloV3 here.

                         Results:
The results were really really poor. The boxes dimensions were big, not detecting what it should have. For ex:
                         ![image](https://user-images.githubusercontent.com/96236642/152175824-a9a361c9-83d0-4eb2-9a4c-721d4de0027b.png)

                         ![image](https://user-images.githubusercontent.com/96236642/152175874-f30a14f4-8777-4ed6-b411-a26ea74d6b46.png)



![image](https://user-images.githubusercontent.com/96236642/152175946-ba07935c-7943-4f59-9dac-e774542867d8.png)

                         ![image](https://user-images.githubusercontent.com/96236642/152176017-d4cc98a5-58f3-4fb9-8e4f-f97fbba7ffbc.png)


I have also uploaded the testing notebook for tained weights, here.
YoloV5
                         
Preparing the dataset:
                         
Step 1: Create two folders “images” and “labels” as you did in yoloV3. But here don’t dump all the images together in “images” folder. Rather create subfolders “train” and “valid” in “images” folder as well as “labels” folder.
Step 2: Split all the images into train and valid folders which you created above. Rather than copy pasting randomly, you can do so by writing scripts and using the “train.txt” and “val.txt” files which contains directory paths for training images and validation images. I have already created those txt files while training yoloV3 above.

So you data directory should now look like this:
                         
                         ![image](https://user-images.githubusercontent.com/96236642/152176084-86708695-1802-4999-bf62-d0c415eba4e9.png)


Step 3: Clone the yoloV5 repo using:
!git clone https://github.com/ultralytics/yolov5
Step 4: After cloning, go to yolov5 directory and open “dataset.yaml” file and edit according to your configuration. In my case:

Step 4: You can choose different models from “yolov5/models”. For example, “yolov5l” (large), “yolov5s” (small), “yolov5x” (largest), etc. The larger the model you will select, the larger will be the number of parameters. So select you model and copy it at “yolov5/” outside the “models” subfolder. Make sure you keep just one model outside. In my case, I chose “yolov5l.yaml”
                         
                         ![image](https://user-images.githubusercontent.com/96236642/152176189-042b8896-21e7-4654-b6a9-6d9f4dbdfd55.png)

Step 5: Open selected model in “yolov5/”, (yolov5l.yaml) and change:
line 2, set “nc” equals to 6 (number of classes).
line 3 & 4, “depth_multiple” and “width_multiple” equals to 1.0.

                         Training:
Now that you have configured your “yolov5” folder, zip it and upload it on google drive. Open google colab and mount your drive. You can refer my training notebook here.
Results:
                         
Results were very satisying. Model detected the damages with proper classes in majority cases. That’s why I decided to go ahead with the trained weights of yoloV5.

                         ![image](https://user-images.githubusercontent.com/96236642/152176305-355c87a2-f16b-40ad-965a-148dcfb49535.png)
![image](https://user-images.githubusercontent.com/96236642/152176337-0afa7466-4250-4ae1-aab4-12de9a104dba.png)

                         ![image](https://user-images.githubusercontent.com/96236642/152176376-e85e36e2-0951-48cc-b4ba-17a2e4675ec9.png)
![image](https://user-images.githubusercontent.com/96236642/152176404-7c3c5d16-221e-4285-a82a-ef0255fa9a4d.png)

                         ![image](https://user-images.githubusercontent.com/96236642/152176452-f571645d-f92f-4426-83ef-212cf63d6cb7.png)
![image](https://user-images.githubusercontent.com/96236642/152176499-d5b89c00-3670-408d-ab86-838501f5e02f.png)

                         ![image](https://user-images.githubusercontent.com/96236642/152176651-c3f46906-fa1a-4cb0-9fca-75bf87513188.png)




Deployment using Flask
Now that the model is trained nicely, download the weights obtained after training and lets deploy and create a webpage.
For deployment purpose I used Flask. Flask is a simple web framework, also called as microframework, in Python which does not require heavy libraries as such and used to develop applications.
So we require a python file which has the end to end pipeline of ingesting an image and returning the output image with bounding boxes. Create a file “app.py” and write a function which is solely responsible for taking image as input and saving it’s output at some location in your local. You can take the “detect.py” file in “yolov5” directory and tweak a little bit here and there to manipulate it as per your requirement. So let’s say your function is ready, then below that you can create Flask functions for rendering the html files and uploading the image. You can check out my app.py file.

Obviously you need your html files which you can design as per your taste.

                         https://youtu.be/67hYS4kWW20

Future Work
                         
This problem still has lots of scope. I wanted to try with state of the art EfficientDet weights provided by TensorFlow 2.0 but due to the GPU and time constraints, I was not able to try much. But there is definitely lot of scope there.
Also, with yoloV5, I trained for only 34 epochs on colab. One can definitely try to run for around 99 epochs to see if performance is improving or not.
Since, I have limited resources that’s why I trained with the images provided in the dataset. Although the number of images are sufficient but still one can try with multiple image augmentation techniques to see any improvements.
                         
References:
https://www.appliedaicourse.com/
https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1
https://lionbridge.ai/articles/create-an-end-to-end-object-detection-pipeline-using-yolov5/
