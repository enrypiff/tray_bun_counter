# Trays and buns counter
In this project I decided to improve my previous version of buns counter, by counting also the tray and showing the segmentation area detected. 
I manually annotated 12 photos I took during the commissionig of the production line and augmented them with Roboflow to have some samples for my dataset. 

![instance segmentation](https://github.com/enrypiff/tray_bun_counter/assets/139701172/9ccf5b54-2d8f-4c6b-855e-38091fe1f911)

Then I trained YOLO v8 for instance segmentation (I noticed that is more faster and accurate for very small datasets.

To handle the tracking problem for the counter I've used the SORT tracking algorithm.

This is the final result, a very promising use of YOLO in industrial field.


https://github.com/enrypiff/tray_bun_counter/assets/139701172/e0680d75-55a8-4397-8f4b-de29f60906b8


