# QNX Build 
Using this repository: https://github.com/osaidahmed/Video-Streaming-from-QNX-8.0-to-a-Host-PC-Using-WSL we adapted the code after optimizing some of the files and making the C program parameterized.

 The code runs on QNX to stream a video feed (feed of images) over the network to a the central python script. 

This would work well in a work or home scenario where ping between devices is low and devices can be connected through ethernet.

However, at HTN, the ping between the two devices when simply running ping was over 100ms, resulting in us using a 480p 15fps stream, reducing the quality significantly. This is why we opted for a webcam option for a better demo and an insight into what the program would function like in an optimal setting.
