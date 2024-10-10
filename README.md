# Dynamic Multi-Camera Object Detection and Tracking System

## Vision

This project is to demonstrate the inference optimization technique described in *Efficient Object Detection in Large Images using Deep Reinforcement Learning (2020)*
We combine a **static global view** with a **dynamic zoom-in camera** to provide a better understanding of a scene.

This system uses two cameras:
- A **global view camera** that monitors the overall scene.
- A **dynamic camera** that zooms in on areas of interest, providing detailed information about specific objects.

And one LiDAR sensor:
- A LiDAR sensor will estimate the distance to the area of focus from the **global view**. This will allow the **dynamic view** to calculate the exact zoom level and angle adjustment required to center on the target without needing complex feature matching between the two cameras.

The goal is to **improve detection accuracy** by correlating the understanding between both camera views.

## Key Features

- **Dual-camera system**: A fixed camera captures a global perspective, while a movable camera zooms in for a closer look at objects of interest.
- **Real-time object detection**: Powered by YOLOv8 for accurate detection in different views.
- **deepSORT tracking**: To track objects between camera frames, ensuring correspondence between static and dynamic views.
- **Multi-view object correspondence**: A novel approach to match detected objects across two different perspectives using geometry and LiDAR detection.

## Challenges and Future Work

Currently, this code is just a demo to demonstrate the confidence gain from extracting a higher-resolution realtime subpatch of a video stream.

- Future work is to incorporate the geometric calculation and experiment with two physical cameras, starting by assuming the objects in the global view are a fixed distance on a flat plane away.
- Adding a LiDAR array and logic to determine distance from global view will remove the need for that assumption


