# Dynamic Multi-Camera Object Detection and Tracking System

## Vision

This project is to demonstrate the inference optimization technique described in *Efficient Object Detection in Large Images using Deep Reinforcement Learning (2020)*
We combine a **static global view** with a **dynamic zoom-in camera** to provide a better understanding of a scene.

This system uses two cameras:
- A **global view camera** that monitors the overall scene.
- A **dynamic camera** that zooms in on areas of interest, providing detailed information about specific objects.

The goal is to **improve detection accuracy** by correlating the understanding between both camera views.

## Key Features

- **Dual-camera system**: A fixed camera captures a global perspective, while a movable camera zooms in for a closer look at objects of interest.
- **Real-time object detection**: Powered by YOLOv8 for accurate detection in different views.
- **deepSORT tracking**: To track objects between camera frames, ensuring correspondence between static and dynamic views.
- **Multi-view object correspondence**: A novel approach to match detected objects across two different perspectives using geometry and LiDAR detection.

## Challenges and Future Work

Currently this code is just a demo to demonstrate the confidence gain from extracting a higher-resolution realtime subpatch of a video stream.
