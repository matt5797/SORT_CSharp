# SORT-CSharp
This repository contains a C# implementation of the Simple Online and Realtime Tracking (SORT) algorithm. Originally developed for Python, this port aims to provide a fast, accurate, low-latency object tracking system that is suitable for many applications, from surveillance to autonomous vehicles.

## Overview
The Simple Online and Realtime Tracking (SORT) algorithm is a pragmatic approach to multiple object tracking with a focus on simple, effective methods. The algorithm works by combining both the detection confidence and motion prediction into a single score.

This C# port is designed to offer the same capabilities within a .NET 6.0 environment, maintaining the performance and ease-of-use of the original while taking advantage of the C# language's robustness and features.

## Getting Started
To use the SORT-CSharp library in your project, you can clone this repository and include the source code in your project.

```bash
git clone https://github.com/matt5797/SORT-CSharp.git
```
Or download the .zip file and extract the contents into your project directory.

## Dependencies
This implementation is built on .NET 6.0. Please ensure you have the appropriate .NET runtime installed on your system.

## How to Use
To use the SORT-CSharp library in your project, instantiate the __`Sort`__ class, which serves as the main interface to the library, and call its __`Update`__ method with a list of detections. Each detection is represented as a double array that specifies the bounding box of the object.

```csharp
Sort tracker = new Sort(maxAge: 1, minHits: 3, iouThreshold: 0.3);
double[][] detections = ... // get detections from your detector
double[][] trackedObjects = tracker.Update(detections);
```

For each tracked object, the __`Update`__ method returns an array that includes the bounding box and an ID. The ID is maintained across frames for the same object.

Please refer to the __`Sort.cs`__ and __`KalmanBoxTracker`__ class in the enclosed original code for more details on how to use and customize this implementation.

## Contributing
Contributions are welcome! Please read the contributing guidelines to get started.

## License
This project is licensed under the GPL-3.0 License - see the LICENSE [__LICENSE__](https://github.com/matt5797/SORT_CSharp/blob/main/LICENSE) file for details.

## Acknowledgments
This port is based on the original SORT algorithm developed by Alex Bewley. You can find the original Python implementation [__here__](https://github.com/abewley/sort). This C# version is a direct translation and all credit for the algorithm itself goes to the original authors.