# FastSLAM

## Description

This is a Python implementation of the FastSLAM algorithm for Simultaneous Localisation and Mapping (SLAM).

### SLAM resources

If you are interested in learning more about SLAM, here are some of the resources I have used.

#### Articles
- [**_Bayesian Filtering for Location Estimation_**](http://www.irisa.fr/aspi/legland/ref/fox03a.pdf) -- Dieter Fox et al., 2003
- [**_Simultaneous Localization and Mapping (SLAM) Part I: The Essential Algorithms_**](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/Durrant-Whyte_Bailey_SLAM-tutorial-I.pdf) -- Hugh Durrant-Whyte & Tim Bailey, 2006
- [**_Simultaneous Localization and Mapping (SLAM) Part II: State of the Art_**](https://pdfs.semanticscholar.org/27d4/6db7ed4e96944080052b761c62102f26b23f.pdf) -- Hugh Durrant-Whyte & Tim Bailey, 2006
- [**_FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem_**](http://robots.stanford.edu/papers/montemerlo.fastslam-tr.pdf) - Michael Montemerlo et. al., 2002
- [**_FastSLAM 2.0: An Improved Particle Filtering Algorithm for Simultaneous Localization and Mapping that Provably Converges_**](http://robots.stanford.edu/papers/Montemerlo03a.pdf) - Michael Montemerlo et. al., 2003

#### Books
- [**_Probabilistic Robotics_**](http://www.probabilistic-robotics.org/) -- Sebastian Thrun et al., 2006
- [**_Introduction to Autonomous Mobile Robots_**](https://mitpress.mit.edu/books/introduction-autonomous-mobile-robots-second-edition) -- Roland Siegwart & Illah R. Nourbakhsh, 2004

#### Webpages
- [**_PythonRobotics_**](https://pythonrobotics.readthedocs.io/en/latest/getting_started.html) -- Atsushi Sakai et. al.

#### Videos
- Start with these
    - [**_Particle Filter Explained Without Equations_**](https://www.youtube.com/watch?v=aUkBa1zMKv4&t=5s)
    - [**_Particle Filters Explained_**](https://www.youtube.com/watch?v=sz7cJuMgKFg)
    - [**_Autonomous Navigation, Part 2: Understanding the Particle Filter_**](https://www.youtube.com/watch?v=NrzmH_yerBU)
- Then these
    - [**_Cyrill Stachniss' SLAM lectures_**](https://www.youtube.com/watch?v=U6vr3iNrwRA&list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_)
        - Especially [__Lecture 12__](https://www.youtube.com/watch?v=Tz3pg3d1TIo&list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_&index=14) on FastSLAM
    - [**_Claus Brenner's SLAM course_**](https://www.youtube.com/watch?v=B2qzYCeT9oQ&list=PLpUPoM7Rgzi_7YWn14Va2FODh7LzADBSm)
        - Especially [__Unit G__](https://www.youtube.com/watch?v=9WyrWJcvneE&list=PLpUPoM7Rgzi_7YWn14Va2FODh7LzADBSm&index=60) on FastSLAM

## Requirements
- Assumes an existing ROS 2 installation and workspace (Dashing or newer)

## Installation

1. Clone this repository into the `/src` directory of your ROS 2 workspace
2. Open a terminal and source your ROS installation
    - `source /opt/ros/dashing/setup.bash`
3. Build
    - `colcon build --symlink-install`

## Usage
After you have run `colcon build`:
1. Open 2 new tabs in your terminal and source the workspace in each
    - `. install/setup.bash`
2. In one of the new tabs play the rosbag
    - `ros2 bag play <path_to_rosbag>`
3. In the other new tab run the FastSLAM node
    - `ros2 run fastslam main`

## Authors

- Isaac Vander Sluis - [@ibvandersluis](https://www.github.com/ibvandersluis)