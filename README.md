NDT variants
====

# 3D NDT D2D registration
This is WIP. I am extracting minimum implementation from https://github.com/OrebroUniversity/perception_oru-release/tree/debian/indigo/ndt_registration (AASS Research Center, Orebro University, 2010, 3-clause BSD License)

Stoyanov et al., "Fast and accurate scan registration through minimization of the distance between compact 3D NDT representations"
(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.818.9757&rep=rep1&type=pdf)


# Build and Test

```sh
docker build -t pcl .

# For WSL2
docker run -it --rm -v $(pwd):/home -e DISPLAY=$DISPLAY -e PULSE_SERVER=$PULSE_SERVER -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -v /mnt/wslg:/mnt/wslg -v /tmp/.X11-unix:/tmp/.X11-unix pcl bash
```
