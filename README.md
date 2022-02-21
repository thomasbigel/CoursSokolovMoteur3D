# CoursSokolovMoteur3D

Follow https://github.com/ssloy/tinyrenderer lessons

## Requirement

- c/c++ compiler
- OpenMP

## Launch
In a console :
```bash
# clone repository...
git clone https://github.com/thomasbigel/CoursSokolovMoteur3D
# ...and we move into the directory
cd CoursSokolovMoteur3D

# We launch the makefile generated to setup the application
make

```
How to use: ./main (+optional path to models (default used: obj/african_head.obj)):  

=======

#### Some Images Output
```bash
./main "obj/diablo3_pose/diablo3_pose.obj" "obj/floor.obj"
```
![image output](https://github.com/thomasbigel/CoursSokolovMoteur3D/blob/readmeImage/shoots/diablo3_pose%20with%20floor%20and%20good%20shadow/output.png)

```bash
./main "obj/boggie/body.obj" "obj/boggie/head.obj" "obj/boggie/eyes.obj" "obj/floor.obj"
```
![image output](https://github.com/thomasbigel/CoursSokolovMoteur3D/blob/readmeImage/shoots/boggie%20with%20good%20shadow%20and%20floor/output.png)

More in the shoots directory
