# DeepMatching (CPU)

Two remote sensing image matching of different sizes based on deep convolution matching algorithm（Deep Matching）

Environment：

   ubuntu14.04

   cafee

   C++

Compile：
make clean all

Use：
./deepmatching 2015-c.jpg 2017_2000x2545.jpg -downscale 1 -lltude 97 28 105 20 -v
   
                           图片1      图片2                         经纬度 图片2左上角点的经纬度 右下角点的经纬度


Based:

Forked from Revaud et al.'s *DeepMatching: Hierarchical Deformable Dense Matching*. ([Project page](https://thoth.inrialpes.fr/src/deepmatching/)). 

Python wrapper works for both Python 3.5 and Python 2.7. Change `PYTHON_VERSION=3.5` in Makefile to respective version number desired.
