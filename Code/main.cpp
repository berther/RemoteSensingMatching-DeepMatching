/*
Copyright (C) 2014 Jerome Revaud

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
*/
#include "std.h"
#include "image.h"
#include "io.h"
#include "deep_matching.h"
#include "main.h"
#include <thread>
#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

#include <climits>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <time.h>

int  every_patch_score = 0;

float longitude_start = 0;
float latitude_start = 0;
float longitude_end = 0;
float latitude_end = 0;

void  test()
{
   cv::Mat img = cv::Mat::zeros(100,100,CV_8UC3);
   cv::Rect rect(10, 10, 100, 100);
   cv::Mat img2 ;
   img(rect).copyTo(img2);
}

/*max返回矩阵最大值，rowIndex、colIndex返回最大值坐标*/
void maxMatrix (int **matrix, int row, int col, int *max, int *rowIndex, int *colIndex) 
{
    int i,j;
    *max = matrix[0][0];
    *rowIndex = *colIndex =0;
    for (i=0; i<row; i++) {
        for (j=0; j<col; j++) {
            if (matrix[i][j]>*max) {
                *max = matrix[i][j];
                *rowIndex = i;
               *colIndex = j;
            }  
        }
    }
}

void get_longitude_latitude(const int argc, const char **argv,float *longitude_start, float *latitude_start,float *longitude_end, float *latitude_end)
{
  int current_arg = 0;

  // parse options
  while(current_arg < argc)
  {
    const char* a = argv[current_arg++];
    #define isarg(key)  !strcmp(a,key)
    
    if(isarg("-lltude"))    
      {
        *longitude_start =  atof(argv[current_arg++]);
        *latitude_start = atof(argv[current_arg++]);
        *longitude_end = atof(argv[current_arg++]);
        *latitude_end = atof(argv[current_arg++]);
        break;
      }
  }
}

void usage(const int language)
{
  #define p(msg)  std_printf(msg "\n");
  p("usage:");
  switch(language){
    case EXE_OPTIONS:
      p("./deepmatching image1 image2 [options]");
      p("Compute the 'DeepMatching' between two images and print a list of")
      p("pair-wise point correspondences:")
      p("  x1 y1 x2 y2 score index ...")
      p("(index refers to the local maximum from which the match was retrieved)")
      p("Images must be in PPM, PNG or JPG format. Version 1.2.2")
      break;
    case MATLAB_OPTIONS:
      p("matches = deepmatching(image1, image2 [, options])")
      p("Compute the 'DeepMatching' between two images.")
      p("Images must be HxWx3 single matrices.")
      p("Options is an optional string argument ('' by default).")
      p("The function returns a matrix with 6 columns, each row being x1 y1 x2 y2 score index.")
      p("(index refers to the local maximum from which the match was retrieved)")
      p("Version 1.2.2")
      break;
    case PYTHON_OPTIONS:
      p("matches = deepmatching.deepmatching(image1, image2, options='')")
      p("Compute the 'DeepMatching' between two images.")
      p("Images must be HxWx3 numpy arrays (converted to float32).")
      p("Options is an optional string argument ('' by default).")
      p("The function returns a numpy array with 6 columns, each row being x1 y1 x2 y2 score index.")
      p("(index refers to the local maximum from which the match was retrieved)")
      p("Version 1.2.2")
      break;
  }
  p("")
  p("Options:")
  p("    -h, --help                 print this message")
//p("  HOG parameters (low-level pixel descriptor):")
//p("    -png_settings              (auto) recommended for uncompressed images")
//p("    -jpg_settings              (auto) recommended for compressed images")
//p("   in more details: (for fine-tuning)")
//p("    -hog.presm <f=1.0>         prior image smoothing")
//p("    -hog.midsm <f=1.0>         intermediate HOG smoothing")
//p("    -hog.sig <f=0.2>           sigmoid strength")
//p("    -hog.postsm <f=1.0>        final HOG-smoothing")
//p("    -hog.ninth <f=0.3>         robustness to pixel noise (eg. JPEG artifacts)")
  p("")
  p("  Matching parameters:")
//p("    -iccv_settings             settings used for the ICCV paper")
//p("    -improved_settings         (default) supposedly improved settings")
//p("   in more details: (for fine-tuning)")
  p("    -downscale/-R <n=1>        downsize the input images by a factor 2^n")
//p("    -overlap <n=999>           use overlapping patches in image1 from level n")
//p("    -subref <n=0>              0: denser sampling or 1: not of image1 patches")
  p("    -ngh_rad <n=0>             if n>0: restrict matching to n pxl neighborhood")
  p("    -nlpow <f=1.4>             non-linear rectification x := x^f")
//p("    -maxima_mode <n=0>         0: from all top cells / 1: from local maxima")
//p("    -min_level <n=2>           skip maxima in levels [0, 1, ..., n-1]")
  p("    -mem <n=1>                 if n>0: optimize memory footprint (bit unstable)")
//p("    -scoring_mode <n=1>        type of correspondence scoring mode (0/1)")
  p("")
  p("  Fully scale & rotation invariant DeepMatching:")
  p("    if either one of these options is used, then this mode is activated:")
  p("    -max_scale <factor=5>         max scaling factor")
  p("    -rot_range <from=0> <to=360>  rotation range")
  p("")
  p("  Other parameters:")
  p("    -resize <width> <height>   to resize input images beforehand")
  p("    -v                         increase verbosity")
  p("    -nt <n>                    multi-threading with <n> threads")
  p("    -lltude <n> <n> <n> <n>  longitude_start latitude_start  longitude_end latitude_end")
  if(language==EXE_OPTIONS) {
  p("    -out <file_name>           output correspondences in a file")
  exit(1);}
}

bool endswith(const char *str, const char *suffix)
{
    if(!str || !suffix)  return false;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if(lensuffix >  lenstr)  return false;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

image_t* rescale_image( image_t* im, int width, int height ) 
{
  image_t* res = image_new(width,height);
  image_resize_bilinear_newsize(res, im, width, height);
  image_delete(im);
  return res;
}


const char *parse_options(dm_params_t *params, scalerot_params_t *sr_params, bool *use_scalerot, float *fx, float *fy, const int argc, const char **argv, const int language, image_t **im1, image_t **im2) {
  int current_arg = 0;
  const char* out_filename = NULL;
  
  // parse options
  while(current_arg < argc)
  {
    const char* a = argv[current_arg++];
    #define isarg(key)  !strcmp(a,key)
    
    if(isarg("-h") || isarg("--help") )    usage(language);
  // HOG and patch parameters
  //else if(isarg("-hog.presm"))
  //  params->desc_params.presmooth_sigma = atof(argv[current_arg++]);
  //else if(isarg("-hog.sig"))
  //  params->desc_params.hog_sigmoid = atof(argv[current_arg++]);
  //else if(isarg("-hog.midsm"))
  //  params->desc_params.mid_smoothing = atof(argv[current_arg++]);
  //else if(isarg("-hog.postsm"))
  //  params->desc_params.post_smoothing = atof(argv[current_arg++]);
  //else if(isarg("-hog.ninth"))
  //  params->desc_params.ninth_dim = atof(argv[current_arg++]);
  //else if(isarg("-hog.nrmpix"))
  //  params->desc_params.norm_pixels = atof(argv[current_arg++]);
    else if(isarg("-png_settings")) { 
      params->desc_params.presmooth_sigma = 0; // no image smoothing since the image is uncompressed
      params->desc_params.hog_sigmoid = 0.2;
      params->desc_params.mid_smoothing = 1.5;
      params->desc_params.post_smoothing = 1;
      params->desc_params.ninth_dim = 0.1; } // low ninth_dim since image PSNR is high
    else if(isarg("-jpg_settings")) {
      params->desc_params.presmooth_sigma = 1; // smooth the image to remove jpg artifacts
      params->desc_params.hog_sigmoid = 0.2;
      params->desc_params.mid_smoothing = 1.5;
      params->desc_params.post_smoothing = 1;
      params->desc_params.ninth_dim = 0.3; } // higher ninth_dim because of pixel noise
  // matching parameters
    else if(isarg("-R") || isarg("-downscale"))
      params->prior_img_downscale = atoi(argv[current_arg++]);
  //else if(isarg("-overlap"))
  //  params->overlap = atoi(argv[current_arg++]);
  //else if(isarg("-subref"))
  //  params->subsample_ref = atoi(argv[current_arg++]);
    else if(isarg("-nlpow"))
      params->nlpow = atof(argv[current_arg++]);
    else if(isarg("-ngh_rad"))
      params->ngh_rad = atoi(argv[current_arg++]);
  // maxima parameters
  //else if(isarg("-maxima_mode"))
  //  params->maxima_mode = atoi(argv[current_arg++]);
    else if(isarg("-mem")) {
      params->low_mem = atoi(argv[current_arg++]); }
    //else if(isarg("-min_level"))
    //  params->min_level = atoi(argv[current_arg++]);
  //else if(isarg("-scoring_mode"))
  //  params->scoring_mode = atoi(argv[current_arg++]);
    //else if(isarg("-iccv_settings")) {
    //  params->prior_img_downscale = 2;
    //  params->overlap = 0; // overlap from level 0
    //  params->subsample_ref = 1;
    //  params->nlpow = 1.6;
    //  params->maxima_mode = 1;
    //  params->low_mem = 0;
    //  params->min_level = 2;
    //  params->scoring_mode = 0; }
    //else if(isarg("-improved_settings")) {
    //  params->prior_img_downscale = 1; // less down-scale
    //  params->overlap = 999; // no overlap
    //  params->subsample_ref = 0; // dense patch sampling at every level in first image
    //  params->nlpow = 1.4;
    //  params->maxima_mode = 0;
    //  params->low_mem = 1;
    //  params->min_level = 2;
    //  params->scoring_mode = 1; } // improved scoring
    //else if(isarg("-max_psize")) {
    //  params->max_psize = atoi(argv[current_arg++]); }
  // scale & rot invariant version
    else if(isarg("-scale") || isarg("-max_scale")) {
      *use_scalerot = true;
      float scale = atof(argv[current_arg++]);
      sr_params->max_sc0 = sr_params->max_sc1 = int(1 + 2*log2(scale)); }
    else if(isarg("-rot") || isarg("-rot_range")) {
      *use_scalerot = true;
      int min_rot = atoi(argv[current_arg++]);
      int max_rot = atoi(argv[current_arg++]);
      while( min_rot < 0 ) {
        min_rot += 360;
        max_rot += 360;
      }
      sr_params->min_rot = int(floor(0.5 + min_rot/45.));
      sr_params->max_rot = int(floor(1.5 + max_rot/45.));
      while( sr_params->max_rot - sr_params->min_rot > 8 )  
        sr_params->max_rot--;
      assert( sr_params->min_rot < sr_params->max_rot ); }
  // other parameters
    else if(isarg("-resize")) {
      assert((*im1)->width==(*im2)->width && (*im1)->height==(*im2)->height);
      int width = atoi(argv[current_arg++]);
      int height = atoi(argv[current_arg++]);
      *fx *= (*im1)->width / float(width);
      *fy *= (*im1)->height / float(height);
      *im1 = rescale_image(*im1, width, height);
      *im2 = rescale_image(*im2, width, height); }
    else if(isarg("-v"))
      params->verbose++;
    else if(isarg("-nt")) {
      params->n_thread = atoi(argv[current_arg++]);
      if (params->n_thread==0)
        params->n_thread = std::thread::hardware_concurrency(); }
    else if(language == EXE_OPTIONS && isarg("-out"))
      out_filename = argv[current_arg++];
    else  if(isarg("-lltude"))    
      {
        longitude_start =  atof(argv[current_arg++]);
        latitude_start = atof(argv[current_arg++]);
        longitude_end = atof(argv[current_arg++]);
        latitude_end = atof(argv[current_arg++]);
      }
    else {
      err_printf("error: unexpected parameter '%s'", a);
      exit(-1);
    }
  }
  
  if( *use_scalerot )
    assert( params->ngh_rad == 0 || !"max trans cannot be used in full scale and rotation mode");
  else
    if( params->subsample_ref && (!ispowerof2((*im1)->width) || !ispowerof2((*im1)->height)) ) {
      err_printf("WARNING: first image has dimension which are not power-of-2\n");
      err_printf("For improved results, you should consider resizing the images with '-resize <w> <h>'\n");
    }
  
  return out_filename;
}



int main(int argc, const char ** argv)
{
  if( argc<=2 || !strcmp(argv[1],"-h") || !strcmp(argv[1],"--help") )  usage(EXE_OPTIONS); 
   
   clock_t start_time=clock();
   const char * argv_name = argv[1];

    cv::Mat image1 =cv::imread( argv[1], 1 );
    cv::Mat image2 =cv::imread( argv[2], 1 );
/*****show image1***/
   // cv::namedWindow("match Input Image", CV_WINDOW_AUTOSIZE );
   // cv::imshow("match Input  Image", image1);
   //cv::waitKey(1000);

    int image1_width = image1.size().width;
    int image1_height = image1.size().height;
    int image2_width = image2.size().width;
    int image2_height = image2.size().height;

   int x = image2_width/image1_width;
   int y = image2_height/image1_height;

   int curt_x = 0;
   int curt_y = 0;
  std::string image_name;
     if(image2_width%image1_width > 0)
     {
     x = x +1;
     }
     if(image2_height%image1_height > 0)
     {
      y = y +1;
     }
//std_printf("%d %d\n",y,x);

  int score[y][x];//

/*
---------------------------------->   x
|
|
|
|
|
> y
*/
for (int j = 0; j< y; ++j) //col  x 
  for (int i= 0; i< x; ++i)//row   y  
{
  if (((image2_width-i*image1_width) < image1_width) &&  ((image2_width-i*image1_width) >0))
  {
   curt_x = image2_width-image1_width;
  }
  else
  {
        curt_x = i*image1_width;
  }

  if (((image2_height-j*image1_height) < image1_height) &&  ((image2_height-j*image1_height) >0))
  {
    curt_y = image2_height-image1_height;
  }
  else
  {
        curt_y = j*image1_height;                                        
  }

   cv::Rect rect(curt_x, curt_y, image1_width, image1_height);//x y width height   image1
   cv::Mat img2 ;
   image2(rect).copyTo(img2);
   // cv::namedWindow("Process Image", CV_WINDOW_AUTOSIZE );
   // cv::imshow("Process Image", img2);
   //cv::waitKey(1000);

   std::stringstream ss1,ss2;
   std::string str1,str2;
    ss1<<j;
    ss1>>str1;
    ss2<<i;
    ss2>>str2;

  image_name= "./CutImage/";
   image_name = image_name + str1 + "_" + str2 + ".png";  
   imwrite(image_name, img2);

  int current_arg = 3;
  image_t *im1=NULL, *im2=NULL;
  {
    color_image_t *cim1 = color_image_load(argv_name);//argv[1]
    const char * curt_image_name=image_name.c_str(); 
    // std::cout<<argv_name<<endl;  
    // std::cout<<curt_image_name<<endl;  
    color_image_t *cim2 = color_image_load(curt_image_name);//((const char*)(&(img2.data)) );//argv[2] "./CutImage/1.png" "./CutImage/1.jpg"
    //   cv::namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
    //  cv::imshow("Display Image", cim2);
    // cv::waitKey(1000);

    // Following deactivated because quite useless/dangerous in practice
    // default behavior == always using -jpg_settings
    
    //if( endswith(argv[1],"png") || endswith(argv[1],"PNG") )
    //  argv[--current_arg] = "-png_settings";  // set default
    //if( endswith(argv[1],"ppm") || endswith(argv[1],"PPM") )
    //  argv[--current_arg] = "-png_settings";  // set default
    //if( endswith(argv[1],"jpg") || endswith(argv[1],"JPG") )
    //  argv[--current_arg] = "-jpg_settings";  // set default
    //if( endswith(argv[1],"jpeg") || endswith(argv[1],"JPEG") )
    //  argv[--current_arg] = "-jpg_settings";  // set default
    
    im1 = image_gray_from_color(cim1);
    im2 = image_gray_from_color(cim2);
    color_image_delete(cim1);
    color_image_delete(cim2);
  }
  
  // set params to default
  dm_params_t params;
  set_default_dm_params(&params);
  scalerot_params_t sr_params;
  set_default_scalerot_params(&sr_params);
  bool use_scalerot = false;
  float fx=1, fy=1;
  
  // parse options
  const char* out_filename = parse_options(&params, &sr_params, &use_scalerot, &fx, &fy, argc-current_arg, 
                                           &argv[current_arg], EXE_OPTIONS, &im1, &im2);
  
  // compute deep matching
  float_image* corres = use_scalerot ? 
         deep_matching_scale_rot( im1, im2, &params, &sr_params ) : 
         deep_matching          ( im1, im2, &params, NULL );  // standard call
  
  // save result
  output_correspondences( out_filename, (corres_t*)corres->pixels, corres->ty, fx, fy );
  
  free_image(corres);
  image_delete(im1);
  image_delete(im2);
  score[j][i]  = 0;
  score[j][i] = get_every_image_score();
  //std::printf("main  %d\n", score[j][i]);
}
// for (int i = 0; i < y; ++i)
// {
//   for (int j = 0; j < x; ++j)
//   {
//     std::printf("score[%d][%d] = %d  ",i,j,score[i][j]);
//   }
//   std_printf("\n");
// }

int max_score = 0;

int rowIndex=0;
int colIndex = 0;

int  **p=new int *[y];//开辟行空间  
for(int i=0;i<y;i++)  
           p[i]=new int[x];//开辟列空间 
for(int i=0;i<y;i++)
{    //赋值  
        for(int j=0;j<x;j++)
        {  
            p[i][j]=score[i][j];
        }  
}  

// for (int i = 0; i < y; ++i)
// {
//   for (int j = 0; j < x; ++j)
//   {
//     std::printf("p [%d][%d] = %d  ",i,j,p[i][j]);
//   }
//   std_printf("\n");
// }

maxMatrix (p, y, x, &max_score,&rowIndex,&colIndex);
//std_printf("free before\n");
for(int i=0;i<y;i++) 
{
    delete [] p[i];  
  }
delete []p; 
//std_printf("free end\n");

for (int j= 0; j< y; ++j)
{
  for (int i = 0; i < x; ++i)
  {
    if (score[j][i] == max_score)
    {
     rowIndex = j;
     colIndex = i;
    }
  }
}

  //std::printf("max score [%d][%d] = %d\n",rowIndex,colIndex,max_score);
   std::stringstream ss1,ss2;
   std::string str1,str2;
    ss1<<rowIndex;
    ss1>>str1;
    ss2<<colIndex;
    ss2>>str2;

   image_name= "./CutImage/";
   image_name = image_name + str1 + "_" + str2 + ".png";  
   
   const char * match_result_image_name=image_name.c_str(); 
 
   cv::Mat  img_result;
   img_result =  cv::imread(match_result_image_name);
   //  cv::namedWindow("First match result", CV_WINDOW_AUTOSIZE );
   // cv::imshow("First match result", img_result);
   //cv::waitKey(0);

   int pointx = 0;
   int pointy = 0;
   if (colIndex * image1_width + image1_width <= image2_width)
   {
     pointx = colIndex * image1_width;
   }
   else 
      pointx =image2_width- image1_width;

    if (rowIndex * image1_height + image1_height <= image2_height)
   {
     pointy = rowIndex * image1_height; 
   }
   else 
      pointy =image2_height- image1_height;

   cv::Mat image_add;
   image2.copyTo(image_add);
   cv::rectangle(image_add,cvPoint(pointx,pointy),cvPoint(pointx + image1_width,pointy+image1_height),cvScalar(0,0,255),1,1,0);
   // cv::namedWindow("First match all result", CV_WINDOW_AUTOSIZE );
   // cv::imshow("First match all result", image_add);   
  
   //cv::waitKey(0);

   int index_x = colIndex;
   int index_y = rowIndex;
   //std_printf("%d %d  ",index_x,index_y);
   int new_image_start_x = -1;//new image start x zuobiao
   int new_image_start_y = -1;

    int new_image_width = -1;
    int new_image_height = -1;

   /***compute the new image's start x y***/
   if (  ((index_x == 0)  && (index_y == 0)) ||   ((index_x == 0)  && (index_y == (y-1) ) )  || ((index_x == (x-1)) && (index_y == (y-1))) || ((index_x == (x-1))  && (index_y == 0))  )  
   {/*4  ge jiao*/
     if  ( (index_x == 0)  && (index_y == 0) ) 
     {
       new_image_start_x  = 0;
       new_image_start_y = 0;

       new_image_width = image1_width*3/2;
       if (new_image_width >image2_width)//////////////////////////
       {
         new_image_width = image2_width;
       }
       new_image_height = image1_height*3/2;
       if (new_image_height > image2_height)
       {
         new_image_height = image2_height;
       }
     }
     else if (  (index_x == 0)  && (index_y == (y-1))  )/////////////////////////
     {
        new_image_start_x = 0;
       new_image_width = image1_width*3/2;
       if (new_image_width >= image2_width)
       {
         new_image_width = image2_width;
       }

       new_image_start_y = image2_height - image1_height*3/2;
       if (new_image_start_y <=0)
       {
          new_image_start_y = 0;
          new_image_height = image2_height;
       }
       else
          new_image_height = image1_height*3/2;
     }
     else if ((index_x == (x-1))  && (index_y == 0))//////////////////
     {
        new_image_start_x = image2_width - image1_width*3/2;
       if (new_image_start_x <=0)
       {
          new_image_start_x = 0;
          new_image_width = image2_width;
       }
       else
          new_image_width = image1_width*3/2;

       new_image_start_y = 0;
       new_image_height = image1_height*3/2;
       if (new_image_height >= image2_height)
       {
         new_image_height = image2_height;
       }
     }
     else if((index_x == (x-1)) && (index_y == (y-1)))////////////////////////////
     {
        new_image_start_x = image2_width - image1_width*3/2;
       if (new_image_start_x <=0)
       {
          new_image_start_x = 0;
          new_image_width = image2_width;
       }
       else
          new_image_width = image1_width*3/2;

       new_image_start_y = image2_height - image1_height*3/2;
       if (new_image_start_y <=0)
       {
          new_image_start_y = 0;
          new_image_height = image2_height;
       }
       else
          new_image_height = image1_height*3/2;
     }
   }
   else if( (index_y == 0 || index_y == (y-1))  && (index_x != 0 && index_x != (x-1) ) )/* x zhou   bian yuan kuai**/
   {
     if (index_y == 0)//index_y ==0
     {
       new_image_start_x = index_x * image1_width - image1_width/2;
       if (new_image_start_x <= 0)
       {
         new_image_start_x = 0;      
       }
       if(  (index_x*image1_width + image1_width*3/2) >=  image2_width)
         {
           new_image_width = image2_width - new_image_start_x;
         }
      else
          new_image_width =  index_x*image1_width - new_image_start_x + image1_width*3/2;


       new_image_start_y = 0;
       if (image1_height*3/2 >= image2_height)
       {
         new_image_height = image2_height;
       }
       else
        new_image_height = image1_height*3/2;

     }
     else if ( index_y == (y-1) ) //index_y = y-1
     {
      new_image_start_x = index_x * image1_width - image1_width/2;
       if (new_image_start_x <= 0)
       {
         new_image_start_x = 0;      
       }
       if(  (index_x*image1_width + image1_width*3/2) >=  image2_width)
         {
           new_image_width = image2_width - new_image_start_x;
         }
      else
          new_image_width =  index_x*image1_width - new_image_start_x + image1_width*3/2;
     
       new_image_start_y = image2_height - image1_height*3/2;
       if (new_image_start_y <= 0)
       {
         new_image_start_y = 0;
         new_image_height = image2_height;
       }
       else
          new_image_height = image1_height*3/2;

   }
  else if( (index_x == 0 || index_x == (x-1))  && (index_y != 0 && index_y != (y-1) ) ) /*y zhou de  bian yuan  kuai*/
   {
     if (index_x == 0)//index_x ==0
     {
       new_image_start_x = 0;
       if (image1_width*3/2 >= image2_width)
       {
         new_image_width = image2_width;
       }
       else
        new_image_width = image1_width*3/2;

       new_image_start_y = index_y * image1_height - image1_height/2;
       if (new_image_start_y <= 0)
       {
         new_image_start_y = 0;
       }
       if ( (index_y*image1_height + image1_height*3/2) >= image2_height )
       {
         new_image_height = image2_height - new_image_start_y;
       }
       else
          new_image_height = index_y*image1_height - new_image_start_y + image1_height*3/2;

     }
     else if ( index_x == (x-1) ) //index_x = x-1
     {
       new_image_start_x = image2_width - image1_width*3/2;
       if (new_image_start_x <= 0)
       {
         new_image_start_x = 0;
         new_image_width = image2_width;
       }
       else
          new_image_width = image2_width*3/2;       

       new_image_start_y = index_y * image1_height - image1_height/2;
       if (new_image_start_y <= 0)
       {
         new_image_start_y= 0;
       }
      if ((index_y*image1_height - image1_height*3/2)  >= image2_height)
       {
         new_image_height = image2_height - new_image_start_y;
       }
       else
        new_image_height = image2_height - new_image_start_y + image1_height*3/2;
     }
   }
}
else/****mid  image *****/
    {
      new_image_start_x = index_x*image1_width - image1_width/2;
      if (new_image_start_x <=0)
      {
        new_image_start_x = 0;
      }

      new_image_start_y = index_y * image1_height -image1_height/2;
      if(new_image_start_y <=0)
      {
        new_image_start_y = 0;
      }

         /**get the new image width and height ***/
      if ((image2_width - 2*image1_width ) >= new_image_start_x)
      {
        if (index_x == 0)
        {
          new_image_width = image1_width*3/2;
        }
        else
          new_image_width = 2*image1_width;
      }
      else
        new_image_width = image2_width - new_image_start_x;

      if ( (image2_height - 2*image1_height) >= new_image_start_y )
      {
        if (index_y == 0 )
        {
          new_image_height = image1_height*3/2;
        }
        else
          new_image_height = 2*image1_height;
      }
      else
        new_image_height = image2_height - new_image_start_y;

      
     }


/********new image ********/
 //  std_printf("%d  %d  %d  %d \n",new_image_start_x,new_image_start_y,new_image_width,new_image_height);
   cv::Rect rect(new_image_start_x, new_image_start_y, new_image_width, new_image_height);//x y width height   image1
   cv::Mat new_image ;
   image2(rect).copyTo(new_image);
   
   // cv::namedWindow("second match! new  Image", CV_WINDOW_AUTOSIZE );
   // cv::imshow("second match! new  Image", new_image);
  
  //cv::waitKey(1000);

   imwrite("./CutImage2/Second_Match_Input_Image.png", new_image);


if (1)
{
    int current_arg = 3;
    image_t *im1=NULL, *im2=NULL;
    {
      color_image_t *cim1 = color_image_load(argv_name);
      color_image_t *cim2 = color_image_load("./CutImage2/Second_Match_Input_Image.png");
       
      im1 = image_gray_from_color(cim1);
      im2 = image_gray_from_color(cim2);
      color_image_delete(cim1);
      color_image_delete(cim2);
    }
    
    // set params to default
    dm_params_t params;
    set_default_dm_params(&params);
    scalerot_params_t sr_params;
    set_default_scalerot_params(&sr_params);
    bool use_scalerot = false;
    float fx=1, fy=1;
    
    // parse options
    const char* out_filename = parse_options(&params, &sr_params, &use_scalerot, &fx, &fy, argc-current_arg, 
                                             &argv[current_arg], EXE_OPTIONS, &im1, &im2);
    
    // compute deep matching
    float_image* corres = use_scalerot ? 
           deep_matching_scale_rot( im1, im2, &params, &sr_params ) : 
           deep_matching          ( im1, im2, &params, NULL );  // standard call
    
    // save result
    output_correspondences( out_filename, (corres_t*)corres->pixels, corres->ty, fx, fy );
    
    free_image(corres);
    image_delete(im1);
    image_delete(im2);
}
    float shift_x = get_image_shift_x();
    float shift_y = get_image_shift_y();


   cv::Mat result_image_add = image2;
   float second_pointx = new_image_start_x +shift_x;
   float second_pointy = new_image_start_y + shift_y;
   //std::printf("%f %f\n",shift_x,shift_y );
   cv::rectangle(result_image_add,cvPoint(second_pointx,second_pointy),cvPoint(second_pointx + image1_width,second_pointy+image1_height),cvScalar(255,0,0),8,1,0);
   //cv::rectangle(result_image_add,cvPoint(new_image_start_x,new_image_start_y),cvPoint(new_image_start_x + image1_width,new_image_start_y+image1_height),cvScalar(0,255,255),1,1,0);
   
   //cv::namedWindow(" Second match  result", CV_WINDOW_AUTOSIZE );
   //cv::imshow("second match  result", result_image_add);

   imwrite("./CutImage2/Second_Match_Result_Image.png", result_image_add);


   //get_longitude_latitude(argc, argv,&longitude_start, &latitude_start,&longitude_end,&latitude_end);
   //std_printf("image2 longgitue latitude:%g,%g,%g,%g\n",longitude_start, latitude_start,longitude_end,latitude_end);

   float longitude_piexl = (longitude_end - longitude_start)/image2_width;
   float latirude_piexl = (latitude_end - latitude_start)/image2_height;

  float match_image_longitude_start = longitude_start  + longitude_piexl * shift_x;
  float match_image_longitude_end = longitude_start +  longitude_piexl * (shift_x+image1_width);
  float match_image_latitude_start = latitude_start +  latirude_piexl*shift_y;
  float match_image_latitude_end = latitude_start + latirude_piexl * (shift_y + image1_height);

  std_printf("image1 (longgitue latitude:%g,%g,%g,%g)\n",match_image_longitude_start,match_image_latitude_start,match_image_longitude_end,match_image_latitude_end);


   clock_t end_time=clock();
   std_printf("Running Time :{%g ms}\n",static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000);

   cv::waitKey(0);
  return 0;
}