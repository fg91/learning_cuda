// "Copyright 2018 <Fabio M. Graetz>"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using cv::Mat;

///////////////////

cv::Mat imageRGBA;
cv::Mat imageGrey;


void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols)
{
  for (size_t r = 0; r < imageRGBA.rows; ++r) {
    for (size_t c = 0; c < imageRGBA.cols; ++c) {
      uchar4 rgba = rgbaImage[r * imageRGBA.cols + c];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[r * imageRGBA.cols + c] = channelSum;
    }
  }
}


uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const string &filename) {
  checkCudaErrors(cudaFree(0));

  Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "Could not open file: " << filename << endl;
    exit(1);
  }

  // convert to RGBA image
  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  // allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = imageRGBA.rows * imageRGBA.cols;

  // allocate memory on the device for input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  // and make sure there is nothing left laying around
  checkCudaErrors(cudaMemset(*d_greyImage, 0,
                             sizeof(unsigned char) * numPixels));

  // copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage,
                             sizeof(uchar4) * numPixels,
                             cudaMemcpyHostToDevice));
  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void cleanup() {
  // free memory on device
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

void postProcess(const string &output_file, unsigned char *data_ptr) {
  Mat output(imageRGBA.rows, imageRGBA.cols, CV_8UC1, (void*)data_ptr);
  // write the output image
  cv::imwrite(output_file.c_str(), output);

}


///////////////////////


int main(int argc, char **argv) {
  // pointers for input and output image on host and device
  uchar4 *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  // input and output filenames
  string input_file;
  string output_file;

  switch (argc) {
    case 2:
      input_file = string(argv[1]);
      output_file = "gray_scale.png";
      break;
    case 3:
      input_file = string(argv[1]);
      output_file = string(argv[2]);
      break;
    default:
      cerr << "Usage: ./HW1 input_file [output_filename]" << endl;
      exit(1);
  }
  // load the image and give us input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage,
             & d_rgbaImage, &d_greyImage,
             input_file);

  referenceCalculation(h_rgbaImage, h_greyImage, imageRGBA.rows, imageRGBA.cols);
  
  postProcess(output_file, h_greyImage);
  
  cleanup();
}
