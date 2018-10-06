// "Copyright 2018 <Fabio M. Graetz>"
#include "./utils.h"

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

cv::Mat imageRGBA;
cv::Mat imageGrey;

size_t numRows() {
  return imageRGBA.rows;
}

size_t numCols() {
  return imageRGBA.cols;
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const string &filename) {
  checkCudaErrors(cudaFree(0));
  // read image
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

  *inputImage = reinterpret_cast<uchar4 *>(imageRGBA.ptr<unsigned char>(0));
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

  // for cleanup
  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void cleanup() {
  // free memory on device
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

void postProcess(const string &output_file, unsigned char *data_ptr)  {
  Mat output(imageRGBA.rows, imageRGBA.cols, CV_8UC1,
             reinterpret_cast<void*>(data_ptr));
  // write the output image
  cv::imwrite(output_file.c_str(), output);
}

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols) {
  for (size_t r = 0; r < imageRGBA.rows; ++r) {
    for (size_t c = 0; c < imageRGBA.cols; ++c) {
      uchar4 rgba = rgbaImage[r * imageRGBA.cols + c];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[r * imageRGBA.cols + c] = channelSum;
    }
  }
}
