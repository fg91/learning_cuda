// "Copyright 2018 <Fabio M. Graetz>"
#include "./utils.h"

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

float *h_filter__;

size_t numRows() {
  return imageInputRGBA.rows;
}

size_t numCols() {
  return imageInputRGBA.cols;
}

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const string &filename) {
  checkCudaErrors(cudaFree(0));
  // read image
  Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "Could not open file: " << filename << endl;
    exit(1);
  }
  
  // convert to RGBA image
  cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

  // allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  *h_inputImageRGBA = reinterpret_cast<uchar4 *>(imageInputRGBA.ptr<unsigned char>(0));
  *h_outputImageRGBA = reinterpret_cast<uchar4 *>(imageOutputRGBA.ptr<unsigned char>(0));

  const size_t numPixels = imageInputRGBA.rows * imageInputRGBA.cols;

  // allocate memory on the device for input and output
  checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));

  // and make sure there is nothing left laying around
  checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0,
                             sizeof(uchar4) * numPixels));

  // copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA,
                             sizeof(uchar4) * numPixels,
                             cudaMemcpyHostToDevice));

  // for cleanup
  d_inputImageRGBA__ = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;

  // now create the filter we will use
  const int blurKernelWidth = 9;
  const float blurKernelSigma = 2.;

  *filterWidth = blurKernelWidth;

  // create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;  // for clean up

  float filterSum = 0.f;  // we will use this for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }
  
  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }

  // blurred
  checkCudaErrors(cudaMalloc(d_redBlurred, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_greenBlurred, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueBlurred, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_redBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_blueBlurred, 0, sizeof(unsigned char) * numPixels));
}

void cleanup() {
  // free memory on device
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
  delete [] h_filter__;
}

void postProcess(const string &output_file, uchar4 *data_ptr)  {
  Mat output(imageInputRGBA.rows, imageInputRGBA.cols, CV_8UC4,
             reinterpret_cast<void*>(data_ptr));
  // write the output image
  Mat imageOutputBGR;
  cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
  cv::imwrite(output_file.c_str(), imageOutputBGR);
}
