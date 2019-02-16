//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */
#include "utils.h"
#include <thrust/host_vector.h>

using std::swap;

__device__
int2 calcThreadCoordinates() {
  return make_int2(threadIdx.x + blockIdx.x * blockDim.x,
		   threadIdx.y + blockIdx.y * blockDim.y);
}

__device__
int calcIndexRowMajorLayout(int col, int row, size_t numCols) {
  return row * numCols + col;
}

__device__
bool inBoundaries(int col, int row, const size_t numRows, const size_t numCols) {
  return ((col >= 0) && (col < numCols) && (row >= 0) && (row < numRows));
}

__device__
bool inMask(uchar4 pixel) {
  return (pixel.x != 255 || pixel.y != 255 || pixel.z != 255);
}

__global__
void calcMaskBorder(const uchar4 * d_sourceImg,
		    const size_t numRowsSource, const size_t numColsSource,
		    bool * d_pixel_is_interior, bool * d_pixel_is_border) {
  
  int2 threadCoords = calcThreadCoordinates();
  int indexRowMajorLayout = calcIndexRowMajorLayout(threadCoords.x,
						    threadCoords.y,
						    numColsSource);
  
  if(!inBoundaries(threadCoords.x, threadCoords.y, numRowsSource, numColsSource)) {
    return;
  }
  
  if (inMask(d_sourceImg[indexRowMajorLayout])) {
    /*
      For every pixel in the mask;
      count how many neighbours a pixel has in total
      and then how many are in the mask. If the count is equal,
      the pixel is in the interiour. If the count is not equal but 
      the number of neighbours in the mask is > 0, then this pixel belongs
      to the border.
     */
    int neighboursInMask = 0;
    int neighboursInBoundaries = 0;

    // top neighbour
    if (inBoundaries(threadCoords.x, threadCoords.y - 1, numRowsSource, numColsSource)) {
      neighboursInBoundaries++;
      if (inMask(d_sourceImg[calcIndexRowMajorLayout(threadCoords.x,
						     threadCoords.y - 1,
						     numColsSource)])) {
	neighboursInMask++;
      }
    }

    // bottom neighbour
    if (inBoundaries(threadCoords.x, threadCoords.y + 1, numRowsSource, numColsSource)) {
      neighboursInBoundaries++;
      if (inMask(d_sourceImg[calcIndexRowMajorLayout(threadCoords.x,
						     threadCoords.y + 1,
						     numColsSource)])) {
	neighboursInMask++;
      }
    }

    // left neighbour
    if (inBoundaries(threadCoords.x - 1, threadCoords.y, numRowsSource, numColsSource)) {
      neighboursInBoundaries++;
      if (inMask(d_sourceImg[calcIndexRowMajorLayout(threadCoords.x - 1,
						     threadCoords.y,
						     numColsSource)])) {
	neighboursInMask++;
      }
    }

    // right neighbour
    if (inBoundaries(threadCoords.x + 1, threadCoords.y, numRowsSource, numColsSource)) {
      neighboursInBoundaries++;
      if (inMask(d_sourceImg[calcIndexRowMajorLayout(threadCoords.x + 1,
						     threadCoords.y,
						     numColsSource)])) {
	neighboursInMask++;
      }
    }

    if (neighboursInBoundaries == neighboursInMask) {
      d_pixel_is_interior[indexRowMajorLayout] = true;
    } else if(neighboursInMask > 0){
      d_pixel_is_border[indexRowMajorLayout] = true;
    }
  }
}

__global__
void separateColorChannels(const uchar4 * d_inputImg,
			   float * red,
			   float * green,
			   float * blue,
			   const size_t numRowsSource, const size_t numColsSource) {
  
  int2 threadCoords = calcThreadCoordinates();
  if(!inBoundaries(threadCoords.x, threadCoords.y, numRowsSource, numColsSource)) {
    return;
  }
  
  int indexRowMajorLayout = calcIndexRowMajorLayout(threadCoords.x,
						    threadCoords.y,
						    numColsSource);
  uchar4 pixel = d_inputImg[indexRowMajorLayout];

  red[indexRowMajorLayout]   = static_cast<float>(pixel.x);
  green[indexRowMajorLayout] = static_cast<float>(pixel.y);
  blue[indexRowMajorLayout]  = static_cast<float>(pixel.z);
}

__global__
void recombineColorChannels(uchar4 * d_outputImg,
			    const float * red,
			    const float * green,
			    const float * blue,
			    const size_t numRowsSource, const size_t numColsSource) {
  
  int2 threadCoords = calcThreadCoordinates();
  if(!inBoundaries(threadCoords.x, threadCoords.y, numRowsSource, numColsSource)) {
    return;
  }
  
  int indexRowMajorLayout = calcIndexRowMajorLayout(threadCoords.x,
						    threadCoords.y,
						    numColsSource);
  d_outputImg[indexRowMajorLayout].x = static_cast<char>(red[indexRowMajorLayout]);
  d_outputImg[indexRowMajorLayout].y = static_cast<char>(green[indexRowMajorLayout]);
  d_outputImg[indexRowMajorLayout].z = static_cast<char>(blue[indexRowMajorLayout]);
}

__global__
void jacobi(float * d_ImageGuess_prev,
	    float * d_ImageGuess_next,
	    const bool * d_pixel_is_border,
	    const bool * d_pixel_is_interior,
	    float * d_source,
	    float * d_dest,
	    size_t numRows, size_t numCols) {
  /*
    A: sum of I_k's neighbours (in the strict interior)
    B: sum of T's neighbours (corresponding pixel value in the target image and only if in BORDER)
    C: sum of the differences of s and it's neighbours where s is the corresponding pixel value in the source image)
    D: number of I_k's neighbours (can be less than 4 if at the border of the image)
   */

  int2 threadCoords = calcThreadCoordinates();
  int index = calcIndexRowMajorLayout(threadCoords.x,
				      threadCoords.y,
				      numCols);
  
  if(!inBoundaries(threadCoords.x, threadCoords.y, numRows, numCols)) {
    return;
  }

  if (d_pixel_is_interior[index] == true) {
    int indexNeighbour;
    float sourcePixel = d_source[index];
    float A = 0.0f, B = 0.0f, C = 0.0f, D = 0.0f;

    // right neighbour
    if(inBoundaries(threadCoords.x + 1, threadCoords.y, numRows, numCols)) {
	indexNeighbour = calcIndexRowMajorLayout(threadCoords.x + 1, threadCoords.y, numCols);
	if(d_pixel_is_interior[indexNeighbour] == true) {
	  A += d_ImageGuess_prev[indexNeighbour];
	} else if (d_pixel_is_border[indexNeighbour] == true) {
	  B += d_dest[indexNeighbour];
	}
	C += sourcePixel - d_source[indexNeighbour];
	D += 1.;
    }

    // left neighbour
    if(inBoundaries(threadCoords.x - 1, threadCoords.y, numRows, numCols)) {
	indexNeighbour = calcIndexRowMajorLayout(threadCoords.x - 1, threadCoords.y, numCols);
	if(d_pixel_is_interior[indexNeighbour] == true) {
	  A += d_ImageGuess_prev[indexNeighbour];
	} else if (d_pixel_is_border[indexNeighbour] == true) {
	  B += d_dest[indexNeighbour];
	}
	C += sourcePixel - d_source[indexNeighbour];
	D += 1.;
    }

    // top neighbour
    if(inBoundaries(threadCoords.x, threadCoords.y - 1, numRows, numCols)) {
	indexNeighbour = calcIndexRowMajorLayout(threadCoords.x, threadCoords.y - 1, numCols);
	if(d_pixel_is_interior[indexNeighbour] == true) {
	  A += d_ImageGuess_prev[indexNeighbour];
	} else if (d_pixel_is_border[indexNeighbour] == true) {
	  B += d_dest[indexNeighbour];
	}
	C += sourcePixel - d_source[indexNeighbour];
	D += 1.;
    }

    // bottom neighbour
    if(inBoundaries(threadCoords.x, threadCoords.y + 1, numRows, numCols)) {
	indexNeighbour = calcIndexRowMajorLayout(threadCoords.x, threadCoords.y + 1, numCols);
	if(d_pixel_is_interior[indexNeighbour] == true) {
	  A += d_ImageGuess_prev[indexNeighbour];
	} else if (d_pixel_is_border[indexNeighbour] == true) {
	  B += d_dest[indexNeighbour];
	}
	C += sourcePixel - d_source[indexNeighbour];
	D += 1.;
    }
    d_ImageGuess_next[index] = max(0.0f, min((A + B + C)/D, 255.0f));
  } else {
    d_ImageGuess_next[index] = d_dest[index];  // if outside of mask
  }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
  const int SIZE = 32;
  const dim3 blockSize(SIZE, SIZE);
  const dim3 gridSize(numColsSource/blockSize.x + 1, numRowsSource/blockSize.y + 1);

  // Allocate device memory for source and destination image
  const size_t numPixels = numRowsSource * numColsSource;

  uchar4 * d_sourceImg;
  uchar4 * d_destImg;
  uchar4 * d_blendedImg;

  checkCudaErrors(cudaMalloc(&d_sourceImg, numPixels * sizeof(uchar4)));
  checkCudaErrors(cudaMalloc(&d_destImg, numPixels * sizeof(uchar4)));
  checkCudaErrors(cudaMalloc(&d_blendedImg, numPixels * sizeof(uchar4)));

  // Copy source and destination image to device
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice));
  
  /* To Recap here are the steps you need to implement */
  
  /*    1) Compute a mask of the pixels from the source image to be copied */
  /*       The pixels that shouldn't be copied are completely white, they */
  /*       have R=255, G=255, B=255.  Any other pixels SHOULD be copied. */

  /*    2) Compute the interior and border regions of the mask.  An interior */
  /*       pixel has all 4 neighbors also inside the mask.  A border pixel is */
  /*       in the mask itself, but has at least one neighbor that isn't. */

  bool * d_pixel_is_border;
  bool * d_pixel_is_interior;

  checkCudaErrors(cudaMalloc(&d_pixel_is_border, numPixels * sizeof(bool)));
  checkCudaErrors(cudaMalloc(&d_pixel_is_interior, numPixels * sizeof(bool)));

  calcMaskBorder<<<gridSize, blockSize>>>(d_sourceImg,
					  numRowsSource, numColsSource,
					  d_pixel_is_interior, d_pixel_is_border);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  /*    3) Separate out the incoming image into three separate channels */
  float *d_sourceImgR, *d_sourceImgG, *d_sourceImgB, *d_destImgR, *d_destImgG, *d_destImgB;
  checkCudaErrors(cudaMalloc(&d_sourceImgR, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_sourceImgG, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_sourceImgB, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_destImgR, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_destImgG, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_destImgB, numPixels * sizeof(float)));

  separateColorChannels<<<gridSize, blockSize>>>(d_sourceImg,
  						 d_sourceImgR,
  						 d_sourceImgG,
  						 d_sourceImgB,
  						 numRowsSource, numColsSource);
						 
  separateColorChannels<<<gridSize, blockSize>>>(d_destImg,
  						 d_destImgR,
  						 d_destImgG,
  						 d_destImgB,
  						 numRowsSource, numColsSource);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
						 
  /*    4) Create two float(!) buffers for each color channel that will */
  /*       act as our guesses.  Initialize them to the respective color */
  /*       channel of the source image since that will act as our intial guess. */
  float *d_guessBuffer1R, *d_guessBuffer1G, *d_guessBuffer1B;
  float *d_guessBuffer2R, *d_guessBuffer2G, *d_guessBuffer2B;

  checkCudaErrors(cudaMalloc(&d_guessBuffer1R, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_guessBuffer1G, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_guessBuffer1B, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_guessBuffer2R, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_guessBuffer2G, numPixels * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_guessBuffer2B, numPixels * sizeof(float)));

  // copy source imge to initial guess buffer
  checkCudaErrors(cudaMemcpy(d_guessBuffer1R, d_sourceImgR, numPixels * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_guessBuffer1G, d_sourceImgG, numPixels * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_guessBuffer1B, d_sourceImgB, numPixels * sizeof(float), cudaMemcpyDeviceToDevice));
  
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  /*    5) For each color channel perform the Jacobi iteration described  */
  /*       above 800 times. */

  const int ITERATIONS = 800;

  for (int i = 0; i < ITERATIONS; i++) {
    jacobi<<<gridSize, blockSize>>>(d_guessBuffer1R,
				    d_guessBuffer2R,
				    d_pixel_is_border,
				    d_pixel_is_interior,
				    d_sourceImgR,
				    d_destImgR,
				    numRowsSource, numColsSource);
    swap(d_guessBuffer1R, d_guessBuffer2R);

    jacobi<<<gridSize, blockSize>>>(d_guessBuffer1G,
				    d_guessBuffer2G,
				    d_pixel_is_border,
				    d_pixel_is_interior,
				    d_sourceImgG,
				    d_destImgG,
				    numRowsSource, numColsSource);
    swap(d_guessBuffer1G, d_guessBuffer2G);


    jacobi<<<gridSize, blockSize>>>(d_guessBuffer1B,
				    d_guessBuffer2B,
				    d_pixel_is_border,
				    d_pixel_is_interior,
				    d_sourceImgB,
				    d_destImgB,
				    numRowsSource, numColsSource);
    swap(d_guessBuffer1B, d_guessBuffer2B);


  }
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  /*    6) Create the output image by replacing all the interior pixels */
  /*       in the destination image with the result of the Jacobi iterations. */
  /*       Just cast the floating point values to unsigned chars since we have */
  /*       already made sure to clamp them to the correct range. */

  recombineColorChannels<<<gridSize, blockSize>>>(d_blendedImg,
						  d_guessBuffer1R,
						  d_guessBuffer1G,
						  d_guessBuffer1B,
						  numRowsSource, numColsSource);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, numPixels * sizeof(uchar4), cudaMemcpyDeviceToHost));

  /*     Since this is final assignment we provide little boilerplate code to */
  /*     help you.  Notice that all the input/output pointers are HOST pointers. */

  /*     You will have to allocate all of your own GPU memory and perform your own */
  /*     memcopies to get data in and out of the GPU memory. */

  /*     Remember to wrap all of your calls with checkCudaErrors() to catch any */
  /*     thing that might go wrong.  After each kernel call do: */

  /*     cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); */

  /*     to catch any errors that happened while executing the kernel. */

  checkCudaErrors(cudaFree(d_sourceImg));
  checkCudaErrors(cudaFree(d_destImg));
  checkCudaErrors(cudaFree(d_blendedImg));
  checkCudaErrors(cudaFree(d_pixel_is_border));
  checkCudaErrors(cudaFree(d_pixel_is_interior));
  checkCudaErrors(cudaFree(d_sourceImgR));
  checkCudaErrors(cudaFree(d_sourceImgG));
  checkCudaErrors(cudaFree(d_sourceImgB));
  checkCudaErrors(cudaFree(d_destImgR));
  checkCudaErrors(cudaFree(d_destImgG));
  checkCudaErrors(cudaFree(d_destImgB));
  checkCudaErrors(cudaFree(d_guessBuffer1R));
  checkCudaErrors(cudaFree(d_guessBuffer1G));
  checkCudaErrors(cudaFree(d_guessBuffer1B));
  checkCudaErrors(cudaFree(d_guessBuffer2R));
  checkCudaErrors(cudaFree(d_guessBuffer2G));
  checkCudaErrors(cudaFree(d_guessBuffer2B));
}


/*
  https://github.com/ibebrett/CUDA-CS344/blob/master/Problem%20Sets/Problem%20Set%206/student_func.cu
 */
