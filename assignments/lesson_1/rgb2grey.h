// "Copyright 2018 <Fabio M. Graetz>"
#ifndef RGB2GREY_H
#define RGB2GREY_H

void solution_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
                            uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage,
                            size_t numRows, size_t numCols,
                            int blockWidth);
#endif  // RGB2GREY_H
