# Assignment N 4

### Compile
```
mkdir build
cd build
cmake ..
make
```
### Usage
`./hw4 ../red_eye_effect_5.jpg ../red_eye_effect_template_5.jpg`

In `student_func.cu` I simply used `thrust::sort_by_key` to solve the problem. My implementation of *radix sort* can be found in the directory `working_examples_of_algorithms`.