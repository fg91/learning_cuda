# My solution to homework assignment 2
## Blurs an image on GPU
### Compile
```
mkdir build
cd build
cmake ..
make
```
### Usage
`./hw2 ../example.jpg`

### Runtime with and without using shared memory
Without: 12.8611msecs

With:

![](blurred.png)
![](example.jpg)
