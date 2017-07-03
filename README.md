# Cuda mandelbulb
Rendering the mandelbulb fractal with cuda.

# Scripts
 - sh build.sh
 - sh render.sh
 - sh upload.sh
 - sh test.sh

Test results are accumulated in the test_results.txt file

# Example

```cpp
pixel* h_screen_buff;
pixel* d_screen_buff;
check_result(cudaMallocHost(&h_screen_buff, num_pixels * sizeof(pixel)));
check_result(cudaMalloc(&d_screen_buff, num_pixels * sizeof(pixel)));

dim3 block_dim(width / group_width, height / group_height);
dim3 group_dim(group_width, group_height);

printf("Starting kernel execution...\n");
d_main<<<block_dim, group_dim>>>(d_screen_buff, width, height);

// Read screen buffer
check_result(cudaMemcpy(h_screen_buff, d_screen_buff, num_pixels * sizeof(pixel), cudaMemcpyDeviceToHost));

// Render
//for(size_t y = 0;y < height;y++) {
//      for(size_t x = 0;x < width;x++) {
//              printf("%i ", (int) h_screen_buff[y * width + x].x);
//      }
//      printf("\n");
//}

cudaFreeHost(h_screen_buff);
cudaFree(d_screen_buff);
```

