#include <stdio.h>

typedef float3 pixel;

typedef struct {
	float3 o;
	float3 d;
} ray;

__global__ void d_main(
	pixel* screen_buffer,
	const size_t width,
	const size_t height
) {
	size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
	size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;

	screen_buffer[y * width + x] = make_float3(255.0f, 255.0f, 255.0f);
}

int main(int argc, char** argv) {
	printf("Mandelbulb\n");
	
	size_t width = 512;
	size_t height = 512;
	size_t num_pixels = width * height;

	size_t group_width = 1;
	size_t group_height = 1;

	// Setup buffers
	pixel* h_screen_buff;
	pixel* d_screen_buff;
	cudaMallocHost(&h_screen_buff, num_pixels * sizeof(pixel));
	cudaMalloc(&d_screen_buff, num_pixels * sizeof(pixel));

	dim3 block_dim(width / group_width, height / group_height);
	dim3 group_dim(group_width, group_height);

	// Execute on devicie
	printf("Starting kernel execution...\n");
	d_main<<<block_dim, group_dim>>>(d_screen_buff, width, height);
	printf("Kernel execution ended.\n");

	// Read screen buffer from device
	cudaMemcpy(h_screen_buff, d_screen_buff, num_pixels * sizeof(pixel), cudaMemcpyDeviceToHost);	

	cudaFreeHost(h_screen_buff);
	cudaFree(d_screen_buff);

	return 0;
}

