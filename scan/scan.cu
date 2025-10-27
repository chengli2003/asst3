#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upSweep(int* data, int two_d, int N) {
    int two_dplus1 = 2 * two_d;
    
    // Each thread handles one iteration of the parallel for loop
    // Use long long to avoid integer overflow
    long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * two_dplus1;
    
    if (i + two_dplus1 - 1 < N) {
        data[i + two_dplus1 - 1] += data[i + two_d - 1];
    }
}

__global__ void downSweep(int* data, int two_d, int N) {
    int two_dplus1 = 2 * two_d;
    
    // Each thread handles one iteration of the parallel for loop
    // Use long long to avoid integer overflow
    long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * two_dplus1;
    
    if (i + two_dplus1 - 1 < N) {
        int temp = data[i + two_d - 1];
        data[i + two_d - 1] = data[i + two_dplus1 - 1];
        data[i + two_dplus1 - 1] += temp;
    }
}

__global__ void setLastElementToZero(int* data, int N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        data[N - 1] = 0;
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    // Round up to next power of 2 for the algorithm
    // The arrays are allocated with this size, so we can safely use it
    int rounded_N = nextPow2(N);

    // Step 1: Up-sweep (reduce) phase
    for (int two_d = 1; two_d <= rounded_N/2; two_d *= 2) {
        int iterations = rounded_N / (2 * two_d);
        int numBlocks = (iterations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        if (numBlocks > 0) {
            upSweep<<<numBlocks, THREADS_PER_BLOCK>>>(result, two_d, rounded_N);
            cudaCheckError(cudaDeviceSynchronize());
        }
    }

    // Set the last element to 0 for exclusive scan
    setLastElementToZero<<<1, 1>>>(result, rounded_N);
    cudaCheckError(cudaDeviceSynchronize());

    // Step 2: Down-sweep phase
    for (int two_d = rounded_N/2; two_d >= 1; two_d /= 2) {
        int iterations = rounded_N / (2 * two_d);
        int numBlocks = (iterations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        if (numBlocks > 0) {
            downSweep<<<numBlocks, THREADS_PER_BLOCK>>>(result, two_d, rounded_N);
            cudaCheckError(cudaDeviceSynchronize());
        }
    }

}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// Kernel to create flags for consecutive equal elements
__global__ void createFlags(int* input, int* flags, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length - 1) {
        flags[idx] = (input[idx] == input[idx + 1]) ? 1 : 0;
    } else if (idx == length - 1) {
        flags[idx] = 0;  // Last element can't have a repeat
    }
}

// Kernel to compact the results using scan output
__global__ void compactResults(int* flags, int* scan_result, int* output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length - 1 && flags[idx] == 1) {
        output[scan_result[idx]] = idx;
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    // Allocate temporary arrays (using power-of-2 size)
    int rounded_length = nextPow2(length);
    int* flags;
    int* scan_result;

    cudaCheckError(cudaMalloc((void**)&flags, sizeof(int) * rounded_length));
    cudaCheckError(cudaMalloc((void**)&scan_result, sizeof(int) * rounded_length));

    // Initialize arrays to 0
    cudaCheckError(cudaMemset(flags, 0, sizeof(int) * rounded_length));
    cudaCheckError(cudaMemset(scan_result, 0, sizeof(int) * rounded_length));
    
    // Step 1: Create flags for consecutive equal elements
    int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    createFlags<<<numBlocks, THREADS_PER_BLOCK>>>(device_input, flags, length);
    cudaCheckError(cudaDeviceSynchronize());
    
    // Step 2: Exclusive scan on flags to get output positions
    // Copy flags to scan_result for in-place scan
    cudaCheckError(cudaMemcpy(scan_result, flags, sizeof(int) * rounded_length, cudaMemcpyDeviceToDevice));
    exclusive_scan(flags, rounded_length, scan_result);
    
    // Step 3: Compact results - write indices to output array
    compactResults<<<numBlocks, THREADS_PER_BLOCK>>>(flags, scan_result, device_output, length);
    cudaCheckError(cudaDeviceSynchronize());
    
    // Step 4: Get the total count from the last element of scan
    int total_count;    
    cudaCheckError(cudaMemcpy(&total_count, &scan_result[length - 1], sizeof(int), cudaMemcpyDeviceToHost));
    
    // Clean up
    cudaFree(flags);
    cudaFree(scan_result);
    
    return total_count; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
