// cuDNN-free convolution building blocks for the CUDA EP (Jetson Nano gen1 fork).
// im2col + cuBLAS GEMM replaces cudnnConvolutionForward so the CUDA EP needs no cuDNN.
// Layout: NCHW. Conv1d is handled by the caller mapping the single spatial dim onto W (H=1).
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

// Caffe-style im2col. Produces a column matrix of shape [C*kH*kW, outH*outW] (row-major),
// row = (c*kH + i)*kW + j, col = oy*outW + ox. n_threads = C*outH*outW.
template <typename T>
__global__ void Im2colNCHWKernel(const int n, const T* data_im,
                                 const int height, const int width,
                                 const int kh, const int kw,
                                 const int pad_h, const int pad_w,
                                 const int stride_h, const int stride_w,
                                 const int dil_h, const int dil_w,
                                 const int out_h, const int out_w, T* data_col) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
       index += blockDim.x * gridDim.x) {
    const int w_out = index % out_w;
    int h_index = index / out_w;
    const int h_out = h_index % out_h;
    const int channel_in = h_index / out_h;
    const int channel_out = channel_in * kh * kw;
    const int h_in = h_out * stride_h - pad_h;
    const int w_in = w_out * stride_w - pad_w;
    T* col_ptr = data_col + (channel_out * out_h + h_out) * out_w + w_out;
    const T* im_ptr = data_im + (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kh; ++i) {
      for (int j = 0; j < kw; ++j) {
        const int h = h_in + i * dil_h;
        const int w = w_in + j * dil_w;
        *col_ptr = (h >= 0 && w >= 0 && h < height && w < width)
                       ? im_ptr[i * dil_h * width + j * dil_w]
                       : T(0);
        col_ptr += out_h * out_w;
      }
    }
  }
}

// Y[m, p] += bias[m]  (Y is [M, outH*outW] row-major, bias length M). n_threads = M*outHW.
template <typename T>
__global__ void AddBiasNCHWKernel(const int n, T* y, const T* bias, const int out_hw) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
       index += blockDim.x * gridDim.x) {
    y[index] = y[index] + bias[index / out_hw];
  }
}

template <typename T>
void Im2colNCHWLauncher(cudaStream_t stream, const T* data_im, int channels, int height,
                        int width, int kh, int kw, int pad_h, int pad_w, int stride_h,
                        int stride_w, int dil_h, int dil_w, int out_h, int out_w, T* data_col) {
  const int n = channels * out_h * out_w;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  Im2colNCHWKernel<T><<<blocks, threads, 0, stream>>>(
      n, data_im, height, width, kh, kw, pad_h, pad_w, stride_h, stride_w,
      dil_h, dil_w, out_h, out_w, data_col);
}

template <typename T>
void AddBiasNCHWLauncher(cudaStream_t stream, T* y, const T* bias, int m, int out_hw) {
  const int n = m * out_hw;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  AddBiasNCHWKernel<T><<<blocks, threads, 0, stream>>>(n, y, bias, out_hw);
}

// explicit instantiations
template void Im2colNCHWLauncher<float>(cudaStream_t, const float*, int, int, int, int, int,
                                        int, int, int, int, int, int, int, int, float*);
template void AddBiasNCHWLauncher<float>(cudaStream_t, float*, const float*, int, int);
template void Im2colNCHWLauncher<half>(cudaStream_t, const half*, int, int, int, int, int,
                                       int, int, int, int, int, int, int, int, half*);
template void AddBiasNCHWLauncher<half>(cudaStream_t, half*, const half*, int, int);
template void Im2colNCHWLauncher<double>(cudaStream_t, const double*, int, int, int, int, int,
                                         int, int, int, int, int, int, int, int, double*);
template void AddBiasNCHWLauncher<double>(cudaStream_t, double*, const double*, int, int);

}  // namespace cuda
}  // namespace onnxruntime
