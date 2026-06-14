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

// Caffe-style col2im (scatter-add): the reverse of im2col, for ConvTranspose.
// col is [channels*kH*kW, col_h*col_w]; accumulates into data_im [channels, img_h, img_w].
// img_* = output (Y) spatial; col_* = input (X) spatial. data_im must be pre-zeroed.
template <typename T>
__global__ void Col2imNCHWKernel(const int n, const T* data_col,
                                 const int img_h, const int img_w, const int kh, const int kw,
                                 const int pad_t, const int pad_l, const int stride_h, const int stride_w,
                                 const int dil_h, const int dil_w, const int col_h, const int col_w,
                                 T* data_im) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
       index += blockDim.x * gridDim.x) {
    T val = T(0);
    const int w_im = index % img_w + pad_l;
    const int h_im = (index / img_w) % img_h + pad_t;
    const int c = index / (img_w * img_h);
    const int kext_w = (kw - 1) * dil_w + 1;
    const int kext_h = (kh - 1) * dil_h + 1;
    const int w_col_start = (w_im < kext_w) ? 0 : (w_im - kext_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, col_w);
    const int h_col_start = (h_im < kext_h) ? 0 : (h_im - kext_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, col_h);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        int h_k = h_im - h_col * stride_h;
        int w_k = w_im - w_col * stride_w;
        if (h_k % dil_h == 0 && w_k % dil_w == 0) {
          h_k /= dil_h; w_k /= dil_w;
          const int col_idx = (((c * kh + h_k) * kw + w_k) * col_h + h_col) * col_w + w_col;
          val += data_col[col_idx];
        }
      }
    }
    data_im[index] = data_im[index] + val;
  }
}

template <typename T>
void Col2imNCHWLauncher(cudaStream_t stream, const T* data_col, int channels, int img_h, int img_w,
                        int col_h, int col_w, int kh, int kw, int pad_t, int pad_l, int stride_h,
                        int stride_w, int dil_h, int dil_w, T* data_im) {
  const int n = channels * img_h * img_w;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  Col2imNCHWKernel<T><<<blocks, threads, 0, stream>>>(
      n, data_col, img_h, img_w, kh, kw, pad_t, pad_l, stride_h, stride_w, dil_h, dil_w,
      col_h, col_w, data_im);
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
template void Col2imNCHWLauncher<float>(cudaStream_t, const float*, int, int, int, int, int, int,
                                        int, int, int, int, int, int, int, float*);
template void Col2imNCHWLauncher<half>(cudaStream_t, const half*, int, int, int, int, int, int,
                                       int, int, int, int, int, int, int, half*);
template void Col2imNCHWLauncher<double>(cudaStream_t, const double*, int, int, int, int, int, int,
                                         int, int, int, int, int, int, int, double*);

}  // namespace cuda
}  // namespace onnxruntime
