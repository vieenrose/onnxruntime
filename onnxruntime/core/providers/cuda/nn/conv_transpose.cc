// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_transpose.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"  // cublasGemmHelper (cuDNN-free convtranspose)

namespace onnxruntime {
namespace cuda {

// cuDNN-free convtranspose launchers (defined in conv_nocudnn.cu) — Jetson Nano gen1 fork.
template <typename U>
void Col2imNCHWLauncher(cudaStream_t, const U*, int, int, int, int, int, int, int, int, int, int,
                        int, int, int, U*);
template <typename U>
void AddBiasNCHWLauncher(cudaStream_t, U*, const U*, int, int);

// Op Set 11 for ConvTranspose only update document to clarify default dilations and strides value.
// which are already covered by op set 11 cpu version, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      ConvTranspose,                                                                       \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T>);                                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      ConvTranspose,                                                                       \
      kOnnxDomain,                                                                         \
      11,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status ConvTranspose<T>::ComputeInternal(OpKernelContext* context) const {
#if defined(ORT_CUDA_NO_CUDNN_CONV)
  {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const bool has_bias = context->InputCount() >= 3;
    typename ConvTransposeAttributes::Prepare p;
    ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p, false));
    const int krank = static_cast<int>(p.kernel_shape.size());
    if (p.Y->Shape().Size() != 0 && (krank == 1 || krank == 2)) {
      const int64_t group = conv_transpose_attrs_.group;
      const int Cin_g = static_cast<int>(p.num_input_channels / group);
      const int Cout_g = static_cast<int>(p.num_output_channels / group);
      int H, Wd, kH, kW, padT, padL, strH, strW, dilH, dilW, outH, outW;
      if (krank == 2) {
        H = (int)p.input_shape[0]; Wd = (int)p.input_shape[1];
        kH = (int)p.kernel_shape[0]; kW = (int)p.kernel_shape[1];
        padT = (int)p.pads[0]; padL = (int)p.pads[1];
        strH = (int)p.strides[0]; strW = (int)p.strides[1];
        dilH = (int)p.dilations[0]; dilW = (int)p.dilations[1];
        outH = (int)p.Y->Shape()[2]; outW = (int)p.Y->Shape()[3];
      } else {  // ConvTranspose1d -> (H=1, W=L)
        H = 1; Wd = (int)p.input_shape[0];
        kH = 1; kW = (int)p.kernel_shape[0];
        padT = 0; padL = (int)p.pads[0];
        strH = 1; strW = (int)p.strides[0];
        dilH = 1; dilW = (int)p.dilations[0];
        outH = 1; outW = (int)p.Y->Shape()[2];
      }
      const int input_image_size = H * Wd;
      const int kernel_dim = Cout_g * kH * kW;
      auto col = GetScratchBuffer<CudaT>(static_cast<size_t>(kernel_dim) * input_image_size);
      cudaStream_t stream = Stream();
      cublasHandle_t cublas = CublasHandle();
      const CudaT* xdata = reinterpret_cast<const CudaT*>(p.X->template Data<T>());
      const CudaT* fdata = reinterpret_cast<const CudaT*>(p.F->template Data<T>());
      CudaT* ydata = reinterpret_cast<CudaT*>(p.Y->template MutableData<T>());
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(ydata, 0, p.Y->Shape().Size() * sizeof(CudaT), stream));
      const CudaT one = Consts<CudaT>::One;
      const CudaT zero = Consts<CudaT>::Zero;
      for (int64_t n = 0; n < p.N; ++n)
        for (int64_t g = 0; g < group; ++g) {
          const CudaT* xg = xdata + (n * p.num_input_channels + g * Cin_g) * (int64_t)input_image_size;
          const CudaT* fg = fdata + g * (int64_t)Cin_g * kernel_dim;
          CudaT* yg = ydata + (n * p.num_output_channels + g * Cout_g) * (int64_t)outH * outW;
          CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                                  input_image_size, kernel_dim, Cin_g, &one,
                                                  xg, input_image_size, fg, kernel_dim, &zero,
                                                  col.get(), input_image_size, GetDeviceProp()));
          Col2imNCHWLauncher<CudaT>(stream, col.get(), Cout_g, outH, outW, H, Wd, kH, kW,
                                    padT, padL, strH, strW, dilH, dilW, yg);
        }
      if (p.B != nullptr) {
        const CudaT* bdata = reinterpret_cast<const CudaT*>(p.B->template Data<T>());
        for (int64_t n = 0; n < p.N; ++n)
          AddBiasNCHWLauncher<CudaT>(stream, ydata + n * p.num_output_channels * (int64_t)outH * outW,
                                     bdata, (int)p.num_output_channels, outH * outW);
      }
      return Status::OK();
    }
  }
#endif
  return DoConvTranspose(context, false);
}

template <typename T>
Status ConvTranspose<T>::DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  auto x_dims = x_shape.AsShapeVector();
  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());

  auto x_dimensions = X->Shape().NumDimensions();
  if (x_dimensions < 3 || x_dimensions > 5) {
    // TODO: the error message should tell which operator raises it.
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input X must be 3-, 4- or 5-dimensional.",
                           " X: ", X->Shape().ToString().c_str());
  }
  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  TensorShapeVector w_dims = w_shape.AsShapeVector();
  auto w_data = reinterpret_cast<const CudaT*>(W->template Data<T>());

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  bool has_bias = dynamic_padding ? num_inputs == 4 : num_inputs == 3;

  CudaT* y_data = nullptr;
  if (x_dimensions == 3) {
    x_dims.insert(x_dims.begin() + 2, 1);
    w_dims.insert(w_dims.begin() + 2, 1);
  }

  {
    std::lock_guard<OrtMutex> lock(s_.mutex);
    // TODO: add a global cache if need to handle cases for multiple frames running simultaneously with different batch_size
    bool input_dims_changed = (s_.last_x_dims.AsShapeVector() != x_dims);
    bool w_dims_changed = (s_.last_w_dims.AsShapeVector() != w_dims);
    if (input_dims_changed || w_dims_changed) {
      if (input_dims_changed)
        s_.last_x_dims = gsl::make_span(x_dims);

      if (w_dims_changed) {
        s_.last_w_dims = gsl::make_span(w_dims);
        s_.cached_benchmark_results.clear();
      }

      ConvTransposeAttributes::Prepare p;
      ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p, dynamic_padding));

      auto y_dims = p.Y->Shape().AsShapeVector();
      if (x_dimensions == 3) {
        y_dims.insert(y_dims.begin() + 2, 1);
        p.kernel_shape.insert(p.kernel_shape.begin(), 1);
        p.pads.insert(p.pads.begin(), 0);
        p.pads.insert(p.pads.begin() + 2, 0);
        p.strides.insert(p.strides.begin(), 1);
        p.dilations.insert(p.dilations.begin(), 1);
      }
      s_.y_dims = gsl::make_span(y_dims);

      if (w_dims_changed)
        ORT_RETURN_IF_ERROR(s_.w_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

      // Special case when there is a dim value of 0 in the shape.
      // Return only after we have cached the following for subsequent runs :
      // 1) `w_dims` in the `w_desc`
      // 2) `y_dims` in s_.y_dims
      if (p.Y->Shape().Size() == 0) {
        return Status::OK();
      }

      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims, CudnnTensor::GetDataType<CudaT>()));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims, CudnnTensor::GetDataType<CudaT>()));

      cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
      ORT_RETURN_IF_ERROR(s_.conv_desc.Set(p.kernel_shape.size(), p.pads, p.strides, p.dilations,
                                           gsl::narrow_cast<int>(conv_transpose_attrs_.group),
                                           mode, CudnnTensor::GetDataType<CudaT>()));

      if (has_bias) {
        const auto& b_shape = p.B->Shape();
        ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
        TensorShapeVector b_dims(2 + p.kernel_shape.size());
        b_dims[0] = 1;           // N
        b_dims[1] = b_shape[0];  // C
        for (size_t i = 0; i < p.kernel_shape.size(); i++)
          b_dims[2 + i] = 1;

        ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      }

      y_data = reinterpret_cast<CudaT*>(p.Y->template MutableData<T>());

      if (!s_.cached_benchmark_results.contains(x_dims)) {
        IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);

        // set math type to tensor core before algorithm search
        if constexpr (std::is_same<T, MLFloat16>::value)
          CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));

        cudnnConvolutionBwdDataAlgoPerf_t perf;
        int algo_count = 1;
        CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
            CudnnHandle(),
            s_.w_desc,
            w_data,
            s_.x_tensor,
            x_data,
            s_.conv_desc,
            s_.y_tensor,
            y_data,
            1,
            &algo_count,
            &perf,
            algo_search_workspace.get(),
            AlgoSearchWorkspaceSize));
        s_.cached_benchmark_results.insert(x_dims, {perf.algo, perf.memory, perf.mathType});
      }

      const auto& perf = s_.cached_benchmark_results.at(x_dims);
      CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
      s_.algo = perf.algo;
      s_.workspace_bytes = perf.memory;
    }

    // The following block will be executed in case there has been no change in the shapes of the
    // input and the filter compared to the previous run
    if (!y_data) {
      auto y_dims = s_.y_dims.AsShapeVector();
      if (x_dimensions == 3) {
        y_dims.erase(y_dims.begin() + 2);
      }
      Tensor* Y = context->Output(0, TensorShape(y_dims));
      y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

      // Bail out early if one of the output dimensions is zero.
      if (Y->Shape().Size() == 0) {
        return Status::OK();
      }
    }

    const auto alpha = Consts<CudaT>::One;
    const auto beta = Consts<CudaT>::Zero;

    IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes);

    CUDNN_RETURN_IF_ERROR(
        cudnnConvolutionBackwardData(
            CudnnHandle(),
            &alpha,
            s_.w_desc,
            w_data,
            s_.x_tensor,
            x_data,
            s_.conv_desc,
            s_.algo,
            workspace.get(),
            s_.workspace_bytes,
            &beta,
            s_.y_tensor,
            y_data));

    if (has_bias) {
      const Tensor* B = dynamic_padding ? context->Input<Tensor>(3) : context->Input<Tensor>(2);
      auto b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
      CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, s_.b_tensor, b_data, &alpha, s_.y_tensor, y_data));
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
