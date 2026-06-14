# onnxruntime — cuDNN-free CUDA Execution Provider

> **This branch (`cudnn-free-cuda-jetson-nano-gen1`, onnxruntime v1.11.0)** is the **Jetson Nano gen1 / CUDA 10.2** build — the last ORT release that supports the Nano's GPU.

> Fork of [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) adding a **cuDNN-free,
> cuBLAS-only CUDA Execution Provider**. Build with **`-Donnxruntime_CUDA_NO_CUDNN=ON`**.

## Why
On memory-constrained NVIDIA devices — notably the **Jetson Nano gen1** (Tegra X1, CUDA 10.2) —
cuDNN's runtime footprint (~**782 MB**: `libcudnn_cnn_infer` 395 MB + `ops_infer` 121 MB + conv
workspaces) is prohibitive. The stock CUDA EP loads cuDNN unconditionally (even `cudnnCreate` at
init faults it in). This fork lets the CUDA EP run **without cuDNN at all**, keeping cuBLAS.

Measured: a cuBLAS GEMM leaves `libcudnn` resident at ~200 KB; one stock cuDNN conv faults in
~130 MB (cuDNN 9) / ~395–782 MB (Jetson cuDNN 8). The cuDNN-free EP avoids all of it.

## What `-Donnxruntime_CUDA_NO_CUDNN=ON` does
| cuDNN op family | cuDNN-free handling |
|---|---|
| **Conv** | im2col + `cublasGemmHelper` + bias. Handles stride/pad/dilation/**groups**/Conv1d and **auto_pad + asymmetric / TF-"SAME"** padding (per-dim `ComputePadAndOutputShape`), so depthwise/separable TF-exported convs work. |
| **ConvTranspose** | filter^T @ X (cuBLAS) + col2im. |
| **Softmax** | ORT already has custom kernels — no change. |
| **RNN / GRU / LSTM** | routed to the **CPU EP**. |
| **Pooling** (`cudnnPoolingForward`) | routed to the CPU EP. |
| **Reduce\*** (`cudnnReduceTensor`) | routed to the CPU EP. |
| **`cudnnCreate`** | skipped (cuBLAS handle kept) — cuDNN is never initialized/loaded. |

cuDNN is still *linked* but never *loaded* at runtime, so its pages don't fault into RSS.

## Branches
| Branch | onnxruntime | Runs on | Purpose |
|---|---|---|---|
| **`cudnn-free-cuda-ep`** | **v1.23.1** | modern NVIDIA GPUs, **CUDA 12 / 13** | Reference implementation (developed & validated on a CUDA-13 GPU). **Does NOT run on the Jetson Nano gen1** — ORT 1.23 requires CUDA 11+. |
| **`cudnn-free-cuda-jetson-nano-gen1`** | **v1.11.0** | **Jetson Nano gen1, CUDA 10.2** | The deployable Nano build. 1.11.0 is the last ORT release supporting CUDA 10.2. |

## Validation
Conv matches the CPU reference to **9.5e-7**, ConvTranspose to **4.8e-7**. On the 1.11.0 branch,
built+run cuDNN-free in an L4T r32.7 / CUDA-10.2 container (sm_53): SenseVoice, silero-VAD,
melo8k (opset-16), TEN-VAD, and the X-ASR streaming zipformer all run with **no cuDNN**.

## Notes for the Jetson Nano gen1
ORT 1.11 caps at **ai.onnx opset 16** — models exported at opset 17+ (e.g. melo8k's
`LayerNormalization`) must be decomposed to opset 16 first. cuDNN-free trades cuDNN's RAM for
running a few ops (Pool/Reduce/RNN) on the CPU EP — fine for the small layers in these speech models.

---

# (upstream onnxruntime README below)

<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

**ONNX Runtime is a cross-platform inference and training machine-learning accelerator**.

**ONNX Runtime inference** can enable faster customer experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc. ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-inferencing)

**ONNX Runtime training** can accelerate the model training time on multi-node NVIDIA GPUs for transformer models with a one-line addition for existing PyTorch training scripts. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-training)


## Get Started

**General Information**: [onnxruntime.ai](https://onnxruntime.ai)

**Usage documention and tutorials**: [onnxruntime.ai/docs](https://onnxruntime.ai/docs)

**Companion sample repositories**: 
- ONNX Runtime Inferencing: [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)
- ONNX Runtime Training: [microsoft/onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples)


## Build Pipeline Status
|System|CPU|GPU|EPs|
|---|---|---|---|
|Windows|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20TensorRT%20CI%20Pipeline?label=Windows+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=47)|
|Linux|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20Minimal%20Build%20E2E%20CI%20Pipeline?label=Linux+CPU+Minimal+Build)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=64)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20x64%20NoContribops%20CI%20Pipeline?label=Linux+CPU+x64+No+Contrib+Ops)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=110)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/centos7_cpu?label=Linux+CentOS7)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=78)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-ci-pipeline?label=Linux+CPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=86)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20TensorRT%20CI%20Pipeline?label=Linux+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=45)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-distributed?label=Distributed+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=140)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-gpu-ci-pipeline?label=Linux+GPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=84)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20NUPHAR%20CI%20Pipeline?label=Linux+NUPHAR)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=110)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20OpenVINO%20CI%20Pipeline%20v2?label=Linux+OpenVINO)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=108)|
|Mac|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20NoContribops%20CI%20Pipeline?label=MacOS+NoContribops)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=65)|||
|Android|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Android%20CI%20Pipeline?label=Android)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=53)|
|iOS|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/iOS%20CI%20Pipeline?label=iOS)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=134)|
|WebAssembly|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20WebAssembly%20CI%20Pipeline?label=WASM)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=161)|


## Data/Telemetry

Windows distributions of this project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

## Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

For general discussion or questions, please use [GitHub Discussions](https://github.com/microsoft/onnxruntime/discussions).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This project is licensed under the [MIT License](LICENSE).
