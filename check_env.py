from tensorflow.python.client import device_lib
device_lib.list_local_devices()

'''
errors
2020-10-19 20:48:15.823026: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-10-19 20:48:15.823259: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-10-19 20:48:20.565070: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-10-19 20:48:20.572085: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1a8a68b30c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-10-19 20:48:20.572328: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-10-19 20:48:20.573827: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-10-19 20:48:20.598183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1060 6GB computeCapability: 6.1
coreClock: 1.8475GHz coreCount: 10 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 178.99GiB/s
2020-10-19 20:48:20.599981: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-10-19 20:48:20.601525: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
2020-10-19 20:48:20.606665: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_10.dll
2020-10-19 20:48:20.625643: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library curand64_10.dll
2020-10-19 20:48:20.627298: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
2020-10-19 20:48:20.628885: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
2020-10-19 20:48:20.630419: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudnn64_7.dll'; dlerror: cudnn64_7.dll not found
2020-10-19 20:48:20.630633: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-10-19 20:48:20.692980: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-10-19 20:48:20.693169: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
2020-10-19 20:48:20.693280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
2020-10-19 20:48:20.695301: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1a8a7048600 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-10-19 20:48:20.695535: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1060 6GB, Compute Capability 6.1
'''