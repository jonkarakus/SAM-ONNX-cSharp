# SAM-ONNX-CSharp

This is a C# wrapper around Meta’s Segment Anything Model (SAM) using ONNX Runtime.
I originally created this to support a larger project and figured it might be helpful for others doing segmentation in C#. 

This repository also includes an **ONNX conversion notebook** that starts from a single Meta SAM checkpoint and exports **two** ONNX files: an **image encoder** and a **mask decoder**.
The repository then lets you load SAM’s encoder and decoder ONNX models, preprocess images with OpenCV, and generate segmentation masks from **box** or **point** prompts entirely in .NET (no Python runtime required).
----------------------------------------------------------------------------------------------------------

## Features

- C# wrapper that hides ONNX Runtime details and exposes a simple API for SAM.
- OpenCvSharp preprocessing (resize, pad, normalize to ImageNet statistics).
- Prompt types:
  - **Bounding box prompts**
  - **Foreground/background point prompts**
  - Optional interactive point selection (UI click-to-add points)
- Outputs:
  - Binary segmentation masks
  - IoU-style confidence scores
- Minimal console example to get started quickly.

----------------------------------------------------------------------------------------------------------

## Repository Layout

- `SamPredictor.cs` — Core wrapper (`SamOnnxRuntime`) that runs the SAM encoder and decoder ONNX models.
- `ExampleUsage.cs` — Console example showing box/point prompting and saving result images.
- `convert_SAM_to_ONNX.ipynb` — **Python notebook** used to export/split Meta’s SAM into **encoder** and **decoder** ONNX files.
- `ImageSample.jpg` — Sample image for testing.

----------------------------------------------------------------------------------------------------------

## Where to Get Meta’s SAM Models

- Official repository: https://github.com/facebookresearch/segment-anything  
- Official checkpoints (choose one to start):
  - ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  - ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  - ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

----------------------------------------------------------------------------------------------------------

## Requirements

- .NET 6.0+
- NuGet:
  - `Microsoft.ML.OnnxRuntime`
  - `OpenCvSharp4` (and the appropriate runtime, e.g., `OpenCvSharp4.runtime.win`)
- ONNX models produced from Meta’s SAM:
  - `sam_encoder.onnx`
  - `sam_decoder.onnx`

----------------------------------------------------------------------------------------------------------

## Getting Started (Short)

1. **Download a SAM checkpoint** from Meta (links above).  
2. **Convert to ONNX**: Use `convert_SAM_to_ONNX.ipynb` to export **`sam_encoder.onnx`** and **`sam_decoder.onnx`** from the checkpoint.  
3. **Add dependencies** (`Microsoft.ML.OnnxRuntime`, `OpenCvSharp4`) and run the included example; it will save images like `result_box_prompt.jpg` and `result_point_prompt.jpg`.

----------------------------------------------------------------------------------------------------------

## Model Conversion (Meta SAM → ONNX Encoder/Decoder)

This project expects two ONNX files: `sam_encoder.onnx` and `sam_decoder.onnx`.  
The included notebook `convert_SAM_to_ONNX.ipynb` was used to:

1. **Load** one of Meta’s official SAM checkpoints (ViT-B/L/H).  
2. **Export** the image encoder to ONNX (`sam_encoder.onnx`).  
3. **Export** the mask decoder to ONNX (`sam_decoder.onnx`).  

Notes:

- The wrapper assumes standard I/O names (e.g., encoder input `images`; decoder inputs `image_embeddings`, `point_coords`, `point_labels`, `mask_input`, `has_mask_input`, `orig_im_size`). If your export uses different names, update them in `SamPredictor.cs`.
- Choose the architecture variant (B/L/H) that matches your checkpoint in the notebook before exporting.

----------------------------------------------------------------------------------------------------------

## Notes and Tips

- `ApplyVerticalMaskCorrection` can address small vertical offsets in some camera pipelines.
- For best performance, build in Release and keep ONNX Runtime graph optimizations enabled (already set in the wrapper).

----------------------------------------------------------------------------------------------------------

## Author

Created and maintained by **Jon Karakus**.

----------------------------------------------------------------------------------------------------------

## License

The C# wrapper code in this repository is provided for research and development use, subject to MIT license.  
The underlying SAM models are provided by Meta and are subject to their license (Apache 2.0). When using the models, review and comply with Meta’s original license and terms.
