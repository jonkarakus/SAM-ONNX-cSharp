using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace SamOnnx.Runtime
{   
    /// Segment Anything Model (SAM) inference using ONNX Runtime.
    /// Provides methods for image segmentation using box and point prompts.   
    public class SamOnnxRuntime : IDisposable
    {
        private InferenceSession imageEncoderSession = default!;
        private InferenceSession maskDecoderSession = default!;
        private bool initialized = false;
        private float[] imageEmbedding = Array.Empty<float>();
        private int[] embeddingSize = new int[4];
        private Size originalImageSize;
        
        // Image preprocessing information
        private int paddingTop = 0;
        private int paddingLeft = 0;
        private float imageScale = 1.0f;
        
        // Configuration
        private const int InputSize = 1024;  // SAM's expected input size
           
        /// Enable verbose logging for debugging
        public bool VerboseLogging { get; set; } = false;
           
        /// Apply vertical mask correction (useful for some camera setups)
        public bool ApplyVerticalMaskCorrection { get; set; } = false;

        /// Check if the model has been initialized
        public bool IsInitialized => initialized;

        /// Check if an image has been set for inference
        public bool IsImageSet => imageEmbedding != null && imageEmbedding.Length > 0;
         
        /// Initialize the SAM ONNX models
         
        /// <param name="encoderPath">Path to the encoder ONNX model</param>
        /// <param name="decoderPath">Path to the decoder ONNX model</param>
        /// <returns>True if initialization successful</returns>
        public bool Initialize(string encoderPath, string decoderPath)
        {
            try
            {
                if (!File.Exists(encoderPath))
                    throw new FileNotFoundException($"Encoder model not found: {encoderPath}");
                
                if (!File.Exists(decoderPath))
                    throw new FileNotFoundException($"Decoder model not found: {decoderPath}");
                
                if (VerboseLogging)
                {
                    Console.WriteLine($"Loading encoder: {encoderPath}");
                    Console.WriteLine($"Loading decoder: {decoderPath}");
                }
                
                var sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                
                imageEncoderSession = new InferenceSession(encoderPath, sessionOptions);
                maskDecoderSession = new InferenceSession(decoderPath, sessionOptions);
                
                initialized = true;
                
                if (VerboseLogging)
                    Console.WriteLine("SAM ONNX models loaded successfully");
                
                return true;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error initializing SAM models: {ex.Message}");
                return false;
            }
        }

        /// Set the image for segmentation. Must be called before prediction.
        /// <param name="image">Input image as OpenCV Mat</param>
        public void SetImage(Mat image)
        {
            if (!initialized)
                throw new InvalidOperationException("Models not initialized. Call Initialize() first.");
            
            originalImageSize = image.Size();
            
            // Preprocess and get embedding
            var inputTensor = PreprocessImage(image);
            var imageEmbedding = ComputeImageEmbedding(inputTensor);
            
            this.imageEmbedding = imageEmbedding;
            
            if (VerboseLogging)
            {
                Console.WriteLine($"Image set: {originalImageSize.Width}x{originalImageSize.Height}");
                Console.WriteLine($"Embedding shape: [{string.Join(",", embeddingSize)}]");
            }
        }
 
        /// Generate segmentation masks using a bounding box prompt
        /// <param name="box">Bounding box in format (x, y, width, height)</param>
        /// <param name="multimaskOutput">Return multiple masks if true</param>
        /// <returns>Tuple of (masks matrix, confidence scores)</returns>
        public (Mat Masks, float[] Scores) PredictWithBox(Rect box, bool multimaskOutput = true)
        {
            ValidateImageSet();
            
            // Convert box to corner points with special labels
            float x1 = box.X * imageScale + paddingLeft;
            float y1 = box.Y * imageScale + paddingTop;
            float x2 = (box.X + box.Width) * imageScale + paddingLeft;
            float y2 = (box.Y + box.Height) * imageScale + paddingTop;
            
            float[] pointCoords = new float[] { x1, y1, x2, y2 };
            float[] pointLabels = new float[] { 2.0f, 3.0f };  // Special labels for box corners
            
            return RunDecoder(pointCoords, pointLabels, 2);
        }

        /// Generate segmentation masks using point prompts
        /// <param name="points">Array of points</param>
        /// <param name="labels">Array of labels (1=foreground, 0=background)</param>
        /// <param name="multimaskOutput">Return multiple masks if true</param>
        /// <returns>Tuple of (masks matrix, confidence scores)</returns>
        public (Mat Masks, float[] Scores) PredictWithPoints(
            Point[] points, 
            int[] labels, 
            bool multimaskOutput = true)
        {
            ValidateImageSet();
            
            if (points.Length != labels.Length)
                throw new ArgumentException("Number of points and labels must match");
            
            // Convert points to model coordinates
            float[] pointCoords = new float[points.Length * 2];
            float[] pointLabels = new float[points.Length];
            
            for (int i = 0; i < points.Length; i++)
            {
                pointCoords[i * 2] = points[i].X * imageScale + paddingLeft;
                pointCoords[i * 2 + 1] = points[i].Y * imageScale + paddingTop;
                pointLabels[i] = labels[i];
            }
            
            return RunDecoder(pointCoords, pointLabels, points.Length);
        }

        #region Private Methods

        private void ValidateImageSet()
        {
            if (imageEmbedding == null || imageEmbedding.Length == 0)
                throw new InvalidOperationException("No image set. Call SetImage() first.");
        }

        private float[] PreprocessImage(Mat image)
        {
            // Reset padding values
            paddingTop = 0;
            paddingLeft = 0;
            
            // Resize image maintaining aspect ratio
            Mat resizedImage = ResizeAndPad(image, InputSize, out imageScale, out paddingLeft, out paddingTop);
            
            if (VerboseLogging)
            {
                Console.WriteLine($"Preprocessing: scale={imageScale:F3}, padLeft={paddingLeft}, padTop={paddingTop}");
            }
            
            // Convert to RGB float32 and normalize
            Mat normalizedImage = NormalizeImage(resizedImage);
            
            // Convert to tensor format (NCHW)
            return ImageToTensor(normalizedImage, InputSize);
        }

        private Mat ResizeAndPad(Mat image, int targetSize, out float scale, out int padLeft, out int padTop)
        {
            Mat resized = new Mat();
            
            if (image.Cols > image.Rows)
            {
                scale = (float)targetSize / image.Cols;
                int newHeight = (int)(image.Rows * scale);
                Cv2.Resize(image, resized, new Size(targetSize, newHeight));
                
                padTop = (targetSize - newHeight) / 2;
                padLeft = 0;
                int bottomPad = targetSize - newHeight - padTop;
                
                Cv2.CopyMakeBorder(resized, resized, padTop, bottomPad, 0, 0,
                                  BorderTypes.Constant, new Scalar(0, 0, 0));
            }
            else
            {
                scale = (float)targetSize / image.Rows;
                int newWidth = (int)(image.Cols * scale);
                Cv2.Resize(image, resized, new Size(newWidth, targetSize));
                
                padLeft = (targetSize - newWidth) / 2;
                padTop = 0;
                int rightPad = targetSize - newWidth - padLeft;
                
                Cv2.CopyMakeBorder(resized, resized, 0, 0, padLeft, rightPad,
                                  BorderTypes.Constant, new Scalar(0, 0, 0));
            }
            
            return resized;
        }

        private Mat NormalizeImage(Mat image)
        {
            // Convert to RGB float32 [0,1]
            Mat floatImage = new Mat();
            if (image.Channels() == 3)
            {
                Cv2.CvtColor(image, image, ColorConversionCodes.BGR2RGB);
            }
            image.ConvertTo(floatImage, MatType.CV_32FC3, 1.0 / 255.0);
            
            // Normalize with ImageNet mean and std
            Mat[] channels = Cv2.Split(floatImage);
            
            Cv2.Subtract(channels[0], new Scalar(0.485), channels[0]);
            Cv2.Divide(channels[0], new Scalar(0.229), channels[0]);
            
            Cv2.Subtract(channels[1], new Scalar(0.456), channels[1]);
            Cv2.Divide(channels[1], new Scalar(0.224), channels[1]);
            
            Cv2.Subtract(channels[2], new Scalar(0.406), channels[2]);
            Cv2.Divide(channels[2], new Scalar(0.225), channels[2]);
            
            Cv2.Merge(channels, floatImage);
            
            return floatImage;
        }

        private float[] ImageToTensor(Mat image, int size)
        {
            float[] tensor = new float[1 * 3 * size * size];
            
            // Convert from HWC to CHW format
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < size; h++)
                {
                    for (int w = 0; w < size; w++)
                    {
                        Vec3f pixel = image.At<Vec3f>(h, w);
                        tensor[c * size * size + h * size + w] = pixel[c];
                    }
                }
            }
            
            return tensor;
        }

        private float[] ComputeImageEmbedding(float[] inputTensor)
        {
            var inputMeta = new List<NamedOnnxValue>();
            var tensorShape = new int[] { 1, 3, InputSize, InputSize };
            var tensor = new DenseTensor<float>(inputTensor, tensorShape);
            inputMeta.Add(NamedOnnxValue.CreateFromTensor("images", tensor));
            
            // Run encoder
            var outputMeta = imageEncoderSession.Run(inputMeta);
            var output = outputMeta.First().AsTensor<float>();
            
            // Store embedding dimensions
            for (int i = 0; i < 4; i++)
            {
                embeddingSize[i] = (int)output.Dimensions[i];
            }
            
            return output.ToArray();
        }

        private (Mat Masks, float[] Scores) RunDecoder(
            float[] pointCoords, 
            float[] pointLabels, 
            int numPoints)
        {
            // Prepare decoder inputs
            int maskInputH = embeddingSize[2] * 4;
            int maskInputW = embeddingSize[3] * 4;
            
            var inputs = new List<NamedOnnxValue>();
            
            // Image embeddings
            inputs.Add(NamedOnnxValue.CreateFromTensor("image_embeddings",
                new DenseTensor<float>(imageEmbedding, embeddingSize)));
            
            // Point coordinates
            inputs.Add(NamedOnnxValue.CreateFromTensor("point_coords",
                new DenseTensor<float>(pointCoords, new int[] { 1, numPoints, 2 })));
            
            // Point labels
            inputs.Add(NamedOnnxValue.CreateFromTensor("point_labels",
                new DenseTensor<float>(pointLabels, new int[] { 1, numPoints })));
            
            // Empty mask input
            inputs.Add(NamedOnnxValue.CreateFromTensor("mask_input",
                new DenseTensor<float>(new float[1 * 1 * maskInputH * maskInputW],
                    new int[] { 1, 1, maskInputH, maskInputW })));
            
            // Has mask input (false)
            inputs.Add(NamedOnnxValue.CreateFromTensor("has_mask_input",
                new DenseTensor<float>(new float[] { 0.0f }, new int[] { 1 })));
            
            // Original image size
            inputs.Add(NamedOnnxValue.CreateFromTensor("orig_im_size",
                new DenseTensor<float>(
                    new float[] { originalImageSize.Height, originalImageSize.Width },
                    new int[] { 2 })));
            
            // Run decoder
            var outputs = maskDecoderSession.Run(inputs);
            
            // Get masks and scores
            var masksOutput = outputs.First(x => x.Name == "masks").AsTensor<float>();
            var scoresOutput = outputs.First(x => x.Name == "iou_predictions").AsTensor<float>();
            
            // Process masks
            var dimsSpan = masksOutput.Dimensions;          // ReadOnlySpan<int>
            int[] masksShape = new int[dimsSpan.Length];
            for (int i = 0; i < dimsSpan.Length; i++)
                masksShape[i] = dimsSpan[i];
            var masks = PostProcessMasks(masksOutput.ToArray(), masksShape);
            var scores = scoresOutput.ToArray();
            
            return (masks, scores);
        }

        private Mat PostProcessMasks(float[] masksData, int[] shape)
        {
            int numMasks = shape[1];
            int height = shape[2];
            int width = shape[3];
            
            // Apply vertical correction if enabled
            int yOffset = ApplyVerticalMaskCorrection ? (int)(paddingTop / imageScale) : 0;
            
            // Create output masks
            Mat allMasks = new Mat(numMasks, height * width, MatType.CV_8UC1);
            
            for (int m = 0; m < numMasks; m++)
            {
                Mat tempMask = new Mat(height, width, MatType.CV_32FC1);
                
                // Copy mask data
                long maskOffset = m * height * width;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        float value = masksData[maskOffset + y * width + x];
                        
                        // Apply vertical offset if needed
                        int targetY = ApplyVerticalMaskCorrection ? y - yOffset : y;
                        
                        if (targetY >= 0 && targetY < height)
                        {
                            tempMask.At<float>(targetY, x) = value;
                        }
                    }
                }
                
                // Threshold to binary
                Mat binaryMask = new Mat();
                Cv2.Threshold(tempMask, binaryMask, 0.0, 255.0, ThresholdTypes.Binary);
                binaryMask.ConvertTo(binaryMask, MatType.CV_8UC1);
                
                // Store flattened mask
                Mat flatMask = binaryMask.Reshape(1, 1);
                flatMask.CopyTo(allMasks.Row(m));
            }
            
            return allMasks;
        }

        #endregion

        #region IDisposable

        private bool disposed = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    imageEncoderSession?.Dispose();
                    maskDecoderSession?.Dispose();
                }
                disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        #endregion
    }
}