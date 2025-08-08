using System;
using System.IO;
using System.Linq;
using OpenCvSharp;
using SamOnnx.Runtime;

namespace SamOnnxDemo
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // --- Read assets from the output folder (bin\Debug\net8.0\ etc.) ---
                string baseDir = AppContext.BaseDirectory;
                string encPath = Path.Combine(baseDir, "sam_encoder.onnx");
                string decPath = Path.Combine(baseDir, "sam_decoder.onnx");
                string imgPath = Path.Combine(baseDir, "ImageSample.jpg");

                Console.WriteLine($"[INFO] BaseDir: {baseDir}");
                Console.WriteLine($"[INFO] Exists? enc:{File.Exists(encPath)} dec:{File.Exists(decPath)} img:{File.Exists(imgPath)}");

                if (!File.Exists(encPath) || !File.Exists(decPath) || !File.Exists(imgPath))
                {
                    Console.Error.WriteLine("[ERROR] Missing files in output folder. Ensure your .csproj has CopyToOutputDirectory=Always for the 3 assets.");
                    return;
                }

                using var sam = new SamOnnxRuntime
                {
                    VerboseLogging = true,
                    ApplyVerticalMaskCorrection = false
                };

                Console.WriteLine("[INFO] Loading ONNX models…");
                if (!sam.Initialize(encPath, decPath))
                {
                    Console.Error.WriteLine("[ERROR] Failed to initialize SAM models.");
                    return;
                }
                Console.WriteLine("[INFO] SAM ONNX models loaded.");

                using var image = Cv2.ImRead(imgPath, ImreadModes.Color);
                if (image.Empty())
                {
                    Console.Error.WriteLine("[ERROR] Failed to read image at: " + imgPath);
                    return;
                }

                // Precompute image embedding once
                sam.SetImage(image);
                Console.WriteLine("[INFO] Image set and embedded.");

                // -------- BOX PROMPT --------
                var box = new OpenCvSharp.Rect(image.Width / 8, image.Height / 8, image.Width / 2, image.Height / 2);
                Console.WriteLine($"[INFO] Box: x={box.X}, y={box.Y}, w={box.Width}, h={box.Height}");

                var (boxMasks, boxScores) = sam.PredictWithBox(box, multimaskOutput: true);
                using var boxOverlay = OverlayBestMask(image, boxMasks, boxScores, new Scalar(0, 255, 0), 0.5);
                string boxOut = Path.Combine(baseDir, "result_box.jpg");
                Cv2.ImWrite(boxOut, boxOverlay);
                Console.WriteLine($"[OK] Saved {boxOut} (best score: {boxScores.Max():0.000})");

                // -------- POINT PROMPT (center positive) --------
                var pt = new Point(image.Width / 2, image.Height / 2);
                var (ptMasks, ptScores) = sam.PredictWithPoints(
                    new[] { pt },
                    new[] { 1 }, // 1=positive, 0=negative
                    multimaskOutput: true
                );
                using var ptOverlay = OverlayBestMask(image, ptMasks, ptScores, new Scalar(255, 0, 0), 0.5);
                string ptOut = Path.Combine(baseDir, "result_point.jpg");
                Cv2.ImWrite(ptOut, ptOverlay);
                Console.WriteLine($"[OK] Saved {ptOut} (best score: {ptScores.Max():0.000})");

                Console.WriteLine("[DONE] Inference complete.");
            }
            catch (DllNotFoundException e)
            {
                Console.Error.WriteLine("[ERROR] Native dependency missing. If you see 'OpenCvSharpExtern', ensure:");
                Console.Error.WriteLine("  - OpenCvSharp4.runtime.win is installed");
                Console.Error.WriteLine("  - bin\\...\\runtimes\\win-x64\\native has OpenCvSharpExtern.dll and opencv_world*.dll");
                Console.Error.WriteLine("  - For a quick workaround: copy those DLLs next to the exe.");
                Console.Error.WriteLine(e.ToString());
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("[UNHANDLED] " + ex);
            }
        }

        /// <summary>
        /// Picks highest-score mask row, reshapes to decoder grid (assumed square),
        /// resizes to image size, and blends color overlay.
        /// </summary>
        private static Mat OverlayBestMask(Mat image, Mat flatMasks, float[] scores, Scalar color, double alpha)
        {
            if (flatMasks.Empty() || scores == null || scores.Length == 0)
                return image.Clone();

            int best = Array.IndexOf(scores, scores.Max());

            // Each row is a flattened H*W mask from decoder (typically square like 256x256)
            int flatLen = flatMasks.Cols;
            int side = (int)Math.Round(Math.Sqrt(flatLen));
            if (side * side != flatLen)
                throw new InvalidOperationException($"Mask row length {flatLen} is not a perfect square—cannot infer HxW.");

            using var small = flatMasks.Row(best).Reshape(1, side); // side x side
            using var mask8 = new Mat();
            small.ConvertTo(mask8, MatType.CV_8UC1); // wrapper already thresholds to 0/255

            using var mask = new Mat();
            Cv2.Resize(mask8, mask, new Size(image.Width, image.Height), 0, 0, InterpolationFlags.Nearest);

            var over = image.Clone();
            over.SetTo(color, mask); // paint color where mask>0

            var blended = new Mat();
            Cv2.AddWeighted(over, alpha, image, 1 - alpha, 0, blended);
            return blended;
        }
    }
}
