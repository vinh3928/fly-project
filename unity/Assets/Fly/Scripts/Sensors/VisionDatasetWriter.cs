using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Streams stereo eye frames and <see cref="EyeRayLabel"/>s to a dataset folder read by
    /// <c>python/flybrain/vision/unity_dataset.py</c>:
    /// <list type="bullet">
    /// <item><c>frames.u8</c>: per sample left then right eye, H×W×RGB uint8, top row first
    /// (the ML-Agents camera sensor layout the live mind receives).</item>
    /// <item><c>labels.csv</c>: one row per sample, header <see cref="LabelHeader"/>.</item>
    /// <item><c>meta.json</c>: sample count, frame size and ray columns, written on dispose.</item>
    /// </list>
    /// </summary>
    public sealed class VisionDatasetWriter : IDisposable
    {
        public const string FramesFile = "frames.u8";
        public const string LabelsFile = "labels.csv";
        public const string MetaFile = "meta.json";
        public const int Channels = 3;

        private readonly string _directory;
        private readonly int _width;
        private readonly int _height;
        private readonly FileStream _frames;
        private readonly StreamWriter _labels;
        private readonly byte[] _eyeBytes;

        public VisionDatasetWriter(string directory, int width, int height)
        {
            _directory = directory;
            _width = width;
            _height = height;
            Directory.CreateDirectory(directory);
            _frames = new FileStream(Path.Combine(directory, FramesFile), FileMode.Create, FileAccess.Write);
            _labels = new StreamWriter(Path.Combine(directory, LabelsFile), false, new UTF8Encoding(false));
            _labels.WriteLine(LabelHeader);
            _eyeBytes = new byte[width * height * Channels];
        }

        public int Count { get; private set; }

        public static string LabelHeader
        {
            get
            {
                var sb = new StringBuilder("region,wall_hazard,forward_bias,turn_bias,opening_bearing,opening_strength");
                for (int i = 0; i < EyeRayLabels.Columns; i++)
                {
                    sb.Append(",d").Append(i);
                }

                return sb.ToString();
            }
        }

        public void Write(Color32[] leftBottomFirst, Color32[] rightBottomFirst, VisionPoseRegion region, EyeRayLabel label)
        {
            WriteEye(leftBottomFirst);
            WriteEye(rightBottomFirst);

            var inv = CultureInfo.InvariantCulture;
            var row = new StringBuilder(region.ToString());
            foreach (float v in new[]
                     {
                         label.WallHazard, label.ForwardBias, label.TurnBias, label.OpeningBearing, label.OpeningStrength,
                     })
            {
                row.Append(',').Append(v.ToString("0.#####", inv));
            }

            foreach (float d in label.Distances)
            {
                row.Append(',').Append(d.ToString("0.#####", inv));
            }

            _labels.WriteLine(row.ToString());
            Count++;
        }

        /// <summary>Flip a bottom-first Color32 buffer into top-first RGB bytes.</summary>
        public static void ToTopFirstRgb(Color32[] bottomFirst, int width, int height, byte[] destination)
        {
            int o = 0;
            for (int y = height - 1; y >= 0; y--)
            {
                int row = y * width;
                for (int x = 0; x < width; x++)
                {
                    Color32 p = bottomFirst[row + x];
                    destination[o++] = p.r;
                    destination[o++] = p.g;
                    destination[o++] = p.b;
                }
            }
        }

        private void WriteEye(Color32[] bottomFirst)
        {
            if (bottomFirst == null || bottomFirst.Length != _width * _height)
            {
                throw new ArgumentException($"Expected {_width}x{_height} pixels.");
            }

            ToTopFirstRgb(bottomFirst, _width, _height, _eyeBytes);
            _frames.Write(_eyeBytes, 0, _eyeBytes.Length);
        }

        public void Dispose()
        {
            _frames.Dispose();
            _labels.Dispose();
            File.WriteAllText(
                Path.Combine(_directory, MetaFile),
                "{\n" +
                $"  \"count\": {Count},\n" +
                $"  \"width\": {_width},\n" +
                $"  \"height\": {_height},\n" +
                $"  \"channels\": {Channels},\n" +
                $"  \"columns\": {EyeRayLabels.Columns},\n" +
                $"  \"open_distance\": {EyeRayLabels.OpenDistance.ToString(CultureInfo.InvariantCulture)},\n" +
                $"  \"hazard_range\": {EyeRayLabels.HazardRange.ToString(CultureInfo.InvariantCulture)},\n" +
                $"  \"field_of_view\": {FlyEyeCamera.DefaultFieldOfView.ToString(CultureInfo.InvariantCulture)}\n" +
                "}\n");
        }
    }
}
