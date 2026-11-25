using System.IO;
using FlySimulation.Environment;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Sensors;
using UnityEditor;
using UnityEngine;

namespace FlySimulation.EditorTools
{
    /// <summary>Tools → Fly → Record Vision Dataset: eye frames + ray labels for the Python vision head.</summary>
    public static class VisionDatasetMenu
    {
        [MenuItem("Tools/Fly/Record Vision Dataset", priority = 10)]
        [MenuItem("Fly/Record Vision Dataset", priority = 10)]
        public static void RecordVisionDataset()
        {
            if (EditorApplication.isPlaying)
            {
                EditorUtility.DisplayDialog("Record Vision Dataset", "Exit Play mode first.", "OK");
                return;
            }

            var body = Object.FindAnyObjectByType<FlyBodyController>();
            var chamber = Object.FindAnyObjectByType<ChamberController>();
            if (body == null || chamber == null || chamber.Configuration == null)
            {
                EditorUtility.DisplayDialog(
                    "Record Vision Dataset", "Open the Chamber scene (Tools → Fly → Setup Chamber Scene).", "OK");
                return;
            }

            FlyEyeCamera left = null;
            FlyEyeCamera right = null;
            foreach (var eye in body.GetComponentsInChildren<FlyEyeCamera>(true))
            {
                if (eye.IsLeftEye)
                {
                    left = left != null ? left : eye;
                }
                else
                {
                    right = right != null ? right : eye;
                }
            }

            if (left == null || right == null)
            {
                EditorUtility.DisplayDialog("Record Vision Dataset", "Left/right FlyEyeCamera not found.", "OK");
                return;
            }

            string directory = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", VisionDatasetRecorder.DefaultFolder));
            int written;
            try
            {
                written = VisionDatasetRecorder.Record(
                    body,
                    chamber,
                    left,
                    right,
                    directory,
                    VisionDatasetRecorder.DefaultSamples,
                    chamber.Configuration.RandomSeed,
                    (done, total) => EditorUtility.DisplayCancelableProgressBar(
                        "Record Vision Dataset", $"{done}/{total} samples", done / (float)total));
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }

            Debug.Log($"Vision dataset: {written} samples → {directory}");
            EditorUtility.DisplayDialog(
                "Record Vision Dataset",
                $"Wrote {written} samples to\n{directory}\n\nTrain with:\ncd python && .venv/bin/python -m scripts.train_vision_head --unity",
                "OK");
        }
    }
}
