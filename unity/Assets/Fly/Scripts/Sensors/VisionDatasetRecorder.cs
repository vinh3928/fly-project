using System;
using FlySimulation.Environment;
using FlySimulation.Fly.Body;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Poses the fly at sampled spots (inside the box, by walls, facing the exit, in the tunnel,
    /// outside), renders both eyes and writes them with ray-cast labels. Restores the fly's pose.
    /// </summary>
    public static class VisionDatasetRecorder
    {
        public const int DefaultSamples = 6000;
        public const string DefaultFolder = "data/vision_unity";
        private const int MaxAttemptsPerSample = 20;

        /// <returns>Samples written.</returns>
        public static int Record(
            FlyBodyController body,
            ChamberController chamber,
            FlyEyeCamera left,
            FlyEyeCamera right,
            string directory,
            int samples,
            int seed,
            Func<int, int, bool> progress = null)
        {
            Transform root = body.transform;
            Vector3 savedPosition = root.position;
            Quaternion savedRotation = root.rotation;
            float rootY = savedPosition.y;
            var sampler = new VisionPoseSampler(chamber.Configuration, seed);

            int written = 0;
            try
            {
                using var writer = new VisionDatasetWriter(directory, left.Width, left.Height);
                for (int attempt = 0; written < samples && attempt < samples * MaxAttemptsPerSample; attempt++)
                {
                    VisionPose pose = sampler.Next();
                    Vector3 local = new Vector3(pose.PositionXZ.x, 0f, pose.PositionXZ.y);
                    Vector3 position = chamber.transform.TransformPoint(local);
                    position.y = rootY;
                    Quaternion rotation = chamber.transform.rotation * Quaternion.Euler(0f, pose.YawDegrees, 0f);
                    if (BodyClearance.OverlapsWall(root, position, rotation))
                    {
                        continue;
                    }

                    root.SetPositionAndRotation(position, rotation);
                    Physics.SyncTransforms();
                    Vector3 eyes = 0.5f * (left.Position + right.Position);
                    EyeRayLabel label = EyeRayLabels.Compute(eyes, root.forward);
                    writer.Write(left.CaptureRawPixels(), right.CaptureRawPixels(), pose.Region, label);
                    written++;
                    if (progress != null && progress(written, samples))
                    {
                        break;
                    }
                }
            }
            finally
            {
                root.SetPositionAndRotation(savedPosition, savedRotation);
                Physics.SyncTransforms();
            }

            return written;
        }
    }
}
