using FlySimulation.Environment;
using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Keeps the fly's solid spheres (the root's non-trigger <see cref="SphereCollider"/>s) out of
    /// chamber walls. Locomotion teleports the root with MovePosition and zeroes velocity, so the
    /// physics solver never pushes the head back out; moves are clamped here before they happen.
    /// </summary>
    public static class BodyClearance
    {
        /// <summary>Contact gap in meters: spheres stop this close to a wall.</summary>
        public const float Skin = 0.0005f;

        /// <summary>Walls, ceiling and tunnel walls.</summary>
        public static int WallMask => 1 << ChamberWallHider.WallLayer;

        /// <summary>
        /// Clamp <paramref name="delta"/> so no solid sphere of <paramref name="root"/>, posed at
        /// <paramref name="position"/>/<paramref name="rotation"/>, ends up inside a wall.
        /// </summary>
        public static Vector3 ClampMove(Transform root, Vector3 position, Quaternion rotation, Vector3 delta)
        {
            float distance = delta.magnitude;
            if (root == null || distance < 1e-9f)
            {
                return delta;
            }

            Vector3 direction = delta / distance;
            float allowed = distance;
            foreach (var sphere in root.GetComponents<SphereCollider>())
            {
                if (!IsSolid(sphere))
                {
                    continue;
                }

                Vector3 center = WorldCenter(sphere, position, rotation);
                float radius = CastRadius(sphere);
                if (TryOverlapPushDirection(center, radius, out Vector3 away))
                {
                    // Already touching: only moves that back away from the wall are allowed.
                    if (Vector3.Dot(direction, away) <= 0f)
                    {
                        return Vector3.zero;
                    }

                    continue;
                }

                if (Physics.SphereCast(
                        center,
                        radius,
                        direction,
                        out RaycastHit hit,
                        distance + Skin,
                        WallMask,
                        QueryTriggerInteraction.Ignore))
                {
                    allowed = Mathf.Min(allowed, Mathf.Max(0f, hit.distance - Skin));
                }
            }

            return direction * allowed;
        }

        /// <summary>
        /// True if turning to <paramref name="nextRotation"/> at <paramref name="position"/> keeps
        /// every solid sphere out of the walls (or the body is already touching one and must be
        /// free to turn away).
        /// </summary>
        public static bool CanRotate(Transform root, Vector3 position, Quaternion rotation, Quaternion nextRotation)
        {
            if (root == null)
            {
                return true;
            }

            foreach (var sphere in root.GetComponents<SphereCollider>())
            {
                if (!IsSolid(sphere))
                {
                    continue;
                }

                float radius = CastRadius(sphere);
                if (Overlaps(WorldCenter(sphere, position, rotation), radius))
                {
                    continue;
                }

                if (Overlaps(WorldCenter(sphere, position, nextRotation), radius))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>True if any solid sphere of <paramref name="root"/>, posed as given, is inside a wall.</summary>
        public static bool OverlapsWall(Transform root, Vector3 position, Quaternion rotation)
        {
            if (root == null)
            {
                return false;
            }

            foreach (var sphere in root.GetComponents<SphereCollider>())
            {
                if (IsSolid(sphere) && Overlaps(WorldCenter(sphere, position, rotation), CastRadius(sphere)))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool IsSolid(SphereCollider sphere)
        {
            return sphere != null && sphere.enabled && !sphere.isTrigger;
        }

        private static Vector3 WorldCenter(SphereCollider sphere, Vector3 position, Quaternion rotation)
        {
            return position + rotation * Vector3.Scale(sphere.center, sphere.transform.lossyScale);
        }

        /// <summary>Shrunk by <see cref="Skin"/> so a sphere resting on the floor does not graze it.</summary>
        private static float CastRadius(SphereCollider sphere)
        {
            Vector3 s = sphere.transform.lossyScale;
            float radius = sphere.radius * Mathf.Max(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
            return Mathf.Max(radius - Skin, radius * 0.5f);
        }

        private static bool Overlaps(Vector3 center, float radius)
        {
            return Physics.CheckSphere(center, radius, WallMask, QueryTriggerInteraction.Ignore);
        }

        private static bool TryOverlapPushDirection(Vector3 center, float radius, out Vector3 away)
        {
            away = Vector3.zero;
            var hits = Physics.OverlapSphere(center, radius, WallMask, QueryTriggerInteraction.Ignore);
            if (hits.Length == 0)
            {
                return false;
            }

            foreach (var wall in hits)
            {
                Vector3 offset = center - wall.ClosestPoint(center);
                away += offset.sqrMagnitude > 1e-12f ? offset.normalized : Vector3.zero;
            }

            if (away.sqrMagnitude < 1e-12f)
            {
                away = Vector3.up;
            }

            away.Normalize();
            return true;
        }
    }
}
