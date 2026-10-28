using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Pointy-top hexagonal eye aperture with 6 triangular facets (center diagonals).
    /// Pixel math mirrors Python <c>flybrain.vision.hex_eye</c>.
    /// </summary>
    public static class FlyEyeHexGeometry
    {
        public const int FacetCount = 6;

        public static void CenterAndRadius(int height, int width, out float cx, out float cy, out float radius)
        {
            int h = Mathf.Max(1, height);
            int w = Mathf.Max(1, width);
            cx = 0.5f * (w - 1);
            cy = 0.5f * (h - 1);
            radius = 0.5f * (Mathf.Min(h, w) - 1) * 0.98f;
        }

        /// <summary>Vertex 0 at top; CCW in image space (y down).</summary>
        public static Vector2[] VerticesPx(int height, int width)
        {
            CenterAndRadius(height, width, out float cx, out float cy, out float radius);
            var verts = new Vector2[FacetCount];
            for (int i = 0; i < FacetCount; i++)
            {
                float angle = -0.5f * Mathf.PI + i * (Mathf.PI / 3f);
                verts[i] = new Vector2(
                    cx + radius * Mathf.Cos(angle),
                    cy - radius * Mathf.Sin(angle));
            }

            return verts;
        }

        public static bool PointInTriangle(float px, float py, Vector2 a, Vector2 b, Vector2 c)
        {
            float v0x = c.x - a.x, v0y = c.y - a.y;
            float v1x = b.x - a.x, v1y = b.y - a.y;
            float v2x = px - a.x, v2y = py - a.y;
            float dot00 = v0x * v0x + v0y * v0y;
            float dot01 = v0x * v1x + v0y * v1y;
            float dot02 = v0x * v2x + v0y * v2y;
            float dot11 = v1x * v1x + v1y * v1y;
            float dot12 = v1x * v2x + v1y * v2y;
            float denom = dot00 * dot11 - dot01 * dot01;
            if (Mathf.Abs(denom) < 1e-12f)
            {
                return false;
            }

            float inv = 1f / denom;
            float u = (dot11 * dot02 - dot01 * dot12) * inv;
            float v = (dot00 * dot12 - dot01 * dot02) * inv;
            return u >= -1e-6f && v >= -1e-6f && u + v <= 1f + 1e-6f;
        }

        public static bool InsideHexagon(float px, float py, int height, int width)
        {
            CenterAndRadius(height, width, out float cx, out float cy, out _);
            Vector2[] verts = VerticesPx(height, width);
            Vector2 center = new Vector2(cx, cy);
            for (int i = 0; i < FacetCount; i++)
            {
                int j = (i + 1) % FacetCount;
                if (PointInTriangle(px, py, center, verts[i], verts[j]))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>Zero RGB outside the hexagonal aperture (in-place on Color32 buffer).</summary>
        public static void ApplyApertureMask(Color32[] pixels, int width, int height)
        {
            if (pixels == null || pixels.Length < width * height)
            {
                return;
            }

            for (int y = 0; y < height; y++)
            {
                int row = y * width;
                for (int x = 0; x < width; x++)
                {
                    if (!InsideHexagon(x, y, height, width))
                    {
                        pixels[row + x] = new Color32(0, 0, 0, 255);
                    }
                }
            }
        }

        /// <summary>
        /// Draw hex outline + 6 spokes for PIP visualization (not used on CaptureFrame).
        /// </summary>
        public static void DrawFacetGuides(Color32[] pixels, int width, int height, Color32 color)
        {
            if (pixels == null || pixels.Length < width * height)
            {
                return;
            }

            CenterAndRadius(height, width, out float cx, out float cy, out _);
            Vector2[] verts = VerticesPx(height, width);
            Vector2 center = new Vector2(cx, cy);
            for (int i = 0; i < FacetCount; i++)
            {
                int j = (i + 1) % FacetCount;
                DrawLine(pixels, width, height, verts[i], verts[j], color);
                DrawLine(pixels, width, height, center, verts[i], color);
            }
        }

        private static void DrawLine(Color32[] pixels, int width, int height, Vector2 a, Vector2 b, Color32 color)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b)));
            for (int s = 0; s <= steps; s++)
            {
                float t = s / (float)steps;
                int x = Mathf.RoundToInt(Mathf.Lerp(a.x, b.x, t));
                int y = Mathf.RoundToInt(Mathf.Lerp(a.y, b.y, t));
                if (x < 0 || x >= width || y < 0 || y >= height)
                {
                    continue;
                }

                pixels[y * width + x] = color;
            }
        }
    }
}
