using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Runtime flat-color materials. The scene runs on the Built-in pipeline, where Unlit/Color
    /// never renders pink.
    /// </summary>
    public static class UnlitMaterials
    {
        public static Shader OpaqueShader()
        {
            return Shader.Find("Unlit/Color") ?? Shader.Find("Sprites/Default");
        }

        public static Material Create(string name, Color color)
        {
            return new Material(OpaqueShader()) { name = name, color = color };
        }
    }
}
