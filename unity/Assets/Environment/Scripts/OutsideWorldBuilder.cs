using System;
using System.Collections.Generic;
using FlySimulation.Experiments;
using UnityEngine;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Materials and optional asset-pack prefabs for the outside world. A non-empty prefab array
    /// replaces the primitive version of that feature.
    /// </summary>
    public sealed class OutsideWorldAssets
    {
        public static readonly Color GroundColor = new Color(0.36f, 0.3f, 0.2f);
        public static readonly Color[] GrassColors =
        {
            new Color(0.25f, 0.52f, 0.18f), new Color(0.33f, 0.6f, 0.2f), new Color(0.2f, 0.42f, 0.15f)
        };
        public static readonly Color StemColor = new Color(0.22f, 0.45f, 0.16f);
        public static readonly Color FlowerCenterColor = new Color(0.98f, 0.8f, 0.1f);
        public static readonly Color[] PetalColors =
        {
            new Color(0.97f, 0.97f, 0.95f), new Color(0.95f, 0.45f, 0.65f), new Color(0.6f, 0.35f, 0.85f),
            new Color(0.98f, 0.85f, 0.2f), new Color(0.9f, 0.2f, 0.2f)
        };
        public static readonly Color TrunkColor = new Color(0.36f, 0.25f, 0.16f);
        public static readonly Color[] CanopyColors =
        {
            new Color(0.17f, 0.4f, 0.16f), new Color(0.24f, 0.48f, 0.2f), new Color(0.13f, 0.33f, 0.18f)
        };
        public static readonly Color[] HouseWallColors =
        {
            new Color(0.92f, 0.87f, 0.76f), new Color(0.8f, 0.55f, 0.42f), new Color(0.75f, 0.78f, 0.82f),
            new Color(0.95f, 0.93f, 0.9f)
        };
        public static readonly Color RoofColor = new Color(0.55f, 0.2f, 0.15f);

        public Material Ground;
        public Material[] Grass;
        public Material Stem;
        public Material FlowerCenter;
        public Material[] Petals;
        public Material Trunk;
        public Material[] Canopy;
        public Material[] HouseWalls;
        public Material Roof;

        public GameObject[] FlowerPrefabs = Array.Empty<GameObject>();
        public GameObject[] TreePrefabs = Array.Empty<GameObject>();
        public GameObject[] HousePrefabs = Array.Empty<GameObject>();

        /// <summary>Editor tools swap in PrefabUtility.InstantiatePrefab to keep prefab links.</summary>
        public Func<GameObject, Transform, GameObject> Instantiate = (prefab, parent) => Object.Instantiate(prefab, parent);

        /// <summary>Palette materials; <paramref name="makeMaterial"/> defaults to runtime unlit materials.</summary>
        public static OutsideWorldAssets CreatePalette(Func<string, Color, Material> makeMaterial = null)
        {
            makeMaterial ??= UnlitMaterials.Create;
            return new OutsideWorldAssets
            {
                Ground = makeMaterial("OutsideGroundMat", GroundColor),
                Grass = MakeAll(makeMaterial, "OutsideGrassMat", GrassColors),
                Stem = makeMaterial("OutsideStemMat", StemColor),
                FlowerCenter = makeMaterial("OutsideFlowerCenterMat", FlowerCenterColor),
                Petals = MakeAll(makeMaterial, "OutsidePetalMat", PetalColors),
                Trunk = makeMaterial("OutsideTrunkMat", TrunkColor),
                Canopy = MakeAll(makeMaterial, "OutsideCanopyMat", CanopyColors),
                HouseWalls = MakeAll(makeMaterial, "OutsideHouseMat", HouseWallColors),
                Roof = makeMaterial("OutsideRoofMat", RoofColor),
            };
        }

        private static Material[] MakeAll(Func<string, Color, Material> makeMaterial, string prefix, Color[] colors)
        {
            var materials = new Material[colors.Length];
            for (int i = 0; i < colors.Length; i++)
            {
                materials[i] = makeMaterial($"{prefix}_{i}", colors[i]);
            }

            return materials;
        }
    }

    /// <summary>
    /// Builds the outside world from <see cref="OutsideWorldLayout"/>: ground, meadow grass, flowers,
    /// trees and a town. Flower stems, tree trunks and house bodies keep colliders; decoration does not.
    /// </summary>
    public static class OutsideWorldBuilder
    {
        public const string RootName = "OutsideWorld";
        public const string GroundName = "OutsideGround";
        public const string GrassGroupName = "Grass";
        public const string FlowerGroupName = "Flowers";
        public const string TreeGroupName = "Trees";
        public const string TownGroupName = "Town";
        public const int PetalsPerFlower = 8;

        public static GameObject Build(Transform parent, ExperimentConfiguration config, OutsideWorldAssets assets)
        {
            var root = new GameObject(RootName);
            root.transform.SetParent(parent, false);

            Vector3 size = config.ChamberSize;
            float exitZ = TunnelLayout.ExitZ(config);
            BuildGround(root.transform, size, assets);
            BuildGrass(root.transform, OutsideWorldLayout.GrassBlades(config.RandomSeed, size, exitZ), assets);

            Transform flowers = Group(root.transform, FlowerGroupName);
            Transform trees = Group(root.transform, TreeGroupName);
            Transform town = Group(root.transform, TownGroupName);
            foreach (var placement in OutsideWorldLayout.Generate(config.RandomSeed, size, exitZ))
            {
                switch (placement.Kind)
                {
                    case OutsideFeature.Flower:
                        Place(flowers, placement, assets.FlowerPrefabs, assets, BuildFlower);
                        break;
                    case OutsideFeature.Tree:
                        Place(trees, placement, assets.TreePrefabs, assets, BuildTree);
                        break;
                    case OutsideFeature.House:
                        Place(town, placement, assets.HousePrefabs, assets, BuildHouse);
                        break;
                }
            }

            return root;
        }

        /// <summary>Double-sided blades (both windings share vertices) so Unlit back-face culling never hides them.</summary>
        public static Mesh BuildGrassMesh(IList<GrassBlade> blades)
        {
            var vertices = new Vector3[blades.Count * 3];
            var triangles = new int[blades.Count * 6];
            for (int i = 0; i < blades.Count; i++)
            {
                GrassBlade blade = blades[i];
                Quaternion yaw = Quaternion.Euler(0f, blade.YawDegrees, 0f);
                Vector3 halfWidth = yaw * Vector3.right * (blade.Width * 0.5f);
                Vector3 tip = blade.Base + Vector3.up * blade.Height + yaw * Vector3.forward * (blade.Lean * blade.Height);

                int v = i * 3;
                vertices[v] = blade.Base - halfWidth;
                vertices[v + 1] = blade.Base + halfWidth;
                vertices[v + 2] = tip;

                int t = i * 6;
                triangles[t] = v;
                triangles[t + 1] = v + 2;
                triangles[t + 2] = v + 1;
                triangles[t + 3] = v;
                triangles[t + 4] = v + 1;
                triangles[t + 5] = v + 2;
            }

            var mesh = new Mesh { name = "GrassBlades" };
            if (vertices.Length > 65535)
            {
                mesh.indexFormat = IndexFormat.UInt32;
            }

            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static void BuildGround(Transform root, Vector3 chamberSize, OutsideWorldAssets assets)
        {
            float top = OutsideWorldLayout.GroundTopY(chamberSize);
            var ground = Primitive(PrimitiveType.Cube, GroundName, root, assets.Ground, keepCollider: true);
            ground.transform.localPosition = new Vector3(0f, top - OutsideWorldLayout.GroundThickness * 0.5f, 0f);
            ground.transform.localScale = new Vector3(
                OutsideWorldLayout.GroundHalfExtent * 2f,
                OutsideWorldLayout.GroundThickness,
                OutsideWorldLayout.GroundHalfExtent * 2f);
            AddNeutralSurface(ground);
        }

        private static void BuildGrass(Transform root, List<GrassBlade> blades, OutsideWorldAssets assets)
        {
            Transform group = Group(root, GrassGroupName);
            int shades = assets.Grass != null && assets.Grass.Length > 0 ? assets.Grass.Length : 1;
            var byShade = new List<GrassBlade>[shades];
            for (int i = 0; i < shades; i++)
            {
                byShade[i] = new List<GrassBlade>();
            }

            foreach (var blade in blades)
            {
                byShade[blade.Shade % shades].Add(blade);
            }

            for (int i = 0; i < shades; i++)
            {
                var patch = new GameObject($"GrassShade_{i}");
                patch.transform.SetParent(group, false);
                patch.AddComponent<MeshFilter>().sharedMesh = BuildGrassMesh(byShade[i]);
                var renderer = patch.AddComponent<MeshRenderer>();
                renderer.sharedMaterial = Pick(assets.Grass, i);
                renderer.shadowCastingMode = ShadowCastingMode.Off;
            }
        }

        private static void Place(
            Transform group,
            OutsidePlacement placement,
            GameObject[] prefabs,
            OutsideWorldAssets assets,
            Action<GameObject, OutsidePlacement, OutsideWorldAssets> buildPrimitive)
        {
            GameObject instance;
            if (prefabs != null && prefabs.Length > 0)
            {
                instance = assets.Instantiate(prefabs[placement.Variant % prefabs.Length], group);
                instance.name = $"{placement.Kind}_{group.childCount - 1}";
                PoseOnGround(instance.transform, placement);
                FitPrefabHeight(instance.transform, placement);
                return;
            }

            instance = new GameObject($"{placement.Kind}_{group.childCount}");
            instance.transform.SetParent(group, false);
            PoseOnGround(instance.transform, placement);
            buildPrimitive(instance, placement, assets);
        }

        private static void PoseOnGround(Transform target, OutsidePlacement placement)
        {
            target.localPosition = placement.Position;
            target.localRotation = Quaternion.Euler(0f, placement.YawDegrees, 0f);
        }

        /// <summary>Scale an imported prefab to the layout height and stand it on the ground.</summary>
        public static void FitPrefabHeight(Transform instance, OutsidePlacement placement)
        {
            if (!TryGetRendererBounds(instance, out Bounds bounds) || bounds.size.y <= 1e-6f)
            {
                return;
            }

            float scale = placement.Height / bounds.size.y;
            instance.localScale *= scale;
            TryGetRendererBounds(instance, out bounds);
            Vector3 groundWorld = instance.parent != null
                ? instance.parent.TransformPoint(placement.Position)
                : placement.Position;
            instance.position += Vector3.up * (groundWorld.y - bounds.min.y);
        }

        private static bool TryGetRendererBounds(Transform root, out Bounds bounds)
        {
            bounds = default;
            bool found = false;
            foreach (var renderer in root.GetComponentsInChildren<Renderer>())
            {
                if (!found)
                {
                    bounds = renderer.bounds;
                    found = true;
                }
                else
                {
                    bounds.Encapsulate(renderer.bounds);
                }
            }

            return found;
        }

        private static void BuildFlower(GameObject flower, OutsidePlacement placement, OutsideWorldAssets assets)
        {
            float height = placement.Height;
            float head = placement.Width;
            float stemDiameter = Mathf.Max(0.002f, head * 0.08f);

            var stem = Primitive(PrimitiveType.Cylinder, "Stem", flower.transform, assets.Stem, keepCollider: true);
            stem.transform.localPosition = new Vector3(0f, height * 0.5f, 0f);
            stem.transform.localScale = new Vector3(stemDiameter, height * 0.5f, stemDiameter);
            AddNeutralSurface(stem);

            float center = head * 0.35f;
            var disc = Primitive(PrimitiveType.Sphere, "Center", flower.transform, assets.FlowerCenter, keepCollider: false);
            disc.transform.localPosition = new Vector3(0f, height, 0f);
            disc.transform.localScale = new Vector3(center, center * 0.5f, center);

            float petalLength = head * 0.5f;
            Material petal = Pick(assets.Petals, placement.Variant);
            for (int i = 0; i < PetalsPerFlower; i++)
            {
                var leaf = Primitive(PrimitiveType.Sphere, $"Petal_{i}", flower.transform, petal, keepCollider: false);
                Quaternion around = Quaternion.Euler(0f, i * 360f / PetalsPerFlower, 0f);
                leaf.transform.localRotation = around;
                leaf.transform.localPosition = new Vector3(0f, height, 0f) + around * Vector3.forward * (petalLength * 0.5f + center * 0.25f);
                leaf.transform.localScale = new Vector3(petalLength * 0.4f, petalLength * 0.08f, petalLength);
            }
        }

        private static void BuildTree(GameObject tree, OutsidePlacement placement, OutsideWorldAssets assets)
        {
            float height = placement.Height;
            float trunkHeight = height * 0.5f;
            float trunkDiameter = height * 0.06f;

            var trunk = Primitive(PrimitiveType.Cylinder, "Trunk", tree.transform, assets.Trunk, keepCollider: true);
            trunk.transform.localPosition = new Vector3(0f, trunkHeight * 0.5f, 0f);
            trunk.transform.localScale = new Vector3(trunkDiameter, trunkHeight * 0.5f, trunkDiameter);

            // Three overlapping crowns; the top one reaches the tree's full height.
            Material leaves = Pick(assets.Canopy, placement.Variant);
            float crown = height * 0.42f;
            Vector3[] offsets =
            {
                new Vector3(0f, height - crown * 0.5f, 0f),
                new Vector3(crown * 0.35f, height - crown * 0.85f, crown * 0.1f),
                new Vector3(-crown * 0.3f, height - crown * 0.8f, -crown * 0.2f),
            };
            for (int i = 0; i < offsets.Length; i++)
            {
                var canopy = Primitive(PrimitiveType.Sphere, $"Canopy_{i}", tree.transform, leaves, keepCollider: false);
                canopy.transform.localPosition = offsets[i];
                canopy.transform.localScale = Vector3.one * crown;
            }
        }

        private static void BuildHouse(GameObject house, OutsidePlacement placement, OutsideWorldAssets assets)
        {
            float width = placement.Width;
            float depth = width * OutsideWorldLayout.HouseDepthRatio;
            // Gable roof: a cube turned 45° about the ridge; its diagonal spans the house depth.
            float roofSide = depth / Mathf.Sqrt(2f);
            float wallHeight = Mathf.Max(placement.Height - depth * 0.5f, placement.Height * 0.5f);

            var body = Primitive(PrimitiveType.Cube, "Body", house.transform, Pick(assets.HouseWalls, placement.Variant), keepCollider: true);
            body.transform.localPosition = new Vector3(0f, wallHeight * 0.5f, 0f);
            body.transform.localScale = new Vector3(width, wallHeight, depth);

            var roof = Primitive(PrimitiveType.Cube, "Roof", house.transform, assets.Roof, keepCollider: false);
            roof.transform.localPosition = new Vector3(0f, wallHeight, 0f);
            roof.transform.localRotation = Quaternion.Euler(45f, 0f, 0f);
            roof.transform.localScale = new Vector3(width * 1.05f, roofSide, roofSide);
        }

        private static GameObject Primitive(PrimitiveType type, string name, Transform parent, Material material, bool keepCollider)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.transform.SetParent(parent, false);
            if (!keepCollider)
            {
                Object.DestroyImmediate(go.GetComponent<Collider>());
            }

            var renderer = go.GetComponent<MeshRenderer>();
            if (material != null)
            {
                renderer.sharedMaterial = material;
            }

            return go;
        }

        private static Transform Group(Transform root, string name)
        {
            var group = new GameObject(name);
            group.transform.SetParent(root, false);
            return group.transform;
        }

        private static Material Pick(Material[] materials, int index)
        {
            return materials != null && materials.Length > 0 ? materials[index % materials.Length] : null;
        }

        private static void AddNeutralSurface(GameObject go)
        {
            go.AddComponent<SurfaceChemistry>().SetChannels(ChemicalChannels.Zero);
        }
    }
}
