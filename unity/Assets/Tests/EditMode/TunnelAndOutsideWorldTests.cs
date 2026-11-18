using System.Collections.Generic;
using System.Linq;
using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class TunnelLayoutTests
    {
        private const float Eps = 1e-5f;

        [Test]
        public void DefaultOpening_IsHalfTheBoxWide_FloorToCeiling_80mmDeep()
        {
            var config = ExperimentConfiguration.Default();
            Vector3 size = config.ChamberSize;
            Assert.AreEqual(0.25f, TunnelLayout.OpeningWidth(config), Eps);
            Assert.AreEqual(size.y - ChamberController.FloorTopY(size), TunnelLayout.OpeningHeight(config), Eps);
            Assert.AreEqual(
                0.08f,
                TunnelLayout.ExitZ(config) - TunnelLayout.OuterWallZ(size),
                Eps);
        }

        [Test]
        public void FullHeightOpening_LeavesNoLintel_RoofMeetsCeiling()
        {
            var config = ExperimentConfiguration.Default();
            var lintel = TunnelLayout.NorthWallSegments(config).Single(s => s.Name == TunnelLayout.NorthWallLintel);
            var roof = TunnelLayout.TunnelPieces(config).Single(s => s.Name == TunnelLayout.TunnelRoof);
            Assert.IsFalse(lintel.HasVolume);
            Assert.AreEqual(config.ChamberSize.y, roof.Min.y, Eps);
        }

        [Test]
        public void PartialHeightOpening_UsesTunnelHeight_KeepsLintel()
        {
            var config = ExperimentConfiguration.Default();
            config.TunnelFullHeight = false;
            Assert.AreEqual(0.06f, TunnelLayout.OpeningHeight(config), Eps);
            var lintel = TunnelLayout.NorthWallSegments(config).Single(s => s.Name == TunnelLayout.NorthWallLintel);
            Assert.IsTrue(lintel.HasVolume);
        }

        [Test]
        public void NorthWallSegments_LeaveExactlyTheOpening()
        {
            var config = ExperimentConfiguration.Default();
            Vector3 size = config.ChamberSize;
            float halfOpening = TunnelLayout.OpeningWidth(config) * 0.5f;
            float openingTop = ChamberController.FloorTopY(size) + TunnelLayout.OpeningHeight(config);
            float wall = ChamberController.WallThickness(size);
            var segments = TunnelLayout.NorthWallSegments(config).ToDictionary(s => s.Name);

            BoxSpec left = segments[TunnelLayout.NorthWallLeft];
            BoxSpec right = segments[TunnelLayout.NorthWallRight];
            BoxSpec lintel = segments[TunnelLayout.NorthWallLintel];

            Assert.AreEqual(-size.x * 0.5f, left.Min.x, Eps);
            Assert.AreEqual(-halfOpening, left.Max.x, Eps);
            Assert.AreEqual(halfOpening, right.Min.x, Eps);
            Assert.AreEqual(size.x * 0.5f, right.Max.x, Eps);
            Assert.AreEqual(-halfOpening, lintel.Min.x, Eps);
            Assert.AreEqual(halfOpening, lintel.Max.x, Eps);
            Assert.AreEqual(openingTop, lintel.Min.y, Eps);
            Assert.AreEqual(size.y, lintel.Max.y, Eps);
            foreach (var segment in segments.Values)
            {
                Assert.AreEqual(size.z * 0.5f, segment.Center.z, Eps, segment.Name);
                Assert.AreEqual(wall, segment.Size.z, Eps, segment.Name);
            }
        }

        [Test]
        public void TunnelPieces_FormPassageFromWallToExit()
        {
            var config = ExperimentConfiguration.Default();
            Vector3 size = config.ChamberSize;
            float floorTop = ChamberController.FloorTopY(size);
            float halfOpening = TunnelLayout.OpeningWidth(config) * 0.5f;
            float openingTop = floorTop + TunnelLayout.OpeningHeight(config);
            float outer = TunnelLayout.OuterWallZ(size);
            float exit = TunnelLayout.ExitZ(config);
            var pieces = TunnelLayout.TunnelPieces(config).ToDictionary(s => s.Name);

            BoxSpec floor = pieces[TunnelLayout.TunnelFloor];
            Assert.AreEqual(floorTop, floor.Max.y, Eps, "tunnel floor flush with chamber floor");
            Assert.AreEqual(size.z * 0.5f, floor.Min.z, Eps, "starts where the chamber floor ends");
            Assert.AreEqual(exit, floor.Max.z, Eps);

            Assert.AreEqual(-halfOpening, pieces[TunnelLayout.TunnelWallLeft].Max.x, Eps);
            Assert.AreEqual(halfOpening, pieces[TunnelLayout.TunnelWallRight].Min.x, Eps);
            Assert.AreEqual(openingTop, pieces[TunnelLayout.TunnelRoof].Min.y, Eps);
            foreach (string name in new[] { TunnelLayout.TunnelWallLeft, TunnelLayout.TunnelWallRight, TunnelLayout.TunnelRoof })
            {
                Assert.AreEqual(outer, pieces[name].Min.z, Eps, name);
                Assert.AreEqual(exit, pieces[name].Max.z, Eps, name);
            }
        }

        [Test]
        public void OpeningHeight_ClampsToChamberInterior()
        {
            var config = ExperimentConfiguration.Default();
            config.TunnelFullHeight = false;
            config.TunnelHeight = 10f;
            Assert.AreEqual(
                config.ChamberSize.y - ChamberController.FloorTopY(config.ChamberSize),
                TunnelLayout.OpeningHeight(config),
                Eps);
        }
    }

    public sealed class ChamberTunnelControllerTests
    {
        private readonly List<GameObject> _created = new List<GameObject>();

        [TearDown]
        public void TearDown()
        {
            foreach (var go in _created)
            {
                Object.DestroyImmediate(go);
            }

            _created.Clear();
        }

        private ChamberController BuildChamber(out Transform[] walls, out Transform[] segments, out Transform[] pieces)
        {
            var root = new GameObject("Chamber");
            _created.Add(root);
            var config = ExperimentConfiguration.Default();
            walls = Enumerable.Range(0, 4).Select(i => Child(root, $"{ChamberWallHider.WallPrefix}{i}")).ToArray();
            segments = TunnelLayout.NorthWallSegments(config).Select(s => Child(root, s.Name)).ToArray();
            pieces = TunnelLayout.TunnelPieces(config).Select(s => Child(root, s.Name)).ToArray();
            var chamber = root.AddComponent<ChamberController>();
            chamber.BindStructure(Child(root, "Floor"), Child(root, ChamberWallHider.CeilingName), walls, segments, pieces);
            return chamber;
        }

        private static Transform Child(GameObject root, string name)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(root.transform, false);
            return go.transform;
        }

        [Test]
        public void TunnelEnabled_ReplacesNorthWallWithSegmentsAndPassage()
        {
            var chamber = BuildChamber(out var walls, out var segments, out var pieces);
            var config = ExperimentConfiguration.Default();
            chamber.ApplyConfiguration(config);

            Assert.IsTrue(chamber.TunnelActive);
            Assert.IsFalse(walls[0].gameObject.activeSelf, "solid north wall removed");
            Assert.IsTrue(walls[1].gameObject.activeSelf);
            var specs = TunnelLayout.NorthWallSegments(config).Concat(TunnelLayout.TunnelPieces(config)).ToArray();
            var parts = segments.Concat(pieces).ToArray();
            for (int i = 0; i < parts.Length; i++)
            {
                Assert.AreEqual(specs[i].HasVolume, parts[i].gameObject.activeSelf, parts[i].name);
                Assert.AreEqual(specs[i].Center, parts[i].localPosition, parts[i].name);
                if (specs[i].HasVolume)
                {
                    Assert.AreEqual(specs[i].Size, parts[i].localScale, parts[i].name);
                }
            }

            Assert.IsFalse(
                segments.Single(s => s.name == TunnelLayout.NorthWallLintel).gameObject.activeSelf,
                "full-height opening has no lintel");
        }

        [Test]
        public void TunnelDisabled_KeepsSolidNorthWall()
        {
            var chamber = BuildChamber(out var walls, out var segments, out var pieces);
            var config = ExperimentConfiguration.Default();
            config.TunnelEnabled = false;
            chamber.ApplyConfiguration(config);

            Assert.IsFalse(chamber.TunnelActive);
            Assert.IsTrue(walls[0].gameObject.activeSelf);
            Assert.IsTrue(segments.Concat(pieces).All(t => !t.gameObject.activeSelf));
        }

        [Test]
        public void WallLayer_VisibleToFlyEyes_CulledByOverview()
        {
            var chamber = BuildChamber(out var walls, out var segments, out var pieces);
            foreach (var renderer in chamber.GetComponentsInChildren<MeshRenderer>(true))
            {
                renderer.enabled = false;
            }

            ChamberWallHider.ApplyWallLayer(chamber.transform);

            foreach (var t in walls.Concat(segments))
            {
                Assert.AreEqual(ChamberWallHider.WallLayer, t.gameObject.layer, t.name);
                Assert.IsTrue(t.GetComponent<MeshRenderer>().enabled, t.name);
            }

            var tunnelFloor = pieces.First(p => p.name == TunnelLayout.TunnelFloor);
            Assert.AreEqual(0, tunnelFloor.gameObject.layer, "floors stay visible to the overview camera");
            Assert.IsTrue(pieces.Where(p => p != tunnelFloor).All(p => p.gameObject.layer == ChamberWallHider.WallLayer));

            int wallBit = 1 << ChamberWallHider.WallLayer;
            Assert.AreEqual(0, ChamberWallHider.OverviewCullingMask & wallBit);

            var eyeGo = new GameObject("eye");
            _created.Add(eyeGo);
            var cam = eyeGo.AddComponent<Camera>();
            eyeGo.AddComponent<FlyEyeCamera>().Configure(16, 16, 120f, false);
            Assert.AreNotEqual(0, cam.cullingMask & wallBit, "fly eyes see the walls");
            Assert.AreEqual(FlyEyeCamera.FarClip, cam.farClipPlane);
            cam.targetTexture = null;
        }
    }

    public sealed class OutsideWorldLayoutTests
    {
        private static ExperimentConfiguration Config => ExperimentConfiguration.Default();

        private static List<OutsidePlacement> Generate(int seed = 42)
        {
            var config = Config;
            return OutsideWorldLayout.Generate(seed, config.ChamberSize, TunnelLayout.ExitZ(config));
        }

        [Test]
        public void Generate_IsDeterministicPerSeed()
        {
            var a = Generate(7);
            var b = Generate(7);
            var c = Generate(8);
            Assert.AreEqual(a.Select(p => p.Position), b.Select(p => p.Position));
            Assert.AreNotEqual(a.Select(p => p.Position).ToArray(), c.Select(p => p.Position).ToArray());
        }

        [Test]
        public void Generate_HasExpectedFeatureCounts_AllOnGround()
        {
            var placements = Generate();
            float groundY = OutsideWorldLayout.GroundTopY(Config.ChamberSize);
            Assert.AreEqual(OutsideWorldLayout.FlowerCount, placements.Count(p => p.Kind == OutsideFeature.Flower));
            Assert.AreEqual(OutsideWorldLayout.TreeCount, placements.Count(p => p.Kind == OutsideFeature.Tree));
            Assert.AreEqual(OutsideWorldLayout.HouseCount, placements.Count(p => p.Kind == OutsideFeature.House));
            Assert.IsTrue(placements.All(p => Mathf.Approximately(p.Position.y, groundY)));
        }

        [Test]
        public void Flowers_StandInMeadowBeyondExit_AndTowerOverTheFly()
        {
            var config = Config;
            Rect meadow = OutsideWorldLayout.MeadowRect(TunnelLayout.ExitZ(config));
            float flyHeight = config.FlySpawnPosition.y;
            foreach (var flower in Generate().Where(p => p.Kind == OutsideFeature.Flower))
            {
                Assert.IsTrue(meadow.Contains(new Vector2(flower.Position.x, flower.Position.z)));
                Assert.Greater(flower.Position.z, TunnelLayout.ExitZ(config));
                Assert.GreaterOrEqual(flower.Height, OutsideWorldLayout.FlowerMinHeight);
                Assert.LessOrEqual(flower.Height, OutsideWorldLayout.FlowerMaxHeight);
                Assert.Greater(flower.Height, flyHeight * 2f);
            }
        }

        [Test]
        public void TreesAndTown_AreDistantBackdrops_HousesDoNotOverlap()
        {
            var placements = Generate();
            foreach (var tree in placements.Where(p => p.Kind == OutsideFeature.Tree))
            {
                float distance = new Vector2(tree.Position.x, tree.Position.z).magnitude;
                Assert.GreaterOrEqual(distance, OutsideWorldLayout.TreeMinDistance - 1e-3f);
                Assert.LessOrEqual(distance, OutsideWorldLayout.TreeMaxDistance + 1e-3f);
                Assert.GreaterOrEqual(tree.Height, OutsideWorldLayout.TreeMinHeight);
            }

            var houses = placements.Where(p => p.Kind == OutsideFeature.House).ToList();
            foreach (var house in houses)
            {
                float nearestEdge = house.Position.z - house.Width * OutsideWorldLayout.HouseDepthRatio * 0.5f;
                Assert.Greater(nearestEdge, OutsideWorldLayout.TreeMaxDistance, "town lies beyond the trees");
                Assert.Less(house.Position.magnitude, OutsideWorldLayout.ViewDistance, "inside camera far clip");
            }

            for (int i = 0; i < houses.Count; i++)
            {
                for (int j = i + 1; j < houses.Count; j++)
                {
                    float dx = Mathf.Abs(houses[i].Position.x - houses[j].Position.x);
                    float dz = Mathf.Abs(houses[i].Position.z - houses[j].Position.z);
                    float halfWidths = (houses[i].Width + houses[j].Width) * 0.5f;
                    float halfDepths = halfWidths * OutsideWorldLayout.HouseDepthRatio;
                    Assert.IsTrue(dx >= halfWidths || dz >= halfDepths, $"houses {i} and {j} overlap");
                }
            }
        }

        [Test]
        public void GrassBlades_FillMeadowAtDensity_WithinHeightRange()
        {
            var config = Config;
            float exitZ = TunnelLayout.ExitZ(config);
            Rect meadow = OutsideWorldLayout.MeadowRect(exitZ);
            var blades = OutsideWorldLayout.GrassBlades(config.RandomSeed, config.ChamberSize, exitZ);

            Assert.AreEqual(
                Mathf.RoundToInt(meadow.width * meadow.height * OutsideWorldLayout.GrassDensityPerSquareMeter),
                blades.Count);
            foreach (var blade in blades)
            {
                Assert.IsTrue(meadow.Contains(new Vector2(blade.Base.x, blade.Base.z)));
                Assert.GreaterOrEqual(blade.Height, OutsideWorldLayout.GrassMinHeight);
                Assert.LessOrEqual(blade.Height, OutsideWorldLayout.GrassMaxHeight);
                Assert.Less(blade.Shade, OutsideWorldLayout.GrassShades);
            }
        }

        [Test]
        public void OutsideGround_SitsJustBelowChamberFloor()
        {
            Vector3 size = Config.ChamberSize;
            float drop = ChamberController.FloorTopY(size) - OutsideWorldLayout.GroundTopY(size);
            Assert.AreEqual(OutsideWorldLayout.GroundDrop, drop, 1e-6f);
            Assert.Greater(drop, 0f);
        }
    }

    public sealed class OutsideWorldBuilderTests
    {
        private readonly List<Object> _created = new List<Object>();

        [TearDown]
        public void TearDown()
        {
            foreach (var obj in _created)
            {
                if (obj != null)
                {
                    Object.DestroyImmediate(obj);
                }
            }

            _created.Clear();
        }

        [Test]
        public void GrassMesh_HasDoubleSidedTrianglePerBlade()
        {
            var blades = new List<GrassBlade>
            {
                new GrassBlade(Vector3.zero, 0.1f, 0.004f, 0f, 0f, 0),
                new GrassBlade(new Vector3(0.05f, 0f, 0f), 0.05f, 0.004f, 90f, 0.3f, 1),
            };
            var mesh = OutsideWorldBuilder.BuildGrassMesh(blades);
            _created.Add(mesh);

            Assert.AreEqual(6, mesh.vertexCount);
            Assert.AreEqual(12, mesh.triangles.Length);
            Assert.AreEqual(0.1f, mesh.bounds.max.y, 1e-5f);
            Assert.AreEqual(0f, mesh.bounds.min.y, 1e-5f);
        }

        [Test]
        public void Build_WithPrimitives_CreatesGroundMeadowFlowersTreesAndTown()
        {
            var config = ExperimentConfiguration.Default();
            var root = OutsideWorldBuilder.Build(null, config, OutsideWorldAssets.CreatePalette());
            _created.Add(root);

            Transform ground = root.transform.Find(OutsideWorldBuilder.GroundName);
            Assert.IsNotNull(ground);
            Assert.IsNotNull(ground.GetComponent<Collider>());
            float groundTop = ground.position.y + ground.lossyScale.y * 0.5f;
            Assert.AreEqual(OutsideWorldLayout.GroundTopY(config.ChamberSize), groundTop, 1e-5f);

            Assert.AreEqual(
                OutsideWorldLayout.FlowerCount,
                root.transform.Find(OutsideWorldBuilder.FlowerGroupName).childCount);
            Assert.AreEqual(OutsideWorldLayout.TreeCount, root.transform.Find(OutsideWorldBuilder.TreeGroupName).childCount);
            Assert.AreEqual(OutsideWorldLayout.HouseCount, root.transform.Find(OutsideWorldBuilder.TownGroupName).childCount);
            Assert.AreEqual(
                OutsideWorldLayout.GrassShades,
                root.transform.Find(OutsideWorldBuilder.GrassGroupName).childCount);

            Transform flower = root.transform.Find(OutsideWorldBuilder.FlowerGroupName).GetChild(0);
            Assert.IsNotNull(flower.Find("Stem").GetComponent<Collider>(), "stems are obstacles");
            Assert.IsNull(flower.Find("Petal_0").GetComponent<Collider>(), "petals are decoration");
            Assert.AreEqual(OutsideWorldBuilder.PetalsPerFlower, flower.Cast<Transform>().Count(t => t.name.StartsWith("Petal_")));
        }

        [Test]
        public void Build_WithPrefabs_UsesPackInsteadOfPrimitives()
        {
            var prefab = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            prefab.name = "PackFlower";
            _created.Add(prefab);
            var assets = OutsideWorldAssets.CreatePalette();
            assets.FlowerPrefabs = new[] { prefab };

            var root = OutsideWorldBuilder.Build(null, ExperimentConfiguration.Default(), assets);
            _created.Add(root);

            Transform flowers = root.transform.Find(OutsideWorldBuilder.FlowerGroupName);
            Assert.AreEqual(OutsideWorldLayout.FlowerCount, flowers.childCount);
            foreach (Transform flower in flowers)
            {
                Assert.IsNull(flower.Find("Stem"), "primitive flower should not be built");
                Assert.IsNotNull(flower.GetComponent<CapsuleCollider>(), "prefab clone placed");
            }
        }

        [Test]
        public void FitPrefabHeight_ScalesToLayoutHeight_AndStandsOnGround()
        {
            var parent = new GameObject("group");
            _created.Add(parent);
            var instance = GameObject.CreatePrimitive(PrimitiveType.Cube);
            instance.transform.SetParent(parent.transform, false);
            var placement = new OutsidePlacement(OutsideFeature.Tree, new Vector3(1f, 0.0095f, 3f), 8f, 0f, 0f, 0);
            instance.transform.localPosition = placement.Position;

            OutsideWorldBuilder.FitPrefabHeight(instance.transform, placement);

            Bounds bounds = instance.GetComponent<Renderer>().bounds;
            Assert.AreEqual(8f, bounds.size.y, 1e-3f);
            Assert.AreEqual(placement.Position.y, bounds.min.y, 1e-3f);
        }
    }
}
