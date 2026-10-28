using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class SensorManagerTests
    {
        [Test]
        public void ExpectedVectorSize_MatchesLayoutFormula()
        {
            int expected =
                SensorManager.JointCount * 4 +
                SensorManager.FootCount * 5 +
                SensorManager.FootCount * SensorManager.ChemicalChannels +
                SensorManager.ChemicalChannels * 2 +
                (2 + SensorManager.ChemicalChannels) +
                SensorManager.InternalStateCount +
                SensorManager.BodyStateCount;

            Assert.AreEqual(197, SensorManager.ExpectedVectorSize);
            Assert.AreEqual(expected, SensorManager.ExpectedVectorSize);
            Assert.AreEqual(18, SensorManager.JointCount);
            Assert.AreEqual(6, SensorManager.FootCount);
            Assert.AreEqual(8, SensorManager.ChemicalChannels);
            Assert.AreEqual(8, SensorManager.InternalStateCount);
            Assert.AreEqual(13, SensorManager.BodyStateCount);
        }

        [Test]
        public void CollectVectorObservation_Unbound_ReturnsExpectedLength()
        {
            var go = new GameObject("sensors");
            var manager = go.AddComponent<SensorManager>();

            int length = manager.CollectVectorObservation(0.005f);

            Assert.AreEqual(SensorManager.ExpectedVectorSize, length);
            Assert.AreEqual(SensorManager.ExpectedVectorSize, manager.VectorObservationLength);
            Assert.GreaterOrEqual(manager.VectorObservation.Length, length);
            Object.DestroyImmediate(go);
        }
    }

    public sealed class ChemicalChannelsAlgebraTests
    {
        [Test]
        public void AddAndScale_ThenClamp()
        {
            var a = new ChemicalChannels { Sugar = 0.6f, Bitter = 0.5f };
            var b = new ChemicalChannels { Sugar = 0.7f, Water = 0.2f };
            ChemicalChannels summed = a + b;
            ChemicalChannels scaled = summed * 0.5f;
            ChemicalChannels clamped = ChemicalChannels.Clamp01(summed);

            Assert.AreEqual(1.3f, summed.Sugar, 1e-5f);
            Assert.AreEqual(0.65f, scaled.Sugar, 1e-5f);
            Assert.AreEqual(1f, clamped.Sugar, 1e-5f);
            Assert.AreEqual(0.5f, clamped.Bitter, 1e-5f);
            Assert.AreEqual(0.2f, clamped.Water, 1e-5f);
        }

        [Test]
        public void CopyTo_AndToArray_PreserveChannelOrder()
        {
            var channels = new ChemicalChannels
            {
                Sugar = 0.1f,
                Bitter = 0.2f,
                Salt = 0.3f,
                Acid = 0.4f,
                Water = 0.5f,
                Protein = 0.6f,
                Toxin = 0.7f,
                Odor = 0.8f
            };

            float[] array = channels.ToArray();
            var buffer = new float[10];
            channels.CopyTo(buffer, 2);

            CollectionAssert.AreEqual(
                new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f },
                array);
            Assert.AreEqual(0.1f, buffer[2], 1e-5f);
            Assert.AreEqual(0.8f, buffer[9], 1e-5f);
        }
    }

    public sealed class ChemicalFieldSystemTests
    {
        [Test]
        public void Sample_SumsRegisteredSourcesAndClamps()
        {
            var fieldGo = new GameObject("field");
            var field = fieldGo.AddComponent<ChemicalFieldSystem>();

            var sugarGo = new GameObject("sugar");
            sugarGo.transform.position = Vector3.zero;
            var sugar = sugarGo.AddComponent<ChemicalSource>();
            sugar.Configure(ChemicalSource.SourceType.Sugar, 1f, 0.25f);

            var bitterGo = new GameObject("bitter");
            bitterGo.transform.position = Vector3.zero;
            var bitter = bitterGo.AddComponent<ChemicalSource>();
            bitter.Configure(ChemicalSource.SourceType.Bitter, 1f, 0.25f);

            field.Register(sugar);
            field.Register(bitter);
            field.Register(sugar); // duplicate should be ignored

            ChemicalChannels sample = field.Sample(Vector3.zero);
            Assert.AreEqual(1f, sample.Sugar, 1e-5f);
            Assert.AreEqual(1f, sample.Bitter, 1e-5f);
            // Sugar + bitter both map into Odor, then Clamp01.
            Assert.AreEqual(1f, sample.Odor, 1e-5f);

            field.Unregister(bitter);
            sample = field.Sample(Vector3.zero);
            Assert.AreEqual(1f, sample.Sugar, 1e-5f);
            Assert.AreEqual(0f, sample.Bitter, 1e-5f);
            Assert.AreEqual(1f, sample.Odor, 1e-5f);

            field.Clear();
            sample = field.Sample(Vector3.zero);
            Assert.AreEqual(0f, sample.Sugar, 1e-5f);
            Assert.AreEqual(0f, sample.Odor, 1e-5f);

            Object.DestroyImmediate(fieldGo);
            Object.DestroyImmediate(sugarGo);
            Object.DestroyImmediate(bitterGo);
        }
    }

    public sealed class AntennaOdorSensorTests
    {
        [Test]
        public void Antenna_DetectsStrongerOdorCloserToSugarSource()
        {
            var fieldGo = new GameObject("field");
            var field = fieldGo.AddComponent<ChemicalFieldSystem>();

            var sugarGo = new GameObject("sugar");
            sugarGo.transform.position = Vector3.zero;
            const float radius = 0.100f;
            var sugar = sugarGo.AddComponent<ChemicalSource>();
            sugar.Configure(ChemicalSource.SourceType.Sugar, 1f, radius);
            field.Register(sugar);

            var nearGo = new GameObject("antenna_near");
            nearGo.transform.position = new Vector3(0.02f, 0f, 0f);
            var near = nearGo.AddComponent<AntennaOdorSensor>();
            near.BindField(field);

            var midGo = new GameObject("antenna_mid");
            midGo.transform.position = new Vector3(radius, 0f, 0f);
            var mid = midGo.AddComponent<AntennaOdorSensor>();
            mid.BindField(field);

            var farGo = new GameObject("antenna_far");
            farGo.transform.position = new Vector3(radius + 0.001f, 0f, 0f);
            var far = farGo.AddComponent<AntennaOdorSensor>();
            far.BindField(field);

            ChemicalChannels nearSample = near.SampleNow();
            ChemicalChannels midSample = mid.SampleNow();
            ChemicalChannels farSample = far.SampleNow();

            float expectedNear = Mathf.Exp(-0.02f / radius);
            float expectedMid = Mathf.Exp(-1f);
            const float expectedFar = 0f;

            Assert.AreEqual(expectedNear, nearSample.Odor, 1e-4f);
            Assert.AreEqual(expectedMid, midSample.Odor, 1e-4f);
            Assert.AreEqual(expectedFar, farSample.Odor, 1e-4f);
            // Sugar plume also fills Sugar; Odor is the mapped antenna cue.
            Assert.AreEqual(expectedNear, nearSample.Sugar, 1e-4f);
            Assert.Greater(nearSample.Odor, midSample.Odor);
            Assert.Greater(midSample.Odor, farSample.Odor);

            var buf = new float[8];
            near.CopyObservation(buf, 0);
            Assert.AreEqual(expectedNear, buf[0], 1e-4f); // Sugar
            Assert.AreEqual(expectedNear, buf[7], 1e-4f); // Odor

            Object.DestroyImmediate(fieldGo);
            Object.DestroyImmediate(sugarGo);
            Object.DestroyImmediate(nearGo);
            Object.DestroyImmediate(midGo);
            Object.DestroyImmediate(farGo);
        }

        [Test]
        public void MountedAntennae_RightSmellsMoreSugarWhenSourceIsRight()
        {
            var fieldGo = new GameObject("field");
            var field = fieldGo.AddComponent<ChemicalFieldSystem>();
            var sugarGo = new GameObject("sugar");
            sugarGo.transform.position = new Vector3(0.2f, 0f, 0f);
            var sugar = sugarGo.AddComponent<ChemicalSource>();
            sugar.Configure(
                ChemicalSource.SourceType.Sugar,
                1f,
                ExperimentConfiguration.SugarPlumeRadius,
                ExperimentConfiguration.SugarPlumeEdgeFraction);
            field.Register(sugar);

            var mount = new GameObject("AntennaMount");
            var leftGo = new GameObject("LeftAntenna");
            leftGo.transform.SetParent(mount.transform, false);
            leftGo.transform.localPosition = AntennaOdorSensor.MountOffset(isLeft: true);
            var left = leftGo.AddComponent<AntennaOdorSensor>();
            left.BindField(field);
            var rightGo = new GameObject("RightAntenna");
            rightGo.transform.SetParent(mount.transform, false);
            rightGo.transform.localPosition = AntennaOdorSensor.MountOffset(isLeft: false);
            var right = rightGo.AddComponent<AntennaOdorSensor>();
            right.BindField(field);

            Assert.AreEqual(
                2f * AntennaOdorSensor.HalfSpacing,
                Vector3.Distance(leftGo.transform.position, rightGo.transform.position),
                1e-6f);
            float l = left.SampleNow().Sugar;
            float r = right.SampleNow().Sugar;
            Assert.Greater(r, l);
            // Python amplifies (r - l) / (r + l) by ANTENNA_BEARING_GAIN (20) into a bearing.
            Assert.Greater(20f * (r - l) / (r + l), 0.3f);

            Object.DestroyImmediate(fieldGo);
            Object.DestroyImmediate(sugarGo);
            Object.DestroyImmediate(mount);
        }

        [Test]
        public void AntennaGeometry_TipsSpreadWiderThanBaseAndPointForwardUp()
        {
            foreach (bool isLeft in new[] { true, false })
            {
                Vector3 b = AntennaOdorSensor.BaseOffset(isLeft);
                Vector3 t = AntennaOdorSensor.MountOffset(isLeft);
                Assert.Greater(Mathf.Abs(t.x), Mathf.Abs(b.x));
                Assert.Greater(t.y, b.y);
                Assert.Greater(t.z, b.z);
                Assert.AreEqual(Vector3.Distance(b, t), AntennaOdorSensor.StalkLength, 1e-6f);
            }
            Assert.AreEqual(0.038f, 2f * AntennaOdorSensor.HalfSpacing, 1e-6f);
            Assert.AreEqual(
                Mathf.Abs(AntennaOdorSensor.MountOffset(false).x),
                AntennaOdorSensor.HalfSpacing,
                1e-6f);
        }

        [Test]
        public void AntennaGeometry_StalkIsSixTimesTheOriginalLength()
        {
            float unit = new Vector3(
                AntennaOdorSensor.UnitStalkX,
                AntennaOdorSensor.UnitStalkY,
                AntennaOdorSensor.UnitStalkZ).magnitude;
            Assert.AreEqual(6f, AntennaOdorSensor.StalkScale, 1e-6f);
            Assert.AreEqual(6f * unit, AntennaOdorSensor.StalkLength, 1e-6f);
        }

        [Test]
        public void EyeCamera_CullsFlySelfLayer()
        {
            var go = new GameObject("eye");
            var cam = go.AddComponent<Camera>();
            var eye = go.AddComponent<FlyEyeCamera>();
            eye.Configure(64, 64, 140f, false);
            Assert.AreEqual(0, cam.cullingMask & (1 << FlyEyeCamera.SelfLayer));
            Assert.AreNotEqual(0, cam.cullingMask & 1); // Default layer still visible
            cam.targetTexture = null;
            Object.DestroyImmediate(go);
        }

        [Test]
        public void Antenna_WithoutField_ReadsZero()
        {
            var go = new GameObject("antenna");
            var antenna = go.AddComponent<AntennaOdorSensor>();
            ChemicalChannels sample = antenna.SampleNow();
            Assert.AreEqual(0f, sample.Odor, 1e-5f);
            Assert.AreEqual(0f, sample.Sugar, 1e-5f);
            Object.DestroyImmediate(go);
        }
    }

    public sealed class HomeostasisExtendedTests
    {
        [Test]
        public void CopyNormalized_WritesEightValues()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.InitialEnergy = 0.8f;
            config.InitialHydration = 0.6f;
            config.TemperatureCelsius = 25f;
            homeo.Configure(config);

            var buffer = new float[8];
            homeo.CopyNormalized(buffer);

            Assert.AreEqual(0.8f, buffer[0], 1e-5f);
            Assert.AreEqual(0.6f, buffer[1], 1e-5f);
            Assert.AreEqual(Mathf.Clamp01((25f - 10f) / 30f), buffer[4], 1e-5f);
            Assert.AreEqual(SensorManager.InternalStateCount, buffer.Length);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void DamageWaterToxin_AndIsDepleted()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.InitialEnergy = 1f;
            config.InitialHydration = 1f;
            homeo.Configure(config);

            Assert.IsFalse(homeo.IsDepleted);

            homeo.ConsumeWater(0.2f);
            Assert.AreEqual(1f, homeo.Hydration, 1e-5f); // already full, clamped

            homeo.ApplyCollisionDamage(0.4f);
            Assert.AreEqual(0.4f, homeo.Damage, 1e-5f);

            homeo.ApplyBitterOrToxin(0.7f);
            Assert.GreaterOrEqual(homeo.Damage, 1f - 1e-5f);
            Assert.IsTrue(homeo.IsDepleted);

            Assert.AreEqual(0.1f, FlyHomeostasis.ComputeEnergyDelta(0.5f, 0.6f), 1e-5f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void Tick_WithWings_BurnsExtraEnergy()
        {
            var go = new GameObject("homeostasis");
            var walking = go.AddComponent<FlyHomeostasis>();
            var flying = new GameObject("flying").AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.MetabolismRate = 0.01f;
            config.WingEnergyCost = 0.5f;
            config.MovementEnergyCost = 0f;
            walking.Configure(config);
            flying.Configure(config);

            walking.Tick(1f, motorEffort: 0f, usingWings: false);
            flying.Tick(1f, motorEffort: 0f, usingWings: true);

            Assert.Less(flying.Energy, walking.Energy);
            Object.DestroyImmediate(go);
            Object.DestroyImmediate(flying.gameObject);
        }
    }

    public sealed class RewardCalculatorExtendedTests
    {
        [Test]
        public void Compute_TracksEnergyDelta_AndAppliesFellPenalty()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.CuriosityWeight = 0f;
            config.EnergyWeight = 1f;
            config.HydrationWeight = 0f;
            config.DamageWeight = 0f;
            config.EffortWeight = 0f;
            config.InstabilityWeight = 0f;
            config.InitialEnergy = 0.5f;
            config.MetabolismRate = 0f;
            config.MovementEnergyCost = 0f;
            homeo.Configure(config);

            var calculator = new RewardCalculator(config);
            calculator.Reset(homeo);

            homeo.ConsumeSugar(0.2f);
            float reward = calculator.Compute(homeo, curiosity: 0f, motorEffort: 0f, instability: 0f, fell: false);
            Assert.AreEqual(0.2f, reward, 1e-4f);

            float fellReward = calculator.Compute(homeo, curiosity: 0f, motorEffort: 0f, instability: 0f, fell: true);
            Assert.AreEqual(-1f, fellReward, 1e-4f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void ComputeStatic_ClampsCuriosity()
        {
            var config = ExperimentConfiguration.Default();
            config.CuriosityWeight = 1f;
            config.EnergyWeight = 0f;
            config.HydrationWeight = 0f;
            config.DamageWeight = 0f;
            config.EffortWeight = 0f;
            config.InstabilityWeight = 0f;

            float reward = RewardCalculator.ComputeStatic(
                curiosity: 5f,
                energyImprovement: 0f,
                hydrationImprovement: 0f,
                damage: 0f,
                motorEffort: 0f,
                instability: 0f,
                config: config);

            Assert.AreEqual(1f, reward, 1e-5f);
        }
    }

    public sealed class LegJointEdgeTests
    {
        [Test]
        public void DistanceFromLimits_AtBoundaryIsZero()
        {
            Assert.AreEqual(0f, LegJoint.DistanceFromLimitsNormalized(-60f, -60f, 60f), 1e-4f);
            Assert.AreEqual(0f, LegJoint.DistanceFromLimitsNormalized(60f, -60f, 60f), 1e-4f);
            Assert.AreEqual(0f, LegJoint.DistanceFromLimitsNormalized(100f, -60f, 60f), 1e-4f);
        }

        [Test]
        public void SetNormalizedTarget_MapsMidAndExtremes()
        {
            var go = new GameObject("joint");
            var joint = go.AddComponent<LegJoint>();
            joint.SetNormalizedTarget(0f);
            Assert.AreEqual(0f, joint.TargetDegrees, 1e-3f);
            joint.SetNormalizedTarget(1f);
            Assert.AreEqual(60f, joint.TargetDegrees, 1e-3f);
            joint.SetNormalizedTarget(-1f);
            Assert.AreEqual(-60f, joint.TargetDegrees, 1e-3f);
            Object.DestroyImmediate(go);
        }
    }

    public sealed class ProboscisSensorTests
    {
        [Test]
        public void SetExtension_ClampsAndCopyObservationLayout()
        {
            var go = new GameObject("proboscis");
            var sensor = go.AddComponent<ProboscisSensor>();
            sensor.SetExtension(2f);

            var buffer = new float[10];
            sensor.CopyObservation(buffer, 0);

            Assert.AreEqual(1f, buffer[0], 1e-5f);
            Assert.AreEqual(0f, buffer[1], 1e-5f); // not contacting
            for (int i = 2; i < 10; i++)
            {
                Assert.AreEqual(0f, buffer[i], 1e-5f);
            }

            Object.DestroyImmediate(go);
        }

        private static Collider MakeSurface(string name, ChemicalChannels channels)
        {
            var go = new GameObject(name);
            var col = go.AddComponent<BoxCollider>();
            go.AddComponent<SurfaceChemistry>().SetChannels(channels);
            return col;
        }

        [Test]
        public void ReportContact_SugarSurface_GatesTasteIntoObservation()
        {
            var go = new GameObject("proboscis");
            var sensor = go.AddComponent<ProboscisSensor>();
            sensor.SetExtension(1f);
            var sugar = MakeSurface("sugar", new ChemicalChannels { Sugar = 1f });

            sensor.ReportContact(sugar);
            var buffer = new float[10];
            sensor.CopyObservation(buffer, 0);
            Assert.AreEqual(1f, buffer[1], 1e-5f);
            Assert.AreEqual(1f, buffer[2], 1e-5f); // sugar channel
            Assert.AreEqual(1f, sensor.GatedTaste.Sugar, 1e-5f);

            sensor.ReportExit();
            Assert.IsFalse(sensor.IsContacting);
            Assert.AreEqual(0f, sensor.GatedTaste.Sugar, 1e-5f);

            Object.DestroyImmediate(go);
            Object.DestroyImmediate(sugar.gameObject);
        }

        [Test]
        public void ReportContact_FloorInSameStep_DoesNotMaskSugar()
        {
            var go = new GameObject("proboscis");
            var sensor = go.AddComponent<ProboscisSensor>();
            var sugar = MakeSurface("sugar", new ChemicalChannels { Sugar = 1f });
            var floor = MakeSurface("floor", ChemicalChannels.Zero);

            sensor.ReportContact(sugar);
            sensor.ReportContact(floor);
            Assert.AreEqual(1f, sensor.GatedTaste.Sugar, 1e-5f);

            Object.DestroyImmediate(go);
            Object.DestroyImmediate(sugar.gameObject);
            Object.DestroyImmediate(floor.gameObject);
        }

        [Test]
        public void TipContact_RelaysTriggerToSensor()
        {
            var go = new GameObject("proboscis");
            var sensor = go.AddComponent<ProboscisSensor>();
            var tip = new GameObject("tip");
            tip.AddComponent<ProboscisTipContact>().Bind(sensor);
            var sugar = MakeSurface("sugar", new ChemicalChannels { Sugar = 1f });

            tip.SendMessage("OnTriggerStay", sugar);
            Assert.IsTrue(sensor.IsContacting);
            tip.SendMessage("OnTriggerExit", sugar);
            Assert.IsFalse(sensor.IsContacting);

            Object.DestroyImmediate(go);
            Object.DestroyImmediate(tip);
            Object.DestroyImmediate(sugar.gameObject);
        }

        [Test]
        public void Configure_DownwardMount_TipReachesFloorAtContactExtension()
        {
            const float drop = 0.03f;
            var mount = new GameObject("mount");
            mount.transform.position = new Vector3(0f, 0.04f, 0f);
            var probGo = new GameObject("proboscis");
            probGo.transform.SetParent(mount.transform, false);
            probGo.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            var tip = new GameObject("tip");
            tip.transform.SetParent(probGo.transform, false);
            var sensor = probGo.AddComponent<ProboscisSensor>();
            sensor.Configure(tip.transform, ProboscisSensor.MaxExtensionForDrop(drop));

            sensor.SetExtension(0f);
            Assert.AreEqual(0.04f, tip.transform.position.y, 1e-5f);
            sensor.SetExtension(ProboscisSensor.FloorContactExtension);
            Assert.AreEqual(0.04f - drop, tip.transform.position.y, 1e-5f);

            Object.DestroyImmediate(mount);
        }
    }

    public sealed class FlyHeadPoseTests
    {
        private static (GameObject fly, Transform mount, Transform head, ProboscisSensor prob, FlyHeadPose pose) Build()
        {
            var fly = new GameObject("fly");
            var head = new GameObject("head").transform;
            head.SetParent(fly.transform, false);
            head.localPosition = new Vector3(0f, 0f, 0.01f);
            var mount = new GameObject("mount").transform;
            mount.SetParent(fly.transform, false);
            mount.localPosition = head.localPosition;
            var probGo = new GameObject("proboscis");
            probGo.transform.SetParent(mount, false);
            var prob = probGo.AddComponent<ProboscisSensor>();
            var pose = fly.AddComponent<FlyHeadPose>();
            pose.Bind(mount, head, prob);
            return (fly, mount, head, prob, pose);
        }

        [Test]
        public void TargetLower_OnlyWhileProbing()
        {
            Assert.AreEqual(0f, FlyHeadPose.TargetLower(0f), 1e-6f);
            Assert.AreEqual(0f, FlyHeadPose.TargetLower(0.35f), 1e-6f); // idle composer extension
            Assert.AreEqual(0f, FlyHeadPose.TargetLower(FlyHeadPose.ProbeExtensionStart), 1e-6f);
            Assert.AreEqual(1f, FlyHeadPose.TargetLower(1f), 1e-6f);
        }

        [Test]
        public void Apply_FullLower_DropsAndPitchesHeadAndLowersAntennaTips()
        {
            var (fly, mount, head, prob, pose) = Build();
            var tip = new GameObject("tip").transform;
            tip.SetParent(mount, false);
            tip.localPosition = AntennaOdorSensor.MountOffset(isLeft: false);
            float restTipY = tip.position.y;
            float restHeadY = head.position.y;

            pose.Apply(1f);

            Assert.AreEqual(restHeadY - FlyHeadPose.MaxDrop, head.position.y, 1e-6f);
            Assert.AreEqual(FlyHeadPose.MaxPitchDegrees, mount.localEulerAngles.x, 1e-3f);
            Assert.Less(tip.position.y, restTipY - FlyHeadPose.MaxDrop);
            Assert.Greater(tip.position.z, AntennaOdorSensor.MountOffset(false).z + 0.01f);
            Object.DestroyImmediate(fly);
        }

        [Test]
        public void Apply_KeepsProboscisPointingStraightDown()
        {
            var (fly, _, _, prob, pose) = Build();
            foreach (float lower in new[] { 0f, 0.5f, 1f })
            {
                pose.Apply(lower);
                Vector3 axis = prob.transform.forward; // extension axis (local +Z)
                Assert.AreEqual(-1f, axis.y, 1e-4f, $"lower={lower}");
            }

            Object.DestroyImmediate(fly);
        }

        [Test]
        public void FullExtension_FullyLoweredHead_TipLandsOnFloorNotThroughIt()
        {
            var (fly, mount, _, prob, pose) = Build();
            const float floorTop = 0.01f;
            fly.transform.position = new Vector3(0f, 0.038f, 0f);
            var baseOnMount = new Vector3(0f, -0.003f, 0.004f);
            prob.transform.localPosition = baseOnMount;
            var tip = new GameObject("tip").transform;
            tip.SetParent(prob.transform, false);
            pose.Apply(0f);
            float drop = prob.transform.position.y - floorTop - FlyHeadPose.FullLowerExtraDrop(baseOnMount);
            prob.Configure(tip, ProboscisSensor.MaxExtensionForDrop(drop));

            pose.Apply(1f);
            prob.SetExtension(1f);
            Assert.AreEqual(floorTop, tip.position.y, 1e-5f);

            pose.Apply(0f);
            Assert.Greater(tip.position.y, floorTop + 0.005f, "raised head keeps the extended tip off the floor");
            Object.DestroyImmediate(fly);
        }

        [Test]
        public void Apply_ZeroRestoresRestPose()
        {
            var (fly, mount, head, _, pose) = Build();
            Vector3 mountRest = mount.localPosition;
            Vector3 headRest = head.localPosition;
            pose.Apply(1f);
            pose.Apply(0f);
            Assert.AreEqual(mountRest, mount.localPosition);
            Assert.AreEqual(headRest, head.localPosition);
            Assert.AreEqual(Quaternion.identity, mount.localRotation);
            Object.DestroyImmediate(fly);
        }
    }

    public sealed class ExperimentConfigurationExtendedTests
    {
        [Test]
        public void Default_RewardWeights_MatchPythonDefaults()
        {
            var config = ExperimentConfiguration.Default();
            Assert.AreEqual(0.1f, config.CuriosityWeight, 1e-5f);
            Assert.AreEqual(1f, config.EnergyWeight, 1e-5f);
            Assert.AreEqual(1f, config.HydrationWeight, 1e-5f);
            Assert.AreEqual(2f, config.DamageWeight, 1e-5f);
            Assert.AreEqual(0.05f, config.EffortWeight, 1e-5f);
            Assert.AreEqual(0.5f, config.InstabilityWeight, 1e-5f);
            Assert.AreEqual(new Vector3(0.5f, 0.25f, 0.5f), config.ChamberSize);
            Assert.AreEqual(1f, config.BodyScale, 1e-5f);
            Assert.AreEqual(0f, config.FlySpawnPosition.x, 1e-5f);
            Assert.AreEqual(0f, config.FlySpawnPosition.z, 1e-5f);
            Assert.Greater(config.FlySpawnPosition.y, 0f);
            Assert.IsTrue(config.LocomotionEnabled);
        }

        [Test]
        public void Json_RoundTripsRewardWeights()
        {
            var config = ExperimentConfiguration.Default();
            config.CuriosityWeight = 0.25f;
            config.DamageWeight = 3.5f;
            var restored = ExperimentConfiguration.FromJson(config.ToJson());
            Assert.AreEqual(0.25f, restored.CuriosityWeight, 1e-5f);
            Assert.AreEqual(3.5f, restored.DamageWeight, 1e-5f);
        }
    }
}
