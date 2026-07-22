using System.Collections.Generic;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Aggregates chemical sources into a field sampled at world positions.
    /// </summary>
    public sealed class ChemicalFieldSystem : MonoBehaviour
    {
        private readonly List<ChemicalSource> _sources = new List<ChemicalSource>(8);

        public void Register(ChemicalSource source)
        {
            if (source != null && !_sources.Contains(source))
            {
                _sources.Add(source);
            }
        }

        public void Unregister(ChemicalSource source)
        {
            _sources.Remove(source);
        }

        public void Clear()
        {
            _sources.Clear();
        }

        public void DiscoverSources()
        {
            _sources.Clear();
            _sources.AddRange(FindObjectsByType<ChemicalSource>());
        }

        public ChemicalChannels Sample(Vector3 worldPosition)
        {
            var total = ChemicalChannels.Zero;
            for (int i = 0; i < _sources.Count; i++)
            {
                total += _sources[i].SampleAt(worldPosition);
            }

            return ChemicalChannels.Clamp01(total);
        }
    }
}
