# Finding Your Way Home Without a Map App

*What a simulated fly taught us about memory, hunger, and the minimum data you need to return to a place that mattered.*

---

Most navigation stacks start with a GPS pin. Ours started with a hungry insect.

In our embodied fly simulation, the agent does not get a global map overlay or a “navigate to sugar” button. It walks a small chamber, steps through a tunnel into an outside world, smells plumes, tastes puddles, and builds a **place graph**—a mental map of *views*, *walls*, and *what happened here*. When energy drops, a behavior we call **route home** wakes up: instead of chemotaxing in a straight line through walls, the fly walks remembered **adjacent places** back to food.

That distinction sounds academic until you watch the failure modes. Straight-line smell following aims at the source *through* geometry. Graph routing aims at the source *through experience*: openings, corridors, and the places you actually visited.

## What “route home” actually does

At a high level, route home is a hungry-bout policy on top of a Neo4j-backed place graph:

1. **Lock one destination** for the bout. Once a home is chosen, we refuse to thrash between a nearby plume and a distant puddle mid-route.
2. **Prefer the earliest real food experience** when several sugar sites exist—not always the strongest current smell.
3. **Fall back** to session smell trails, outdoor plume peaks, bookmarked enclosed sites, or the last chamber place before the fly first stepped outside.
4. **Steer along the path**: hop-to-hop edges, opening-aware shortest paths, optional world XYZ bearings when places carry coordinates, and a landmark scan spin when the graph cannot yet reach home.
5. **Hand off** to local sugar search once contact taste says “you’re on it.”

Memory is not a string key. A place is a stereo visual embedding plus linked wall/sense detail. Transitions write `ADJACENT` edges. Hunger queries those edges—and the sugar / sugar_smell senses hanging off places—to rebuild a walkable story of “how I got here” and “where the meal was.”

In other words: **home is not a coordinate by default. Home is a remembered place in a graph of experience.**

## The data a real “route home” still needs

Simulation can cheat a little. Unity already knows where the body is. Places can carry `world_xyz`. Session trails are ordered by the steps we took.

If you lift this idea into robots, field agents, or multi-site autonomy, the graph alone is not enough for *grounded* return. A place that matters needs at least:

### 1. XYZ (GPS / local frame) coordinates

Without a position, “home” is only a similarity match in feature space. That is powerful for recognition—“I’ve seen this view before”—but weak for actuation when the path graph is incomplete. Coordinates let you:

- bear toward the next hop when adjacency is sparse
- detect arrival in meters, not only in embedding space
- debug routes as polylines in the real world

Call it GPS, ECEF, or a yard-local ENU frame—the requirement is the same: **a 3D (or at least 2D) pose attached to the experience.**

### 2. A timestamp

Time is how you decide *which* home wins.

Our stack already prefers the **first** sugar experience over a later, stronger smell. That policy only makes sense if every food memory has a time of encoding—`created_at`, GPS time, mission clock, whatever your stack trusts. Timestamps also age trails, expire stale outdoor plume peaks, and keep “yesterday’s kitchen” from beating “the meal I just left.”

Recognition without time collapses experience into a bag of places. **Navigation needs order.**

### 3. A planet name (when you leave Earth)

This sounds like science fiction until you write the schema.

GPS coordinates are not universal. A latitude/longitude pair, a Mars fixed frame, and a lunar local site survey are different worlds that happen to share the letters X, Y, and Z. If the same autonomy stack ever operates across bodies—Earth yard today, planetary analog tomorrow, multi-planetary eventually—you need a **planet (or body) identifier** beside pose and time:

```text
home = {
  planet: "Earth",          # or Mars, Moon, Ceres, …
  xyz: [x, y, z],           # frame defined for that body
  timestamp: "2026-11-22T16:01:00Z",
  place_id: "…",            # graph node / experience id
}
```

Without `planet`, two identical XYZ tuples are an accident waiting to happen. With it, route home stays well-typed: *return to this experience, on this body, at this recorded pose and time.*

## Why the fly still matters

We are not claiming a Drosophila has WGS84. The point of the simulation is different: **behavior first, schema second.**

Hunger should not freeze the agent while a planner recomputes the universe. One active route per bout. Smell can reinforce a heading without stealing the destination. Walls outside should bias you toward openings, not through plaster. Deep thought and stimulus interrupts should not yank you off a committed path unless the mission says so.

Those are product rules for any agent that must “get back to the thing that worked.” Coordinates, timestamps, and planet IDs are how you *anchor* those rules when the sandbox becomes the solar system.

## A practical checklist

If you are implementing route-home style memory in your own stack:

| Field | Why |
| --- | --- |
| **Place / experience id** | Stable node in the mental map |
| **XYZ coordinates** | Actuation, arrival, debugging |
| **Timestamp** | First-vs-strongest, trail aging, bout order |
| **Planet / body name** | Disambiguate frames across worlds |
| **Local senses** | Sugar, hazard, opening—why this place is home |
| **Adjacency** | How you actually walk there |

Vision embeddings and smell vectors are optional richness. Pose + time + planet are the minimum passport for a memory you intend to return to.

## Closing

Route home, in our project, is a small piece of fly-inspired autonomy: a hungry agent walking a place graph back to food. Scale the same idea outward and the poetry stays, but the passport gets stricter.

**To go home, remember where you were, when you were there—and, if you travel far enough, which world that “where” belonged to.**

---

*Built as part of an embodied fly-inspired learning simulation (Unity + Python mind, Neo4j place graph). Noncommercial research software; see the project license for terms.*
