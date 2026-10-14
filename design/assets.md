# Animals, roles, and visual assets

This is an asset contract and organization proposal, not a request to generate
artwork in this pass. Backend extension should never require redrawing an animal.

## Three independent identities

| Identity | Example | Owner |
| --- | --- | --- |
| Agent / policy identity | `agent_0`, policy slot 0 | Environment and agent configuration |
| Physical role | Carrier with strength 2, scout with wider sensing | Environment rules and observations |
| Visual species | Mouse, monkey, elephant | Theme and scene configuration |

Changing a skin alone does not change learning or evaluation. Changing strength,
speed, sensing, or available actions changes the task and belongs in environment
configuration, with a new evaluated policy where needed. Species do not live in
the Q-table implementation.

## Initial art direction

Use one consistent top-down, simple outline family, extending the current mouse
style: transparent backgrounds, consistent outline weight, restrained detail,
and a shared body scale. The proposal keeps the dark neutral gallery background
but permits a light theme through palette tokens. Agent identity uses accent
color plus a small shape/label cue rather than species alone.

Begin with reusable SVG artwork for mouse, monkey, and elephant. Canonical facing
is positive x (east); rotate the asset for direction. The initial visual states
are idle and moving; movement can be a renderer transform without new artwork.
Add a collect pose only when needed. No mandatory sprite-sheet generation pipeline
or large animation-state matrix.

Scientific charts continue to use the existing `STYLE.md`. Scene art has its own
theme; chart typography should not become an accidental dependency of the runtime.

## Proposed storage

```text
web/assets/
  themes/
    outline-v1.json
  animals/
    mouse/
      outline-v1/
        manifest.json
        source.svg
        idle.svg
        moving.svg
    monkey/
      outline-v1/
    elephant/
      outline-v1/
  README.md                     # conventions and asset inventory
```

Each asset manifest records `id`, `version`, view box, normalized anchor, nominal
body length, default facing, available poses, author/source, and license. SVG
geometry is canonical; generated PNGs or atlases, if later useful, are derived
files. Keep external assets' source and license alongside them. Record generated
art's provenance when used, without putting prompts in simulation code.

Theme tokens define world background, grid, target, active target, agent accents,
outline width, and animation durations. One asset family can be reused across
many demos; each demo should not contain its own copied mouse drawing function.

## Scene binding

The gallery catalog binds an artifact to presentation, for example:

```json
{
  "demo": "mice-rendezvous",
  "artifact": "mice-rendezvous/v1/manifest.json",
  "theme": "outline-v1",
  "entities": [
    {"agent_id": "agent_0", "asset": "mouse/outline-v1", "accent": "agent_a"},
    {"agent_id": "agent_1", "asset": "mouse/outline-v1", "accent": "agent_b"}
  ]
}
```

A scene adapter projects environment state into entity positions/headings and
world markers. The renderer resolves the asset once and reuses it each frame.
Physical body radius and collision geometry come from the environment; decorative
ears, tails, and elephant trunks do not change collisions.

## Asset workflow

Create or select an animal once, normalize its anchor/scale/facing, add provenance,
and preview it in a small contact sheet with the other animals. Reuse it by ID in
scene bindings. Review the family together when changing a theme. This is a manual
design task performed occasionally, not a step repeated whenever a policy is
trained or a new experiment is added.
