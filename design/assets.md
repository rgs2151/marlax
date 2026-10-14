# Visual assets

The approved aesthetic is specified in [MARLAX visual standard](aesthetics.md):
Vector / 84 pieces, black outlines on white, dark-red highlights, the Circuit
board, and the existing target beacon and meeting-point markers.

## Separate identities

| Identity | Owner |
| --- | --- |
| Agent and policy slot | Environment and agent configuration |
| Physical capabilities | Environment dynamics and observations |
| Visual species | Scene and SVG assets |

Changing artwork does not change learning, movement rules, or evaluation.

## Active files

```text
site/assets/
  manifest.json
  animals/
    mice/
    monkeys/
    elephant.svg
    fox.svg
  vehicles/
  boards/circuit.svg
  markers/beacon.svg
  markers/portal.svg
  previews/approved-scene.svg
  collections/                 # preserved design editions
```

Active animals and vehicles have a 256-unit square view box, a centre anchor,
and a north-facing orientation. The renderer loads the SVG files and moves each
piece as a whole. The gallery downloads those same files. Board and marker
placement follow the geometry in the visual standard.
