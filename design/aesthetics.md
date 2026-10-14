# MARLAX visual standard

Approved direction: **Vector / 84**, a geometric, top-down vector-terminal style.
This is the repository's settled illustration direction. The active SVGs in
`site/assets/` are the source of truth for the asset gallery and simulation.

## Palette

| Role | Colour | Application |
| --- | --- | --- |
| Background and opaque interiors | White `#FFFFFF` | Pages, canvas, animal bodies |
| Outline and text | Black `#000000` | Animals, vehicles, board structure, navigation |
| Highlight | Dark red `#8B0000` (`darkred`) | Target beacons, meeting-point chevrons, selected pieces, keyboard focus |

Dark red communicates a destination, interaction state, or focus. It is not an
animal ornament: no arbitrary forehead stripes, coloured body panels, or coloured
ear interiors. Target beacons have dark-red outlines. The meeting point retains
its black octagonal outline and dark-red chevrons. A dragged agent has a dark-red
selection outline. Distinguish agents through silhouette and identity, not extra
colours. Do not introduce cyan, pink, gradients, glow, shading, or textures.

## Pieces

Use the simpler Vector / 84 silhouettes. Keep the polygonal ears, angular body
outline, pointed muzzle, and bent tail. Monkeys retain their long tail. The more
detailed Vector / 88 edition is archived and is not a selectable production theme.

Every animal and vehicle is viewed from overhead. No front-facing portrait or
standing character on the board. Interior lines describe an essential feature,
such as a vehicle window; they do not simulate fur, volume, lighting, or engraving.

The source view box is `0 0 256 256`, with anchor `(128, 128)` and forward direction
north (up in the SVG). Animal and vehicle outlines use a four-unit stroke with
square caps and miter joins. Keep the complete silhouette, including its tail,
inside the view box. The exterior is transparent; white body fills prevent
underlying board geometry from showing through the piece.

Movement translates and rotates the complete SVG. Do not animate individual body
parts, deform paths, or add idle motion. Preserve heading when a piece stops. The
canvas renderer converts its movement vector to a clockwise SVG rotation; no
redrawing of species inside the renderer.

## Board and markers

The selected world surface is **Circuit board**: `site/assets/boards/circuit.svg`.
Use its corner brackets, cell-centre points, and outer registration lines. Do not
replace it with a conventional ruled grid or the alternative Island/Lattice boards.

The board view box is `0 0 512 512`. The playable area starts at `(56, 56)` and spans
400 units: five rows and columns of 80-unit cells. Cell centres are
`(96 + 80 × column, 96 + 80 × row)`. Preserve space for the outer registration lines.

Use these existing marker shapes:

- Target: `site/assets/markers/beacon.svg`, the concentric beacon with diamond centre.
- Meeting point: `site/assets/markers/portal.svg`, the octagon with paired chevrons.

The meeting marker is at the centre cell. Target beacons appear at the selected
target cells when the environment reveals them. Markers are drawn beneath agents.
The simulation draws pieces at one cell per SVG view box and markers at 1.35 cells
per view box; the marker geometry occupies only the central part of that box.
Avoid clipping the markers or registration lines at the playable-area boundary.

## Gallery and navigation

Use the same assets and palette in the gallery and simulation. Provide visible
links between Documentation, Simulation, and Assets on desktop and mobile.
Use straightforward sans-serif typography, white space, and black text. Reserve
red for actionable or task-relevant emphasis. The selected design is presented
directly, without a style-comparison or edition-selection interface.

The composed reference is `site/assets/previews/approved-scene.svg`. Check artwork
both enlarged and at its actual cell size. At small sizes, silhouette and heading
must remain clear before any interior feature is considered.

## Scope and preservation

`site/assets/manifest.json` records the selected palette, board, markers, and
simulation pieces. Historical SVGs remain under `site/assets/collections/`; keep
them intact as design records. They are not the runtime source of truth.

Scientific figures continue to follow `STYLE.md`. Python APIs, package structure,
Sphinx documentation, learning rules, and policy artifacts are independent of this
visual standard. Changing a species skin does not change agent capabilities or
simulation dynamics.
