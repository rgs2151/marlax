import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js";


const container = document.getElementById("world");
const response = await fetch("./data/coop_grid_q_learning.json");
const data = await response.json();

const env = data.env;
const q = data.policy.q;
const qShape = data.policy.q_shape;
const size = env.size;
const moves = env.moves;
const targets = env.targets;
const center = env.center;
const cellScale = 1.35;
const half = (size - 1) / 2;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f1ea);

const camera = new THREE.OrthographicCamera(-6, 6, 6, -6, 0.1, 100);
camera.position.set(5.6, 8.2, 7.2);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
container.appendChild(renderer.domElement);

const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const groundHit = new THREE.Vector3();

const random = mulberry32(Math.floor(performance.timeOrigin));
const agents = [
  makeAgent(0, 0x8f2434),
  makeAgent(1, 0x233f83),
];

let active = false;
let targetId = 0;
let stepCount = 0;
let collectedFrames = 0;
let decisionClock = 0;
let draggedAgent = null;
let pointerGrid = null;
let pointerDown = false;

makeLights();
makeGround();
const centerMarker = makeMarker(0xd8a328, "box");
const targetMarkers = [makeMarker(0x2f7d4f, "circle"), makeMarker(0x2f7d4f, "circle")];
resetTrial();
resize();
requestAnimationFrame(animate);

function mulberry32(seed) {
  let value = seed >>> 0;
  return function () {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gridToWorld(point) {
  return new THREE.Vector3((point[0] - half) * cellScale, 0, (point[1] - half) * cellScale);
}

function worldToGrid(point) {
  return [
    clamp(point.x / cellScale + half, 0, size - 1),
    clamp(point.z / cellScale + half, 0, size - 1),
  ];
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function makeLights() {
  scene.add(new THREE.HemisphereLight(0xffffff, 0xc7bba8, 2.4));
  const light = new THREE.DirectionalLight(0xffffff, 2.6);
  light.position.set(5, 8, 4);
  scene.add(light);
}

function makeGround() {
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(size * cellScale, size * cellScale),
    new THREE.MeshStandardMaterial({ color: 0xfaf8f2, roughness: 0.86 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const points = [];
  const min = -half * cellScale;
  const max = half * cellScale;
  for (let i = 0; i < size; i += 1) {
    const p = (i - half) * cellScale;
    points.push(new THREE.Vector3(min, 0.012, p), new THREE.Vector3(max, 0.012, p));
    points.push(new THREE.Vector3(p, 0.012, min), new THREE.Vector3(p, 0.012, max));
  }
  const grid = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0xd7d0c2 })
  );
  scene.add(grid);
}

function makeMarker(color, shape) {
  const geometry = shape === "box"
    ? new THREE.BoxGeometry(cellScale * 0.58, 0.08, cellScale * 0.58)
    : new THREE.CylinderGeometry(cellScale * 0.28, cellScale * 0.28, 0.08, 32);
  const marker = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ color, roughness: 0.65, metalness: 0.02 })
  );
  scene.add(marker);
  return marker;
}

function makeAgent(id, color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.72 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.5 });

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.35, 32, 18), material);
  body.scale.set(1.25, 0.58, 0.78);
  body.position.set(0, 0.3, 0);
  group.add(body);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 14), material);
  head.position.set(0.43, 0.32, 0);
  group.add(head);

  const earA = new THREE.Mesh(new THREE.SphereGeometry(0.095, 16, 10), material);
  earA.position.set(0.46, 0.48, 0.16);
  group.add(earA);

  const earB = earA.clone();
  earB.position.z = -0.16;
  group.add(earB);

  const eyeA = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 8), dark);
  eyeA.position.set(0.62, 0.37, 0.075);
  group.add(eyeA);

  const eyeB = eyeA.clone();
  eyeB.position.z = -0.075;
  group.add(eyeB);

  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), dark);
  nose.position.set(0.66, 0.31, 0);
  group.add(nose);

  const tailPoints = [
    new THREE.Vector3(-0.42, 0.26, 0),
    new THREE.Vector3(-0.74, 0.22, 0.03),
    new THREE.Vector3(-0.92, 0.18, 0),
  ];
  const tail = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(tailPoints),
    new THREE.LineBasicMaterial({ color, linewidth: 2 })
  );
  group.add(tail);

  scene.add(group);
  return {
    id,
    group,
    position: [0, 0],
    velocity: [0, 0],
    cell: [0, 0],
    speed: 3.1 + random() * 0.35,
    temperature: 0.02 + random() * 0.025,
    decisionPhase: random() * 0.08,
  };
}

function randomEdgeCell() {
  const side = Math.floor(random() * 4);
  const value = 1 + Math.floor(random() * (size - 2));
  if (side === 0) {
    return [value, 0];
  }
  if (side === 1) {
    return [size - 1, value];
  }
  if (side === 2) {
    return [value, size - 1];
  }
  return [0, value];
}

function resetTrial() {
  active = false;
  stepCount = 0;
  collectedFrames = 0;
  decisionClock = 0;
  targetId = Math.floor(random() * targets.length);

  agents.forEach((agent) => {
    const start = randomEdgeCell();
    agent.position = [start[0], start[1]];
    agent.velocity = [0, 0];
    agent.cell = start;
    syncAgentMesh(agent);
  });
  syncMarkers();
}

function stateId() {
  const cell0 = agents[0].cell[1] * size + agents[0].cell[0];
  const cell1 = agents[1].cell[1] * size + agents[1].cell[0];
  const targetCode = active ? targetId + 1 : 0;
  return targetCode * size * size * size * size + cell0 + cell1 * size * size;
}

function qValue(agentId, state, action) {
  return q[(agentId * qShape[1] + state) * qShape[2] + action];
}

function chooseAction(agent) {
  const state = stateId();
  const values = moves.map((_, action) => qValue(agent.id, state, action));
  const maxValue = Math.max(...values);
  const weights = values.map((value) => Math.exp((value - maxValue) / agent.temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  let draw = random() * total;
  for (let action = 0; action < weights.length; action += 1) {
    draw -= weights[action];
    if (draw <= 0) {
      return action;
    }
  }
  return 0;
}

function updatePolicy(dt) {
  decisionClock += dt;
  if (decisionClock < 0.18) {
    return;
  }
  decisionClock = 0;
  agents.forEach((agent) => {
    if (draggedAgent === agent) {
      return;
    }
    agent.cell = [
      Math.round(clamp(agent.position[0], 0, size - 1)),
      Math.round(clamp(agent.position[1], 0, size - 1)),
    ];
    const action = chooseAction(agent);
    const move = moves[action];
    const nextCell = [
      clamp(agent.cell[0] + move[0], 0, size - 1),
      clamp(agent.cell[1] + move[1], 0, size - 1),
    ];
    agent.velocity[0] += (nextCell[0] - agent.position[0]) * agent.speed;
    agent.velocity[1] += (nextCell[1] - agent.position[1]) * agent.speed;
  });
}

function updatePhysics(dt) {
  agents.forEach((agent) => {
    if (draggedAgent === agent) {
      return;
    }
    if (pointerDown && pointerGrid) {
      const dx = agent.position[0] - pointerGrid[0];
      const dy = agent.position[1] - pointerGrid[1];
      const dist = Math.hypot(dx, dy);
      if (dist < 1.25 && dist > 0.001) {
        const force = (1.25 - dist) * 6.5 * dt;
        agent.velocity[0] += (dx / dist) * force;
        agent.velocity[1] += (dy / dist) * force;
      }
    }
    agent.velocity[0] *= 0.84;
    agent.velocity[1] *= 0.84;
    agent.position[0] = clamp(agent.position[0] + agent.velocity[0] * dt, 0, size - 1);
    agent.position[1] = clamp(agent.position[1] + agent.velocity[1] * dt, 0, size - 1);
    agent.cell = [
      Math.round(clamp(agent.position[0], 0, size - 1)),
      Math.round(clamp(agent.position[1], 0, size - 1)),
    ];
    syncAgentMesh(agent);
  });
}

function updateEnvState() {
  stepCount += 1;
  const bothCenter = agents.every((agent) => agent.cell[0] === center[0] && agent.cell[1] === center[1]);
  if (!active && bothCenter) {
    active = true;
  }

  const targetCells = targets[targetId];
  const atFirst = agents.every((agent) => agent.cell[0] === targetCells[0][0] && agent.cell[1] === targetCells[0][1]);
  const atSecond = agents.every((agent) => agent.cell[0] === targetCells[1][0] && agent.cell[1] === targetCells[1][1]);
  if (active && (atFirst || atSecond)) {
    collectedFrames += 1;
  } else {
    collectedFrames = 0;
  }

  if (collectedFrames > 24 || stepCount > env.max_steps * 16) {
    resetTrial();
  }
  syncMarkers();
}

function syncMarkers() {
  centerMarker.visible = !active;
  centerMarker.position.copy(gridToWorld(center));
  centerMarker.position.y = 0.055;

  const targetCells = targets[targetId];
  targetMarkers.forEach((marker, index) => {
    marker.visible = active;
    marker.position.copy(gridToWorld(targetCells[index]));
    marker.position.y = 0.055;
  });
}

function syncAgentMesh(agent) {
  const world = gridToWorld(agent.position);
  agent.group.position.set(world.x, 0, world.z);
  const angle = Math.atan2(agent.velocity[1], agent.velocity[0]);
  if (Math.hypot(agent.velocity[0], agent.velocity[1]) > 0.02) {
    agent.group.rotation.y = -angle;
  }
}

function pointerToGrid(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointerNdc, camera);
  raycaster.ray.intersectPlane(groundPlane, groundHit);
  return worldToGrid(groundHit);
}

function nearestAgent(point) {
  return agents
    .map((agent) => ({
      agent,
      distance: Math.hypot(agent.position[0] - point[0], agent.position[1] - point[1]),
    }))
    .sort((a, b) => a.distance - b.distance)[0];
}

function onPointerDown(event) {
  pointerDown = true;
  pointerGrid = pointerToGrid(event);
  const nearest = nearestAgent(pointerGrid);
  if (nearest.distance < 0.85) {
    draggedAgent = nearest.agent;
    renderer.domElement.setPointerCapture(event.pointerId);
  }
}

function onPointerMove(event) {
  pointerGrid = pointerToGrid(event);
  if (draggedAgent) {
    draggedAgent.velocity = [
      (pointerGrid[0] - draggedAgent.position[0]) * 8,
      (pointerGrid[1] - draggedAgent.position[1]) * 8,
    ];
    draggedAgent.position = [pointerGrid[0], pointerGrid[1]];
    draggedAgent.cell = [
      Math.round(clamp(pointerGrid[0], 0, size - 1)),
      Math.round(clamp(pointerGrid[1], 0, size - 1)),
    ];
    syncAgentMesh(draggedAgent);
  }
}

function onPointerUp() {
  pointerDown = false;
  pointerGrid = null;
  draggedAgent = null;
}

function resize() {
  const width = container.clientWidth;
  const height = container.clientHeight;
  const aspect = width / height;
  const view = 5.7;
  camera.left = -view * aspect;
  camera.right = view * aspect;
  camera.top = view;
  camera.bottom = -view;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

let lastTime = performance.now();
function animate(time) {
  const dt = Math.min(0.04, (time - lastTime) / 1000);
  lastTime = time;
  updatePolicy(dt);
  updatePhysics(dt);
  updateEnvState();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

renderer.domElement.addEventListener("pointerdown", onPointerDown);
renderer.domElement.addEventListener("pointermove", onPointerMove);
renderer.domElement.addEventListener("pointerup", onPointerUp);
renderer.domElement.addEventListener("pointercancel", onPointerUp);
window.addEventListener("resize", resize);
