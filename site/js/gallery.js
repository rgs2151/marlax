const canvas = document.getElementById("world-canvas");
const ctx = canvas.getContext("2d");
const response = await fetch("./data/coop_grid_q_learning.json");
const data = await response.json();
const artwork = await Promise.all([
  "boards/circuit.svg",
  "markers/beacon.svg",
  "markers/portal.svg",
  "animals/mice/orbit.svg",
  "animals/mice/dart.svg",
].map(async (path) => {
  const image = new Image();
  image.src = `./assets/${path}`;
  await image.decode();
  return image;
}));


const env = data.env;
const q = data.policy.q;
const qShape = data.policy.q_shape;
const size = env.size;
const moves = env.moves;
const targets = env.targets;
const center = env.center;
const scenarios = data.showcase.scenarios;

const agents = [
  makeAgent(0, artwork[3]),
  makeAgent(1, artwork[4]),
];

let board = { x: 0, y: 0, size: 1, cell: 1 };
let active = false;
let targetId = 0;
let scenarioIndex = 0;
let collectedFrames = 0;
let decisionClock = 0;
let draggedAgent = null;
let pointerGrid = null;
let pointerDown = false;
let lastTime = performance.now();

loadScenario(0);
resize();
requestAnimationFrame(animate);

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function makeAgent(id, image) {
  return {
    id,
    image,
    heading: 0,
    position: [0, 0],
    velocity: [0, 0],
    cell: [0, 0],
    speed: 3.2,
  };
}

function loadScenario(index) {
  const scenario = scenarios[index % scenarios.length];
  active = false;
  collectedFrames = 0;
  decisionClock = 0;
  targetId = scenario.target_id;
  scenarioIndex = index % scenarios.length;
  agents.forEach((agent) => {
    const start = scenario.starts[agent.id];
    agent.position = [start[0], start[1]];
    agent.velocity = [0, 0];
    agent.cell = [start[0], start[1]];
  });
}

function advanceScenario() {
  loadScenario(scenarioIndex + 1);
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
  let bestAction = 0;
  let bestValue = qValue(agent.id, state, 0);
  for (let action = 1; action < moves.length; action += 1) {
    const value = qValue(agent.id, state, action);
    if (value > bestValue) {
      bestAction = action;
      bestValue = value;
    }
  }
  return bestAction;
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
  });
}

function updateEnvState() {
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

  if (collectedFrames > 24) {
    advanceScenario();
  }
}

function gridToCanvas(point) {
  return [
    board.x + (point[0] + 0.5) * board.cell,
    board.y + (point[1] + 0.5) * board.cell,
  ];
}

function pointerToGrid(event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  return [
    clamp((x - board.x) / board.cell - 0.5, 0, size - 1),
    clamp((y - board.y) / board.cell - 0.5, 0, size - 1),
  ];
}

function nearestAgent(point) {
  return agents
    .map((agent) => ({
      agent,
      distance: Math.hypot(agent.position[0] - point[0], agent.position[1] - point[1]),
    }))
    .sort((a, b) => a.distance - b.distance)[0];
}

function draw() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawMarkers();
  agents.forEach(drawMouse);
}

function drawGrid() {
  const scale = board.size / 400;
  ctx.drawImage(artwork[0], board.x - 56 * scale, board.y - 56 * scale, 512 * scale, 512 * scale);
}

function drawMarker(image, point) {
  const [x, y] = gridToCanvas(point);
  const extent = board.cell * 1.35;
  ctx.drawImage(image, x - extent / 2, y - extent / 2, extent, extent);
}

function drawMarkers() {
  drawMarker(artwork[2], center);
  if (active) {
    targets[targetId].forEach((target) => drawMarker(artwork[1], target));
  }
}

function drawMouse(agent) {
  const [x, y] = gridToCanvas(agent.position);
  const speed = Math.hypot(agent.velocity[0], agent.velocity[1]);
  if (speed > 0.02) {
    agent.heading = Math.atan2(agent.velocity[1], agent.velocity[0]) + Math.PI / 2;
  }
  const extent = board.cell;
  ctx.save();
  ctx.translate(x, y);
  if (draggedAgent === agent) {
    ctx.strokeStyle = "#8b0000";
    ctx.lineWidth = Math.max(2, board.cell * 0.025);
    const radius = board.cell * 0.38;
    ctx.strokeRect(-radius, -radius, radius * 2, radius * 2);
  }
  ctx.rotate(agent.heading);
  ctx.drawImage(agent.image, -extent / 2, -extent / 2, extent, extent);
  ctx.restore();
}

function onPointerDown(event) {
  pointerDown = true;
  pointerGrid = pointerToGrid(event);
  const nearest = nearestAgent(pointerGrid);
  if (nearest.distance < 0.85) {
    draggedAgent = nearest.agent;
    canvas.setPointerCapture(event.pointerId);
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
  }
}

function onPointerUp() {
  pointerDown = false;
  pointerGrid = null;
  draggedAgent = null;
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const margin = Math.min(canvas.width, canvas.height) * 0.12;
  const boardSize = Math.min(canvas.width, canvas.height) - margin * 2;
  board = {
    x: (canvas.width - boardSize) / 2,
    y: (canvas.height - boardSize) / 2,
    size: boardSize,
    cell: boardSize / size,
  };
}

function animate(time) {
  const dt = Math.min(0.04, (time - lastTime) / 1000);
  lastTime = time;
  updatePolicy(dt);
  updatePhysics(dt);
  updateEnvState();
  draw();
  requestAnimationFrame(animate);
}

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointercancel", onPointerUp);
window.addEventListener("resize", resize);
