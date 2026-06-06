const canvas = document.getElementById("world-canvas");
const ctx = canvas.getContext("2d");
const response = await fetch("./data/coop_grid_q_learning.json");
const data = await response.json();

const env = data.env;
const q = data.policy.q;
const qShape = data.policy.q_shape;
const size = env.size;
const moves = env.moves;
const targets = env.targets;
const center = env.center;
const random = mulberry32(Math.floor(performance.timeOrigin));

const agents = [
  makeAgent(0, "#f2f2f2"),
  makeAgent(1, "#a9a9a9"),
];

let board = { x: 0, y: 0, size: 1, cell: 1 };
let active = false;
let targetId = 0;
let stepCount = 0;
let collectedFrames = 0;
let decisionClock = 0;
let draggedAgent = null;
let pointerGrid = null;
let pointerDown = false;
let lastTime = performance.now();

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

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function makeAgent(id, stroke) {
  return {
    id,
    stroke,
    position: [0, 0],
    velocity: [0, 0],
    cell: [0, 0],
    speed: 3.1 + random() * 0.35,
    temperature: 0.02 + random() * 0.025,
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
  });
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
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawMarkers();
  agents.forEach(drawMouse);
}

function drawGrid() {
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "#dcdcdc";
  ctx.lineWidth = Math.max(2, board.size * 0.0032);
  ctx.strokeRect(board.x, board.y, board.size, board.size);

  ctx.strokeStyle = "rgba(220, 220, 220, 0.28)";
  ctx.lineWidth = Math.max(1, board.size * 0.0015);
  for (let index = 1; index < size; index += 1) {
    const p = board.x + index * board.cell;
    ctx.beginPath();
    ctx.moveTo(p, board.y);
    ctx.lineTo(p, board.y + board.size);
    ctx.stroke();
    const q = board.y + index * board.cell;
    ctx.beginPath();
    ctx.moveTo(board.x, q);
    ctx.lineTo(board.x + board.size, q);
    ctx.stroke();
  }
  ctx.restore();
}

function drawMarkers() {
  ctx.save();
  ctx.lineWidth = Math.max(2, board.size * 0.003);
  ctx.strokeStyle = active ? "rgba(255, 255, 255, 0.38)" : "rgba(255, 255, 255, 0.72)";
  drawSquare(center, 0.24);

  if (active) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.72)";
    targets[targetId].forEach((target) => {
      drawCircle(target, 0.27);
    });
  }
  ctx.restore();
}

function drawSquare(point, radius) {
  const [x, y] = gridToCanvas(point);
  const r = radius * board.cell;
  ctx.strokeRect(x - r, y - r, r * 2, r * 2);
}

function drawCircle(point, radius) {
  const [x, y] = gridToCanvas(point);
  ctx.beginPath();
  ctx.arc(x, y, radius * board.cell, 0, Math.PI * 2);
  ctx.stroke();
}

function drawMouse(agent) {
  const [x, y] = gridToCanvas(agent.position);
  const speed = Math.hypot(agent.velocity[0], agent.velocity[1]);
  const angle = speed > 0.02 ? Math.atan2(agent.velocity[1], agent.velocity[0]) : 0;
  const scale = board.cell * 0.32;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.strokeStyle = agent.stroke;
  ctx.lineWidth = Math.max(2, board.size * 0.0038);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.beginPath();
  ctx.ellipse(0, 0, scale * 0.72, scale * 1.06, Math.PI / 2, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(scale * 0.72, 0, scale * 0.47, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(scale * 0.8, -scale * 0.34, scale * 0.24, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(scale * 0.8, scale * 0.34, scale * 0.24, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(scale * 1.04, -scale * 0.15, scale * 0.045, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(scale * 1.04, scale * 0.15, scale * 0.045, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(-scale * 0.68, 0);
  ctx.lineTo(-scale * 1.58, 0);
  ctx.stroke();

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
  const margin = Math.min(canvas.width, canvas.height) * 0.055;
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
