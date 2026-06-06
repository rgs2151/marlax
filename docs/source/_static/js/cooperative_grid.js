(function () {
  const canvas = document.getElementById("cooperative-grid-demo");
  if (!canvas) {
    return;
  }

  const ctx = canvas.getContext("2d");
  const gridSize = 11;
  const center = { x: 5, y: 5 };
  const actions = [
    { x: 0, y: 0 },
    { x: 0, y: -1 },
    { x: 0, y: 1 },
    { x: -1, y: 0 },
    { x: 1, y: 0 },
  ];
  const targets = [
    { x: 10, y: 5 },
    { x: 5, y: 10 },
    { x: 0, y: 5 },
    { x: 5, y: 0 },
  ];

  let trial = 0;
  let active = false;
  let collectedFrames = 0;
  let dragged = null;
  let lastPointer = null;
  let pointer = null;
  let agents = [];
  let target = targets[0];
  let lastTime = performance.now();
  let nudgeTimer = 0;
  let centerVisits = [false, false];
  const baseSeed = Math.floor(performance.timeOrigin);

  function rng(seed) {
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

  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function sampleNormal(random) {
    const u = Math.max(random(), 1e-6);
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function makeTraits(seed) {
    const random = rng(seed);
    return {
      temperature: 0.14 + random() * 0.16,
      motorNoise: 0.018 + random() * 0.028,
      cohesion: 0.06 + random() * 0.08,
      inertia: 0.04 + random() * 0.12,
      curvature: (random() - 0.5) * 0.44,
      shoveSensitivity: 1.8 + random() * 1.3,
      wanderRate: 0.35 + random() * 0.35,
      actionBias: actions.map(() => (random() - 0.5) * 0.08),
      random,
    };
  }

  function randomEdgeStart(random) {
    const side = Math.floor(random() * 4);
    const value = 1 + random() * 9;
    if (side === 0) {
      return { x: value, y: 0.8 };
    }
    if (side === 1) {
      return { x: 10.2, y: value };
    }
    if (side === 2) {
      return { x: value, y: 10.2 };
    }
    return { x: 0.8, y: value };
  }

  function resetTrial() {
    const random = rng(baseSeed + 1000 + trial * 47);
    target = targets[Math.floor(random() * targets.length)];
    active = false;
    collectedFrames = 0;
    centerVisits = [false, false];
    nudgeTimer = 1 + random() * 1.5;
    agents = [0, 1].map((id) => {
      const traits = makeTraits(baseSeed + 5000 + trial * 101 + id * 7919);
      return {
        id,
        color: id === 0 ? "darkred" : "midnightblue",
        position: randomEdgeStart(random),
        velocity: { x: 0, y: 0 },
        wander: { x: 0, y: 0 },
        trail: [],
        traits,
      };
    });
    trial += 1;
  }

  function goalFor(agent) {
    if (!active) {
      return center;
    }
    return target;
  }

  function qValue(agent, action, partner) {
    const goal = goalFor(agent);
    const next = {
      x: clamp(agent.position.x + action.x * 0.22, 0, gridSize - 1),
      y: clamp(agent.position.y + action.y * 0.22, 0, gridSize - 1),
    };
    const progress = -distance(next, goal) * 1.8;
    const pairDistance = -distance(next, partner.position) * agent.traits.cohesion;
    const inertia = (action.x * agent.velocity.x + action.y * agent.velocity.y) * agent.traits.inertia;
    const toGoal = { x: goal.x - agent.position.x, y: goal.y - agent.position.y };
    const goalDistance = Math.max(0.1, Math.hypot(toGoal.x, toGoal.y));
    const tangent = { x: -toGoal.y / goalDistance, y: toGoal.x / goalDistance };
    const curvature = (action.x * tangent.x + action.y * tangent.y) * agent.traits.curvature;
    return progress + pairDistance + inertia + curvature;
  }

  function chooseAction(agent, partner) {
    const scores = actions.map((action, index) => qValue(agent, action, partner) + agent.traits.actionBias[index]);
    const maxScore = Math.max(...scores);
    const weights = scores.map((score) => Math.exp((score - maxScore) / agent.traits.temperature));
    const total = weights.reduce((a, b) => a + b, 0);
    let draw = agent.traits.random() * total;
    for (let index = 0; index < actions.length; index += 1) {
      draw -= weights[index];
      if (draw <= 0) {
        return actions[index];
      }
    }
    return actions[0];
  }

  function stepAgent(agent, partner, dt) {
    if (dragged === agent.id) {
      return;
    }
    const action = chooseAction(agent, partner);
    const noise = {
      x: sampleNormal(agent.traits.random) * agent.traits.motorNoise,
      y: sampleNormal(agent.traits.random) * agent.traits.motorNoise,
    };
    agent.wander.x = agent.wander.x * 0.94 + sampleNormal(agent.traits.random) * agent.traits.motorNoise * agent.traits.wanderRate;
    agent.wander.y = agent.wander.y * 0.94 + sampleNormal(agent.traits.random) * agent.traits.motorNoise * agent.traits.wanderRate;
    const desired = {
      x: action.x * 2.45 + noise.x + agent.wander.x,
      y: action.y * 2.45 + noise.y + agent.wander.y,
    };
    agent.velocity.x = agent.velocity.x * 0.72 + desired.x * 0.28;
    agent.velocity.y = agent.velocity.y * 0.72 + desired.y * 0.28;
    agent.position.x = clamp(agent.position.x + agent.velocity.x * dt, 0, gridSize - 1);
    agent.position.y = clamp(agent.position.y + agent.velocity.y * dt, 0, gridSize - 1);
  }

  function applyPointerNudge(agent, dt) {
    if (!pointer || dragged === agent.id) {
      return;
    }
    const d = distance(agent.position, pointer);
    if (d > 1.45) {
      return;
    }
    const safeDistance = Math.max(0.15, d);
    const strength = (1.45 - d) * agent.traits.shoveSensitivity * dt * 7;
    agent.velocity.x += ((agent.position.x - pointer.x) / safeDistance) * strength;
    agent.velocity.y += ((agent.position.y - pointer.y) / safeDistance) * strength;
  }

  function update(dt) {
    stepAgent(agents[0], agents[1], dt);
    stepAgent(agents[1], agents[0], dt);
    nudgeTimer -= dt;
    if (nudgeTimer <= 0) {
      const agent = agents[Math.floor(agents[0].traits.random() * agents.length)];
      agent.velocity.x += sampleNormal(agent.traits.random) * 0.75;
      agent.velocity.y += sampleNormal(agent.traits.random) * 0.75;
      nudgeTimer = 1.3 + agents[1].traits.random() * 1.4;
    }
    agents.forEach((agent) => {
      applyPointerNudge(agent, dt);
    });

    agents.forEach((agent) => {
      agent.trail.push({ x: agent.position.x, y: agent.position.y });
      if (agent.trail.length > 42) {
        agent.trail.shift();
      }
    });

    agents.forEach((agent, index) => {
      if (distance(agent.position, center) < 0.72) {
        centerVisits[index] = true;
      }
    });

    if (!active && centerVisits.every((visited) => visited)) {
      active = true;
      agents.forEach((agent) => {
        agent.trail = [];
      });
    }

    if (active && agents.every((agent) => distance(agent.position, target) < 0.55)) {
      collectedFrames += 1;
    } else {
      collectedFrames = 0;
    }

    if (collectedFrames > 34) {
      resetTrial();
    }
  }

  function toCanvas(point, size) {
    const padding = size * 0.055;
    const scale = (size - padding * 2) / (gridSize - 1);
    return {
      x: padding + point.x * scale,
      y: padding + (gridSize - 1 - point.y) * scale,
      scale,
    };
  }

  function drawGrid(size) {
    const padding = size * 0.055;
    const scale = (size - padding * 2) / (gridSize - 1);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = "#dddddd";
    ctx.lineWidth = 1;
    for (let index = 0; index < gridSize; index += 1) {
      const p = padding + index * scale;
      ctx.beginPath();
      ctx.moveTo(p, padding);
      ctx.lineTo(p, size - padding);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padding, p);
      ctx.lineTo(size - padding, p);
      ctx.stroke();
    }
  }

  function drawMarker(point, color, shape) {
    const size = canvas.width;
    const p = toCanvas(point, size);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.78;
    if (shape === "square") {
      ctx.fillRect(p.x - p.scale * 0.22, p.y - p.scale * 0.22, p.scale * 0.44, p.scale * 0.44);
    } else {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.scale * 0.25, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawTrail(agent) {
    if (agent.trail.length < 2) {
      return;
    }
    const size = canvas.width;
    ctx.strokeStyle = agent.color;
    ctx.globalAlpha = 0.24;
    ctx.lineWidth = 2;
    ctx.beginPath();
    agent.trail.forEach((point, index) => {
      const p = toCanvas(point, size);
      if (index === 0) {
        ctx.moveTo(p.x, p.y);
      } else {
        ctx.lineTo(p.x, p.y);
      }
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawMouse(agent) {
    const size = canvas.width;
    const p = toCanvas(agent.position, size);
    const angle = Math.atan2(agent.velocity.y, agent.velocity.x) || 0;
    const bodyLength = p.scale * 0.78;
    const bodyWidth = p.scale * 0.48;
    const headDistance = bodyLength * 0.45;
    const head = {
      x: p.x + Math.cos(-angle) * headDistance,
      y: p.y + Math.sin(-angle) * headDistance,
    };
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(-angle);
    ctx.fillStyle = agent.color;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    ctx.ellipse(0, 0, bodyLength * 0.48, bodyWidth * 0.52, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = agent.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(-bodyLength * 0.38, 0);
    ctx.lineTo(-bodyLength * 0.78, 0);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = agent.color;
    ctx.beginPath();
    ctx.arc(head.x, head.y, bodyWidth * 0.34, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(head.x - bodyWidth * 0.18, head.y - bodyWidth * 0.18, bodyWidth * 0.13, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(head.x - bodyWidth * 0.18, head.y + bodyWidth * 0.18, bodyWidth * 0.13, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(head.x + bodyWidth * 0.12, head.y - bodyWidth * 0.08, bodyWidth * 0.035, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(head.x + bodyWidth * 0.12, head.y + bodyWidth * 0.08, bodyWidth * 0.035, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawHeart() {
    if (collectedFrames <= 0) {
      return;
    }
    const centerPoint = {
      x: (agents[0].position.x + agents[1].position.x) / 2,
      y: (agents[0].position.y + agents[1].position.y) / 2 + 0.45,
    };
    const p = toCanvas(centerPoint, canvas.width);
    const size = p.scale * 0.65;
    ctx.fillStyle = "purple";
    ctx.globalAlpha = Math.min(1, collectedFrames / 10);
    ctx.beginPath();
    for (let index = 0; index <= 80; index += 1) {
      const t = (index / 80) * Math.PI * 2;
      const x = p.x + size * (16 * Math.sin(t) ** 3) / 20;
      const y = p.y - size * (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 20;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  function draw() {
    drawGrid(canvas.width);
    agents.forEach(drawTrail);
    if (active) {
      drawMarker(target, "darkgreen", "circle");
    } else {
      drawMarker(center, "gold", "square");
    }
    agents.forEach(drawMouse);
    drawHeart();
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const size = Math.max(320, Math.floor(rect.width * dpr));
    canvas.width = size;
    canvas.height = size;
  }

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    const size = canvas.width;
    const padding = size * 0.055;
    const scale = (size - padding * 2) / (gridSize - 1);
    const x = ((event.clientX - rect.left) * (size / rect.width) - padding) / scale;
    const y = gridSize - 1 - ((event.clientY - rect.top) * (size / rect.height) - padding) / scale;
    return { x: clamp(x, 0, gridSize - 1), y: clamp(y, 0, gridSize - 1) };
  }

  canvas.addEventListener("pointerdown", (event) => {
    const position = pointerPosition(event);
    pointer = position;
    const nearest = agents
      .map((agent) => ({ agent, d: distance(agent.position, position) }))
      .sort((a, b) => a.d - b.d)[0];
    if (nearest.d < 1.1) {
      dragged = nearest.agent.id;
      lastPointer = position;
      canvas.setPointerCapture(event.pointerId);
    }
  });

  canvas.addEventListener("pointermove", (event) => {
    pointer = pointerPosition(event);
    if (dragged === null) {
      return;
    }
    const agent = agents.find((item) => item.id === dragged);
    agent.velocity = {
      x: (pointer.x - lastPointer.x) * 7,
      y: (pointer.y - lastPointer.y) * 7,
    };
    agent.position = pointer;
    lastPointer = pointer;
  });

  canvas.addEventListener("pointerup", () => {
    dragged = null;
    lastPointer = null;
  });

  canvas.addEventListener("pointerleave", () => {
    pointer = null;
  });

  window.addEventListener("resize", resize);

  function loop(time) {
    const dt = Math.min(0.04, (time - lastTime) / 1000);
    lastTime = time;
    update(dt);
    draw();
    requestAnimationFrame(loop);
  }

  resetTrial();
  resize();
  requestAnimationFrame(loop);
}());
