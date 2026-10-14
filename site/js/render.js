export function createProjection(bounds, width, height) {
  const [left, bottom, right, top] = bounds;
  const scale = Math.min((width - 32) / (right - left), (height - 32) / (top - bottom));
  const cx = (left + right) / 2, cy = (bottom + top) / 2;
  return {
    scale,
    toScreen: ([x, y]) => [width / 2 + (x - cx) * scale, height / 2 - (y - cy) * scale],
    toWorld: ([x, y]) => [cx + (x - width / 2) / scale, cy - (y - height / 2) / scale],
  };
}

export function createRenderer(canvas, assets) {
  const context = canvas.getContext('2d');
  const renderer = {
    scene: null, projection: null, width: 0, height: 0,
    resize() {
      const rect = canvas.getBoundingClientRect();
      this.width = rect.width;
      this.height = rect.height;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    },
    draw(scene, {selected = null, preview = null, landing = null} = {}) {
      this.scene = scene;
      this.projection = createProjection(scene.bounds, this.width, this.height);
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, this.width, this.height);
      for (const sprite of scene.sprites) {
        const asset = assets.get(sprite.asset);
        const position = preview && sprite.agent === selected ? preview : sprite.position;
        const [x, y] = this.projection.toScreen(position);
        const width = sprite.width * this.projection.scale;
        const height = sprite.height * this.projection.scale;
        context.save();
        context.globalAlpha = sprite.opacity;
        context.translate(x, y);
        context.rotate(sprite.heading);
        context.drawImage(asset.image, -width * asset.anchor[0] / asset.viewBox[2],
          -height * asset.anchor[1] / asset.viewBox[3], width, height);
        context.restore();
        if (sprite.agent === selected) {
          context.strokeStyle = '#8b0000';
          context.lineWidth = 2;
          context.strokeRect(x - width * 0.42, y - height * 0.42, width * 0.84, height * 0.84);
        }
      }
      if (landing) {
        const [x, y] = this.projection.toScreen(landing);
        context.strokeStyle = '#8b0000';
        context.lineWidth = 2;
        context.beginPath();
        context.arc(x, y, 5, 0, Math.PI * 2);
        context.stroke();
      }
    },
    pointer(event) {
      const rect = canvas.getBoundingClientRect();
      return [event.clientX - rect.left, event.clientY - rect.top];
    },
    hits(point) {
      return this.scene.sprites.filter(p => p.agent !== undefined).reverse().filter(p => {
        const screen = this.projection.toScreen(p.position);
        const radius = Math.max(22, p.width * this.projection.scale * 0.36);
        return Math.hypot(screen[0] - point[0], screen[1] - point[1]) <= radius;
      });
    },
  };
  renderer.resize();
  return renderer;
}
