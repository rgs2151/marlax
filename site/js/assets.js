const surfaceToggle = document.getElementById("surface-toggle");
surfaceToggle.hidden = false;
surfaceToggle.addEventListener("click", () => {
  const dark = surfaceToggle.getAttribute("aria-pressed") !== "true";
  document.body.dataset.surface = dark ? "dark" : "light";
  surfaceToggle.setAttribute("aria-pressed", String(dark));
  surfaceToggle.textContent = dark ? "Preview on white" : "Preview on dark";
});
