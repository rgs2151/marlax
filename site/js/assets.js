const editionButtons = document.querySelectorAll("[data-edition]");
const collectionName = document.getElementById("collection-name");

for (const button of editionButtons) {
  button.addEventListener("click", () => {
    const edition = button.dataset.edition;
    for (const item of document.querySelectorAll("[data-v84]")) {
      item.setAttribute(item.tagName === "IMG" ? "src" : "href", item.dataset[`v${edition}`]);
    }
    for (const choice of editionButtons) {
      choice.setAttribute("aria-pressed", String(choice === button));
    }
    collectionName.textContent = `Vector / ${edition}`;
  });
}
