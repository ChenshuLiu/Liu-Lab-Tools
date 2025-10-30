const searchBox = document.getElementById("searchBox");
const categoryFilter = document.getElementById("categoryFilter");
const cards = document.querySelectorAll(".card");

function filterCards() {
  const searchTerm = searchBox.value.toLowerCase();
  const category = categoryFilter.value;

  cards.forEach(card => {
    const title = card.querySelector("h3").textContent.toLowerCase();
    const desc = card.querySelector("p").textContent.toLowerCase();
    const matchesSearch = title.includes(searchTerm) || desc.includes(searchTerm);
    const matchesCategory = category === "all" || card.dataset.category === category;

    if (matchesSearch && matchesCategory) {
      card.style.display = "block";
    } else {
      card.style.display = "none";
    }
  });
}

searchBox.addEventListener("input", filterCards);
categoryFilter.addEventListener("change", filterCards);
