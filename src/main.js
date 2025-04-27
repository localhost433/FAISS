// frontend/src/main.js

import { searchPlaces } from './api.js';
import { clearResults, displayResults } from './ui.js';

document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('searchInput');
  const searchButton = document.getElementById('searchButton');

  searchButton.addEventListener('click', async () => {
    const query = searchInput.value.trim();
    if (!query) return;

    clearResults();
    const results = await searchPlaces(query);
    displayResults(results);
  });
});
