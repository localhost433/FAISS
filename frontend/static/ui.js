// frontend/src/ui.js

export function clearResults() {
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = '';
}

export function displayResults(results) {
  const resultsDiv = document.getElementById('results');

  if (results.length === 0) {
    resultsDiv.innerHTML = '<p>No results found.</p>';
    return;
  }

  results.forEach(item => {
    const div = document.createElement('div');
    div.className = 'result';

    // Split tags string into array
    let tagsHTML = '';
    if (item.tags) {
      const tags = item.tags.split(',').map(tag => tag.trim());
      tagsHTML = tags.map(tag => `<span class="tag">${tag}</span>`).join(' ');
    }

    div.innerHTML = `
          <h3>${item.name || 'Unnamed Place'} ${item.emojis || ''}</h3>
          <p><strong>Neighborhood:</strong> ${item.neighborhood || 'Unknown'}</p>
          <div class="tags">${tagsHTML}</div>
          <p>${item.short_description || item.description || 'No description available.'}</p>
          <p class="distance"><strong>(L2) Distance:</strong> ${item.distance.toFixed(4)}</p>
      `;
    resultsDiv.appendChild(div);
  });
}
