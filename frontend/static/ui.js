// frontend/src/ui.js

const tagEmojiMap = {
  shop: '🛍️',
  cafe: '☕️',
  park: '🌳',
  restaurant: '🍽️',
  museum: '🏛️',
  beach: '🏖️',
  bar: '🍻',
  hotel: '🏨',
  gallery: '🖼️',
  theater: '🎭',
  hiking: '🥾',
  bookstore: '📚',
  vintage: '🧥',
  dessert: '🍰',
  bakery: '🥐',
  nature: '🌲',
  night_club: '🌃',
  health: '🏥',
  culture: '🎨',
  book_store: '📖',
};

export function clearResults() {
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = '';
}

export function displayResults(results) {
  const resultsDiv = document.getElementById('results');

  if (!resultsObj.results || resultsObj.results.length === 0) {
    const message = resultsObj.message || "No results found.";
    resultsDiv.innerHTML = `<p>${message}</p>`;
    return;
  }

  results.forEach(item => {
    const div = document.createElement('div');
    div.className = 'result';

    let tagsHTML = '';
    if (item.tags) {
      const tags = item.tags
        .replace(/[{}]/g, '')
        .split(',')
        .map(tag => tag.trim())
        .filter(tag => tag.length > 0);

      tagsHTML = tags.map(tag => {
        const cleanTag = tag.replace(/_/g, ' ').toLowerCase(); // Replace underscores and lowercase
        const emoji = tagEmojiMap[cleanTag] || '🏷️';  // Default emoji if not found
        const displayTag = cleanTag.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '); // Title Case
        return `<span class="tag">${emoji} ${displayTag}</span>`;
      }).join(' ');
    }

    div.innerHTML = `
          <h3>${item.name || 'Unnamed Place'} ${item.emojis || ''}</h3>
          <p>Neighborhood: ${item.neighborhood || 'Unknown'}</p>
          <div class="tags">${tagsHTML}</div>
          <p>Description: ${item.short_description || item.description || 'No description available.'}</p>
          <p class="distance">L2 Distance: ${item.distance.toFixed(4)}</p>
      `;
    resultsDiv.appendChild(div);
  });
}
