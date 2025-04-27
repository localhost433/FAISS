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
      div.innerHTML = `
        <h3>${item.name || 'Unnamed Place'}</h3>
        <p>${item.description || 'No description available.'}</p>
        <p class="distance"><strong>Distance:</strong> ${item.distance.toFixed(4)}</p>
      `;
      resultsDiv.appendChild(div);
    });
  }
  