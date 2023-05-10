const imageForm = document.getElementById('image-form');
const imageInput = document.getElementById('image-input');
const predictButton = document.getElementById('predict-button');
const resultsDiv = document.getElementById('results');
const loadingSpinner = document.getElementById('loading-spinner');

// Hides the results section on page load
resultsDiv.style.display = 'none';

// Handles form submission
imageForm.addEventListener('submit', (e) => {
  e.preventDefault();

  // Clears any previous results
  resultsDiv.innerHTML = '';

  // Displays the loading spinner
  loadingSpinner.style.display = 'block';

  // Creates a new FormData object and appends the image file to it
  const formData = new FormData();
  formData.append('image', imageInput.files[0]);

  // Sends an AJAX request to the server with the image data
  fetch('/predict', {
    method: 'POST',
    body: formData,
    headers: {
      'X-CSRFToken': getCookie('csrftoken'),
    },
  })
    .then((response) => response.json())
    .then((data) => {
      // Hides the loading spinner
      loadingSpinner.style.display = 'none';

      // Displays the results section
      resultsDiv.style.display = 'block';

      // Creates a new image element and sets its source to the predicted image
      const image = document.createElement('img');
      image.src = data.image_url;

      // Creates a new table element and populates it with the predicted person's details
      const table = document.createElement('table');
      const tbody = document.createElement('tbody');

      for (const [key, value] of Object.entries(data.details)) {
        const row = document.createElement('tr');
        const keyCell = document.createElement('td');
        keyCell.textContent = key;
        const valueCell = document.createElement('td');
        valueCell.textContent = value;
        row.appendChild(keyCell);
        row.appendChild(valueCell);
        tbody.appendChild(row);
      }

      table.appendChild(tbody);
      resultsDiv.appendChild(image);
      resultsDiv.appendChild(table);
    })
    .catch((error) => {
      console.error(error);
      loadingSpinner.style.display = 'none';
      resultsDiv.innerHTML = '<p>An error occurred while processing the image.</p>';
    });
});

// Gets the value of a cookie by name
function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
}
