document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    const glassPanel = document.querySelector('.glass-panel');
    const resultsPanel = document.getElementById('results-panel');
    const loading = document.getElementById('loading');
    const tallyContainer = document.getElementById('tally-container');

    // Handle Drag Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    // Handle File Drop
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFile(file);
    });

    // Handle Click to Upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }

        // Show Image Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropZoneContent.classList.add('hidden');
            imagePreview.classList.remove('hidden');
            
            // Expand panel to show results
            glassPanel.classList.add('has-results');
            resultsPanel.classList.remove('hidden');
            
            analyzeImage(file);
        };
        reader.readAsDataURL(file);
    }

    async function analyzeImage(file) {
        // Reset UI
        tallyContainer.innerHTML = '';
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', file);

        try {
            // Note: Since frontend is served by FastAPI on root, we can hit /predict directly
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data.tallies);
        } catch (error) {
            console.error("Error analyzing image:", error);
            tallyContainer.innerHTML = `<p class="no-results" style="color: #ef4444;">Error analyzing image. Ensure backend is running.</p>`;
        } finally {
            loading.classList.add('hidden');
        }
    }

    function displayResults(tallies) {
        if (!tallies || Object.keys(tallies).length === 0) {
            tallyContainer.innerHTML = `<p class="no-results">No clothing items detected with high confidence.</p>`;
            return;
        }

        let delay = 0;
        for (const [item, count] of Object.entries(tallies)) {
            const div = document.createElement('div');
            div.className = 'tally-item';
            div.style.animationDelay = `${delay}s`;
            
            div.innerHTML = `
                <span class="tally-name">${item}</span>
                <span class="tally-count">${count}</span>
            `;
            
            tallyContainer.appendChild(div);
            delay += 0.1;
        }
    }
});
