document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.getElementById('file-upload');
    const uploadZone = document.getElementById('upload-zone');
    const resultsZone = document.getElementById('results-zone');
    const scannedImg = document.getElementById('scanned-img');
    const resetBtn = document.getElementById('reset-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContent = document.getElementById('results-content');
    const imageContainer = document.getElementById('image-overlay-container');
    
    // Drag and drop support
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.background = 'rgba(255,255,255,0.05)';
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.background = 'transparent';
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.background = 'transparent';
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    resetBtn.addEventListener('click', () => {
        resultsZone.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        fileUpload.value = '';
        
        // Remove old bounding boxes
        const boxes = imageContainer.querySelectorAll('.bounding-box');
        boxes.forEach(b => b.remove());
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            const base64Image = e.target.result;
            
            // Switch UI
            uploadZone.classList.add('hidden');
            resultsZone.classList.remove('hidden');
            scannedImg.src = base64Image;
            
            // Reset results
            resultsContent.innerHTML = '';
            loadingSpinner.classList.remove('hidden');
            
            // Remove old boxes
            const boxes = imageContainer.querySelectorAll('.bounding-box');
            boxes.forEach(b => b.remove());
            
            await scanImage(file);
        };
        reader.readAsDataURL(file);
    }
    
    async function scanImage(file) {
        try {
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('API request failed');
            }
            
            const data = await response.json();
            loadingSpinner.classList.add('hidden');
            
            renderResults(data);
            
            if (scannedImg.complete) {
                drawBoxes(data.people);
            } else {
                scannedImg.onload = () => drawBoxes(data.people);
            }
            
        } catch (error) {
            loadingSpinner.classList.add('hidden');
            resultsContent.innerHTML = `<div class="person-result-card" style="border-left-color: #ef4444;">
                <h4 style="color: #ef4444;">Scan Failed</h4>
                <p>Could not reach the AI Server. Make sure the backend is running.</p>
            </div>`;
            console.error(error);
        }
    }
    
    function renderResults(data) {
        if (!data.people || data.people.length === 0) {
            resultsContent.innerHTML = `<div class="person-result-card">
                <h4>No items detected</h4>
            </div>`;
            return;
        }
        
        let html = '';
        data.people.forEach((person, idx) => {
            const title = person.is_person ? `Person ${idx + 1}` : 'Detected Items';
            html += `<div class="person-result-card">
                <h4>${title}</h4>
                <ul>`;
            
            for (const [item, count] of Object.entries(person.clothes)) {
                html += `<li><span class="item-count">${count}x</span> ${item}</li>`;
            }
            
            if (Object.keys(person.clothes).length === 0) {
                html += `<li>No clothing detected</li>`;
            }
            
            html += `</ul></div>`;
        });
        
        resultsContent.innerHTML = html;
    }
    
    function drawBoxes(peopleData) {
        const naturalW = scannedImg.naturalWidth;
        const naturalH = scannedImg.naturalHeight;
        
        peopleData.forEach((person, idx) => {
            if (!person.is_person || !person.box) return;
            
            const box = person.box;
            
            const leftPct = (box[0] / naturalW) * 100;
            const topPct = (box[1] / naturalH) * 100;
            const widthPct = ((box[2] - box[0]) / naturalW) * 100;
            const heightPct = ((box[3] - box[1]) / naturalH) * 100;
            
            const boxEl = document.createElement('div');
            boxEl.className = 'bounding-box';
            boxEl.style.left = `${leftPct}%`;
            boxEl.style.top = `${topPct}%`;
            boxEl.style.width = `${widthPct}%`;
            boxEl.style.height = `${heightPct}%`;
            
            const label = document.createElement('div');
            label.className = 'person-label';
            label.innerText = `Person ${idx + 1}`;
            boxEl.appendChild(label);
            
            imageContainer.appendChild(boxEl);
        });
    }
});
