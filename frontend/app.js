document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.getElementById('file-upload');
    const uploadZone = document.getElementById('upload-zone');
    const resultsZone = document.getElementById('results-zone');
    const scannedImg = document.getElementById('scanned-img');
    const resetBtn = document.getElementById('reset-btn');
    const cameraAgainBtn = document.getElementById('camera-again-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContent = document.getElementById('results-content');
    const imageContainer = document.getElementById('image-wrapper');

    // Camera UI elements
    const openCameraBtn = document.getElementById('open-camera-btn');
    const cameraModal = document.getElementById('camera-modal');
    const cameraBackdrop = document.getElementById('camera-backdrop');
    const closeCameraBtn = document.getElementById('close-camera-btn');
    const cameraVideo = document.getElementById('camera-video');
    const cameraCanvas = document.getElementById('camera-canvas');
    const captureBtn = document.getElementById('capture-btn');
    const flipCameraBtn = document.getElementById('flip-camera-btn');
    const retakeBtn = document.getElementById('retake-btn');
    const usePhotoBtn = document.getElementById('use-photo-btn');
    const liveControls = document.getElementById('live-controls');
    const previewControls = document.getElementById('preview-controls');
    const viewfinderOverlay = document.getElementById('viewfinder-overlay');
    const cameraErrorContainer = document.getElementById('camera-error-container');
    const cameraErrorMessage = document.getElementById('camera-error-message');
    const cameraFallbackInput = document.getElementById('camera-fallback-input');
    const fallbackCaptureBtn = document.getElementById('fallback-capture-btn');

    // Camera state
    let currentStream = null;
    let currentFacingMode = 'environment';
    let capturedBlob = null;

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

    // Local file browsing
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Native fallback camera input
    cameraFallbackInput?.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            closeCameraModal();
            handleFile(e.target.files[0]);
        }
    });

    fallbackCaptureBtn?.addEventListener('click', () => {
        cameraFallbackInput?.click();
    });

    function resetToUpload() {
        resultsZone.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        fileUpload.value = '';
        if (cameraFallbackInput) cameraFallbackInput.value = '';

        // Remove old bounding boxes
        const boxes = imageContainer.querySelectorAll('.bounding-box');
        boxes.forEach(b => b.remove());
    }

    resetBtn.addEventListener('click', resetToUpload);

    cameraAgainBtn?.addEventListener('click', () => {
        resetToUpload();
        openCameraModal();
    });

    // =========================================================================
    // Camera Controller
    // =========================================================================

    function stopCameraTracks() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (err) {
                    console.error('Error stopping track:', err);
                }
            });
            currentStream = null;
        }
        if (cameraVideo) {
            cameraVideo.srcObject = null;
        }
    }

    async function checkAvailableCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            if (flipCameraBtn) flipCameraBtn.style.visibility = 'hidden';
            return;
        }
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');
            if (flipCameraBtn) {
                flipCameraBtn.style.visibility = videoDevices.length > 1 ? 'visible' : 'hidden';
            }
        } catch (e) {
            console.warn('Could not enumerate media devices:', e);
        }
    }

    function showCameraError(message) {
        cameraVideo.classList.add('hidden');
        viewfinderOverlay.classList.add('hidden');
        liveControls.classList.add('hidden');
        previewControls.classList.add('hidden');
        cameraErrorMessage.textContent = message;
        cameraErrorContainer.classList.remove('hidden');
    }

    async function startCamera(facingMode = currentFacingMode) {
        stopCameraTracks();

        cameraErrorContainer.classList.add('hidden');
        cameraVideo.classList.remove('hidden');
        cameraCanvas.classList.add('hidden');
        viewfinderOverlay.classList.remove('hidden');
        liveControls.classList.remove('hidden');
        previewControls.classList.add('hidden');
        capturedBlob = null;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showCameraError('In-app live camera streaming is not supported on this browser/environment. You can browse or use your device camera directly.');
            return;
        }

        try {
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: facingMode ? { ideal: facingMode } : 'environment',
                        width: { ideal: 1920 },
                        height: { ideal: 1080 }
                    },
                    audio: false
                });
            } catch (err) {
                console.warn('High-res / specific facingMode constraints failed, trying basic video:', err);
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });
            }

            currentStream = stream;
            cameraVideo.srcObject = stream;
            await cameraVideo.play().catch(e => console.log('Video autoplay:', e));

            await checkAvailableCameras();
        } catch (error) {
            console.error('Camera access error:', error);
            let msg = 'Could not access camera. Please allow camera permissions in your browser.';
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                msg = 'Camera permission was denied. Please allow camera access in browser site settings.';
            } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                msg = 'No camera device found on this system.';
            } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                msg = 'Camera is already in use by another application.';
            }
            showCameraError(msg);
        }
    }

    function openCameraModal() {
        cameraModal.classList.remove('hidden');
        startCamera(currentFacingMode);
    }

    function closeCameraModal() {
        stopCameraTracks();
        cameraModal.classList.add('hidden');
        capturedBlob = null;
    }

    async function flipCamera() {
        currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
        await startCamera(currentFacingMode);
    }

    function capturePhoto() {
        if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
            return;
        }

        const width = cameraVideo.videoWidth;
        const height = cameraVideo.videoHeight;

        cameraCanvas.width = width;
        cameraCanvas.height = height;

        const ctx = cameraCanvas.getContext('2d');
        if (currentFacingMode === 'user') {
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(cameraVideo, 0, 0, width, height);

        cameraCanvas.toBlob((blob) => {
            capturedBlob = blob;
        }, 'image/jpeg', 0.95);

        cameraVideo.classList.add('hidden');
        cameraCanvas.classList.remove('hidden');
        viewfinderOverlay.classList.add('hidden');

        liveControls.classList.add('hidden');
        previewControls.classList.remove('hidden');
    }

    function retakePhoto() {
        capturedBlob = null;
        cameraCanvas.classList.add('hidden');
        cameraVideo.classList.remove('hidden');
        viewfinderOverlay.classList.remove('hidden');

        previewControls.classList.add('hidden');
        liveControls.classList.remove('hidden');
    }

    function useCapturedPhoto() {
        if (!capturedBlob) {
            cameraCanvas.toBlob((blob) => {
                if (blob) {
                    processBlobAndProceed(blob);
                }
            }, 'image/jpeg', 0.95);
            return;
        }
        processBlobAndProceed(capturedBlob);
    }

    function processBlobAndProceed(blob) {
        const file = new File([blob], `camera_scan_${Date.now()}.jpg`, { type: 'image/jpeg' });
        closeCameraModal();
        handleFile(file);
    }

    // Camera UI event listeners
    openCameraBtn?.addEventListener('click', openCameraModal);
    closeCameraBtn?.addEventListener('click', closeCameraModal);
    cameraBackdrop?.addEventListener('click', closeCameraModal);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !cameraModal.classList.contains('hidden')) {
            closeCameraModal();
        }
    });

    flipCameraBtn?.addEventListener('click', flipCamera);
    captureBtn?.addEventListener('click', capturePhoto);
    retakeBtn?.addEventListener('click', retakePhoto);
    usePhotoBtn?.addEventListener('click', useCapturedPhoto);

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

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => null);
                throw new Error((errData && (errData.detail || errData.error)) || `Server returned ${response.status}`);
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
                <p>${error.message || 'Could not reach the AI Server. Make sure the backend is running.'}</p>
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
