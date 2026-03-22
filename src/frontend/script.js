document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileCount = document.getElementById('file-count');
    const submitBtn = document.getElementById('submit-btn');
    const jdInput = document.getElementById('jd-input');
    const topKInput = document.getElementById('top-k');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const candidatesContainer = document.getElementById('candidates-container');
    const statsPanel = document.getElementById('stats-panel');

    let selectedFiles = [];

    // --- Drag and Drop Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    });

    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        selectedFiles = Array.from(files);
        fileCount.textContent = `${selectedFiles.length} file(s) selected`;
        fileCount.style.color = '#10b981'; // success green
    }

    // --- API Request Submission ---
    submitBtn.addEventListener('click', async () => {
        const jdText = jdInput.value.trim();
        const topK = parseInt(topKInput.value);

        if (!jdText) {
            alert('Please enter a Job Description');
            return;
        }

        if (selectedFiles.length === 0) {
            alert('Please select at least one resume file');
            return;
        }

        // Prepare FormData
        const formData = new FormData();
        formData.append('job_description', jdText);
        formData.append('top_k', topK);
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        // UI State: Loading
        btnText.textContent = 'Processing...';
        loader.style.display = 'inline-block';
        submitBtn.disabled = true;
        candidatesContainer.innerHTML = '<div class="empty-state"><div class="loader" style="margin: 0 auto 1rem; width: 40px; height: 40px; display: block; filter: invert(1);"></div><p>Calculating semantic similarities and parsing skills...</p></div>';

        try {
            const response = await fetch('/api/v1/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to process resumes');
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error('API Error:', error);
            alert(`Error: ${error.message}`);
            candidatesContainer.innerHTML = `<div class="empty-state"><h3>⚠️ Error Processing</h3><p>${error.message}</p></div>`;
        } finally {
            // Revert UI State
            btnText.textContent = 'Screen Candidates';
            loader.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

    // --- Rendering Results ---
    function renderResults(data) {
        const candidates = data.ranked_candidates || [];
        
        // Update Stats
        document.getElementById('stat-processed').textContent = data.total_resumes_processed;
        document.getElementById('stat-returned').textContent = data.top_k_returned;
        
        if (candidates.length > 0) {
            document.getElementById('stat-best').textContent = `${candidates[0].similarity_score}%`;
        } else {
            document.getElementById('stat-best').textContent = `0%`;
        }
        
        statsPanel.style.display = 'grid';
        candidatesContainer.innerHTML = '';

        if (candidates.length === 0) {
            candidatesContainer.innerHTML = '<div class="empty-state"><h3>No valid results</h3><p>Try uploading different resumes.</p></div>';
            return;
        }

        candidates.forEach((candidate, index) => {
            const delay = index * 0.1; // staggered animation
            
            // Format rank circle
            let rankClass = 'rank-other';
            if (candidate.rank === 1) rankClass = 'rank-1';
            else if (candidate.rank === 2) rankClass = 'rank-2';
            else if (candidate.rank === 3) rankClass = 'rank-3';

            // Format skills
            const skillsHtml = (candidate.skills_matched && candidate.skills_matched.length > 0)
                ? candidate.skills_matched.map(skill => `<span class="skill-tag">${skill}</span>`).join('')
                : '<span class="skill-tag" style="background: rgba(245, 158, 11, 0.1); color: var(--warning-color); border-color: rgba(245, 158, 11, 0.2);">No JD skills matched</span>';

            const cleanFilename = candidate.name.replace(/_/g, ' ').replace('.pdf', '').replace('.docx', '').toUpperCase();

            const card = document.createElement('div');
            card.className = 'candidate-card';
            card.style.animationDelay = `${delay}s`;
            
            card.innerHTML = `
                <div class="rank-badge ${rankClass}">#${candidate.rank}</div>
                
                <div class="candidate-info">
                    <h3>${cleanFilename}</h3>
                    <div class="scores-row">
                        <span class="score-pill">Semantic Match: <span class="score-value">${candidate.semantic_score}%</span></span>
                        <span class="score-pill">Skill Overlap: <span class="score-value">${candidate.skill_score}%</span></span>
                    </div>
                    <div class="skills-container">
                        ${skillsHtml}
                    </div>
                </div>

                <div class="final-score">
                    <div class="score-circle">
                        <span class="val">${candidate.similarity_score}%</span>
                        <span class="lbl">Match</span>
                    </div>
                </div>
            `;

            candidatesContainer.appendChild(card);
        });
    }
});
