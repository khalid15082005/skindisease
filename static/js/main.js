// Main JavaScript for Skin Disease AI

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File input validation
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file size (16MB limit)
                if (file.size > 16 * 1024 * 1024) {
                    alert('File size must be less than 16MB');
                    e.target.value = '';
                    return;
                }
                
                // Check file type
                const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
                if (!allowedTypes.includes(file.type)) {
                    alert('Please upload only PNG, JPG, or JPEG files');
                    e.target.value = '';
                    return;
                }
                
                // Preview image
                previewImage(file);
            }
        });
    }

    // Form submission handling
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleImageUpload();
        });
    }

    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
});

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImg = document.getElementById('previewImg');
        const imagePreview = document.getElementById('imagePreview');
        
        if (previewImg && imagePreview) {
            previewImg.src = e.target.result;
            imagePreview.style.display = 'block';
            imagePreview.classList.add('fade-in');
        }
    };
    reader.readAsDataURL(file);
}

function handleImageUpload() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select a file first.', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    const diagnoseBtn = document.getElementById('diagnoseBtn');
    const originalText = diagnoseBtn.innerHTML;
    diagnoseBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    diagnoseBtn.disabled = true;
    
    // Make AJAX request
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            showAlert('Error: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('An error occurred while processing the image. Please try again.', 'danger');
    })
    .finally(() => {
        // Reset button
        diagnoseBtn.innerHTML = originalText;
        diagnoseBtn.disabled = false;
    });
}

function displayResults(data) {
    const container = document.getElementById('resultsContainer');
    
    const resultsHtml = `
        <div class="fade-in">
            <div class="text-center mb-4">
                <h4 class="text-primary">
                    <i class="fas fa-bullseye me-2"></i>
                    Most Likely Diagnosis
                </h4>
                <h3 class="text-success fw-bold">${data.top_prediction.class}</h3>
                <h4 class="text-warning">${data.top_prediction.confidence.toFixed(1)}% Confidence</h4>
            </div>
            
            <div class="mb-4">
                <h6 class="mb-3">
                    <i class="fas fa-chart-bar me-2"></i>
                    All Predictions
                </h6>
                ${data.results.map((result, index) => `
                    <div class="mb-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="fw-bold ${result.is_top ? 'text-primary' : 'text-muted'}">
                                ${result.class}
                            </span>
                            <span class="badge ${result.is_top ? 'bg-primary' : 'bg-secondary'}">
                                ${result.confidence.toFixed(1)}%
                            </span>
                        </div>
                        <div class="progress" style="height: 25px;">
                            <div class="progress-bar ${result.is_top ? 'bg-primary' : 'bg-secondary'}" 
                                 style="width: ${result.confidence}%"
                                 role="progressbar">
                                <span class="text-white fw-bold">${result.confidence.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <div class="alert alert-info">
                <h6><i class="fas fa-lightbulb me-2"></i>Interpretation</h6>
                <p class="mb-0">${data.interpretation}</p>
            </div>
        </div>
    `;
    
    container.innerHTML = resultsHtml;
    container.scrollIntoView({ behavior: 'smooth' });
}

function showAlert(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the main container
    const mainContainer = document.querySelector('main.container');
    if (mainContainer) {
        mainContainer.insertBefore(alertDiv, mainContainer.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateImageFile(file) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    
    if (file.size > maxSize) {
        return { valid: false, message: 'File size must be less than 16MB' };
    }
    
    if (!allowedTypes.includes(file.type)) {
        return { valid: false, message: 'Please upload only PNG, JPG, or JPEG files' };
    }
    
    return { valid: true };
}
