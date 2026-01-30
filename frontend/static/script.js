async function predictMessage() {
    const message = document.getElementById('messageInput').value.trim();
    
    if (!message) {
        alert('Please enter a message to analyze');
        return;
    }
    
    showLoadingSpinner(true);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoadingSpinner(false);
    }
}

function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const ensemblePrediction = document.getElementById('ensemblePrediction');
    const ensembleConfidence = document.getElementById('ensembleConfidence');
    const modelResults = document.getElementById('modelResults');
    
    // Display ensemble result
    const predictionClass = result.prediction === 'SPAM' ? 'spam' : 'ham';
    ensemblePrediction.textContent = result.prediction;
    ensemblePrediction.className = `prediction-box ${predictionClass}`;
    
    const confidence = parseFloat(result.confidence * 100);
    const filledPercentage = result.prediction === 'SPAM' 
        ? confidence 
        : (100 - confidence);
    
    ensembleConfidence.innerHTML = `
        <div class="confidence-bar-fill" style="width: ${filledPercentage.toFixed(1)}%">
            ${filledPercentage.toFixed(1)}%
        </div>
    `;
    
    // Display individual model results
    modelResults.innerHTML = '';
    for (const [modelName, prediction] of Object.entries(result.individual_predictions)) {
        const modelCard = createModelCard(modelName, prediction);
        modelResults.appendChild(modelCard);
    }
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createModelCard(modelName, prediction) {
    const card = document.createElement('div');
    card.className = 'model-result';
    
    const predictionClass = prediction.prediction === 'SPAM' ? 'spam' : 'ham';
    const modelTitle = formatModelName(modelName);
    
    const spamPercent = parseFloat(prediction.spam_probability * 100).toFixed(2);
    const hamPercent = parseFloat(prediction.ham_probability * 100).toFixed(2);
    
    card.innerHTML = `
        <h4>${modelTitle}</h4>
        <div class="model-result-prediction ${predictionClass}">
            ${prediction.prediction}
        </div>
        <div class="probability-item">
            <span class="probability-label">Spam:</span>
            <span class="probability-value">${spamPercent}%</span>
        </div>
        <div class="probability-bar-container">
            <div class="probability-bar spam" style="width: ${spamPercent}%"></div>
        </div>
        <div class="probability-item" style="margin-top: 10px;">
            <span class="probability-label">Ham:</span>
            <span class="probability-value">${hamPercent}%</span>
        </div>
        <div class="probability-bar-container">
            <div class="probability-bar ham" style="width: ${hamPercent}%"></div>
        </div>
    `;
    
    return card;
}

function formatModelName(name) {
    const names = {
        'naive_bayes': '🤖 Naive Bayes',
        'svm': '📈 SVM',
        'random_forest': '🌳 Random Forest',
        'xgboost': '⚡ XGBoost'
    };
    return names[name] || name;
}

function clearAll() {
    document.getElementById('messageInput').value = '';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('messageInput').focus();
}

function showLoadingSpinner(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.style.display = 'block';
    } else {
        spinner.style.display = 'none';
    }
}

async function predictMessage() {
    const message = document.getElementById('messageInput').value.trim();
    
    if (!message) {
        alert('Please enter a message to analyze');
        return;
    }
    
    showLoadingSpinner(true);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoadingSpinner(false);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('messageInput').focus();
    
    document.getElementById('messageInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            predictMessage();
        }
    });
});