document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('social-text');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsSection = document.getElementById('results-section');
    
    // Result Elements
    const fuzzyLabel = document.getElementById('fuzzy-label');
    const fuzzyScoreBar = document.getElementById('fuzzy-score-bar');
    const fuzzyScoreText = document.getElementById('fuzzy-score-text');
    const vaderVal = document.getElementById('vader-val');
    const textblobVal = document.getElementById('textblob-val');

    analyzeBtn.addEventListener('click', async () => {
        const textToAnalyze = textInput.value.trim();
        if (!textToAnalyze) {
            alert('Please enter some text to analyze.');
            return;
        }

        // Output Reset & UI State Update
        resultsSection.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        analyzeBtn.disabled = true;

        try {
            // Note: Since main.py runs on 8000 and serves static files from /, 
            // the relative path `/analyze` will hit the FastAPI server perfectly.
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: textToAnalyze })
            });

            if (!response.ok) {
                throw new Error('Server responded with an error');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error during analysis:', error);
            alert('An error occurred while analyzing the text. Make sure the backend server is running.');
        } finally {
            loadingIndicator.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Update breakdowns
        vaderVal.textContent = data.vader.compound.toFixed(2);
        textblobVal.textContent = data.textblob.polarity.toFixed(2);

        // Update Fuzzy Score
        const score = data.fuzzy_result.score;
        const labelText = data.fuzzy_result.label;
        
        fuzzyLabel.textContent = labelText;
        fuzzyScoreText.textContent = `${score}/100`;
        
        // Reset label classes
        fuzzyLabel.className = 'label-badge';
        
        // Add Specific Class based on label string
        let positionPercent = score; // 0 to 100
        let bgPostion = 0; // for gradient background mapping

        if (labelText === 'Very Negative') {
            fuzzyLabel.classList.add('label-very-negative');
            bgPostion = 0;
        } else if (labelText === 'Negative') {
            fuzzyLabel.classList.add('label-negative');
            bgPostion = 25;
        } else if (labelText === 'Neutral') {
            fuzzyLabel.classList.add('label-neutral');
            bgPostion = 50;
        } else if (labelText === 'Positive') {
            fuzzyLabel.classList.add('label-positive');
            bgPostion = 75;
        } else if (labelText === 'Very Positive') {
            fuzzyLabel.classList.add('label-very-positive');
            bgPostion = 100;
        }

        // Show Results Section First (so animations can run properly)
        resultsSection.classList.remove('hidden');

        // Animate width (using setTimeout to allow DOM flow before transition)
        setTimeout(() => {
            fuzzyScoreBar.style.width = `${score}%`;
            // Map the background position so the color matches the sentiment segment of the gradient string
            fuzzyScoreBar.style.backgroundPosition = `${bgPostion}% 0`;
        }, 50);
    }
});
