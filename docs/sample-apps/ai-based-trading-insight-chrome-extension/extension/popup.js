document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const elements = {
        startButton: document.getElementById('startAnalysis'),
        errorDiv: document.getElementById('error'),
        timestamp: document.getElementById('timestamp'),
        indicators: {
            macd: document.getElementById('macd-value'),
            rsi: document.getElementById('rsi-value'),
            mfi: document.getElementById('mfi-value'),
            stochastic: document.getElementById('stoch-value'),
            volume: document.getElementById('volume-value'),
            currentPrice: document.getElementById('current-price'),
            vwap: document.getElementById('vwap-value')
        },
        levels: {
            support: document.getElementById('supportLevels'),
            resistance: document.getElementById('resistanceLevels')
        },
        trade: {
            signal: document.getElementById('signal-badge'),
            entry: document.getElementById('entry-price'),
            stopLoss: document.getElementById('stop-loss'),
            target: document.getElementById('price-target')
        },
        summary: document.getElementById('analysis-summary')
    };

    // Helper Functions
    function formatPrice(price) {
        if (typeof price === 'string' && price.includes('$')) {
            return price;
        }
        return typeof price === 'number' ? `$${price.toFixed(2)}` : price;
    }

    function highlightUpdate(element) {
        if (!element) return;
        element.classList.add('updated');
        setTimeout(() => element.classList.remove('updated'), 1000);
    }

    function setLoading(isLoading) {
        if (!elements.startButton) return;
        elements.startButton.disabled = isLoading;
        elements.startButton.textContent = isLoading ? 'Analyzing...' : 'Analyze Chart';
        elements.startButton.classList.toggle('loading', isLoading);
    }

    function showError(message) {
        console.error('Error:', message);
        if (elements.errorDiv) {
            elements.errorDiv.textContent = message;
            elements.errorDiv.classList.remove('hidden');
            elements.errorDiv.classList.add('visible');
        }
        setLoading(false);
    }

    function hideError() {
        if (elements.errorDiv) {
            elements.errorDiv.classList.remove('visible');
            elements.errorDiv.classList.add('hidden');
            elements.errorDiv.textContent = '';
        }
    }

    function resetUI() {
        // Reset indicators
        Object.values(elements.indicators).forEach(el => {
            if (el) {
                el.textContent = '-';
                el.classList.remove('updated');
            }
        });

        // Reset levels
        Object.values(elements.levels).forEach(el => {
            if (el) {
                el.innerHTML = '-';
                el.classList.remove('updated');
            }
        });

        // Reset trade setup
        Object.values(elements.trade).forEach(el => {
            if (el) {
                el.textContent = '-';
                el.classList.remove('updated');
                if (el === elements.trade.signal) {
                    el.className = 'signal-badge';
                }
            }
        });

        // Reset summary
        if (elements.summary) {
            elements.summary.textContent = '-';
            elements.summary.classList.remove('updated');
        }

        hideError();
    }

    function updatePriceLevels(data) {
        // Update support levels
        if (elements.levels.support && Array.isArray(data.support_levels)) {
            const supportHtml = data.support_levels
                .map(level => level ? `<div class="updated">${formatPrice(level)}</div>` : '<div>-</div>')
                .join('');
            elements.levels.support.innerHTML = supportHtml;
        }

        // Update resistance levels
        if (elements.levels.resistance && Array.isArray(data.resistance_levels)) {
            const resistanceHtml = data.resistance_levels
                .map(level => level ? `<div class="updated">${formatPrice(level)}</div>` : '<div>-</div>')
                .join('');
            elements.levels.resistance.innerHTML = resistanceHtml;
        }
    }

    function updateIndicators(indicators) {
        if (!indicators) return;

        const updates = {
            macd: indicators.macd,
            rsi: indicators.rsi,
            mfi: indicators.mfi,
            stochastic: indicators.stochastic,
            volume: indicators.volume,
            currentPrice: formatPrice(indicators.current_price),
            vwap: indicators.vwap
        };

        Object.entries(updates).forEach(([key, value]) => {
            const element = elements.indicators[key];
            if (element && value) {
                element.textContent = value;
                highlightUpdate(element);
            }
        });
    }

    function updateTradeSetup(data) {
        if (!data.trade_setup) return;

        // Update trade setup values
        const setup = data.trade_setup;
        if (elements.trade.entry) {
            elements.trade.entry.textContent = setup.entry || '-';
            highlightUpdate(elements.trade.entry);
        }
        if (elements.trade.stopLoss) {
            elements.trade.stopLoss.textContent = setup.stop_loss || '-';
            highlightUpdate(elements.trade.stopLoss);
        }
        if (elements.trade.target) {
            elements.trade.target.textContent = setup.target || '-';
            highlightUpdate(elements.trade.target);
        }

        // Update signal badge
        if (elements.trade.signal && data.signal_type) {
            elements.trade.signal.textContent = data.signal_type;
            elements.trade.signal.className = `signal-badge signal-${data.signal_type.toLowerCase()}`;
            highlightUpdate(elements.trade.signal);
        }
    }

    function updateUI(data) {
        try {
            resetUI();

            if (!data) {
                throw new Error('No analysis data received');
            }

            // Update timestamp
            if (elements.timestamp) {
                elements.timestamp.textContent = new Date().toLocaleTimeString();
                highlightUpdate(elements.timestamp);
            }

            updatePriceLevels(data);
            updateIndicators(data.technical_indicators);
            updateTradeSetup(data);

            // Update summary
            if (elements.summary && data.summary) {
                elements.summary.textContent = data.summary;
                highlightUpdate(elements.summary);
            }

            setLoading(false);

        } catch (error) {
            console.error('Error in updateUI:', error);
            showError(`Error processing analysis: ${error.message}`);
        }
    }

    async function captureAndAnalyze() {
        try {
            setLoading(true);
            resetUI();

            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab) {
                throw new Error('No active tab found');
            }

            const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            await chrome.storage.local.set({ lastRequestId: requestId });

            const screenshot = await chrome.tabs.captureVisibleTab(null, { format: 'png' });

            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    screenshot,
                    requestId
                })
            });

            if (!response.ok) {
                const error = await response.text();
                throw new Error(`Server error: ${error}`);
            }

            const data = await response.json();

            if (data.request_id !== requestId) {
                throw new Error('Response ID mismatch');
            }

            updateUI(data);

        } catch (error) {
            console.error('Analysis error:', error);
            showError(error.message);
            setLoading(false);
        }
    }

    // Event Listeners
    if (elements.startButton) {
        elements.startButton.addEventListener('click', captureAndAnalyze);
    }

    // Initialize UI
    resetUI();
});