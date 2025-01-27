console.log('AI-based Day Trading Insights background script loaded');

function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install' || details.reason === 'update') {
    checkDisclaimerStatus();
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'startAnalysis') {
    captureScreen();
  }
});

async function captureScreen() {
  try {
    console.log('Starting screen capture...');
    const requestId = generateRequestId();

    // Using activeTab permission instead of tabs
    const screenshot = await chrome.tabs.captureVisibleTab(null, {format: 'jpeg'});
    console.log('Screenshot captured, length:', screenshot.length);

    const response = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId
      },
      body: JSON.stringify({
        screenshot,
        requestId
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${errorText}`);
    }

    const analysis = await response.json();
    console.log('Analysis received:', analysis);

    // Store request ID
    await chrome.storage.local.set({ lastRequestId: requestId });

    chrome.runtime.sendMessage({
      action: 'analysisComplete',
      data: analysis,
      requestId
    });
  } catch (error) {
    console.error('Capture error:', error);
    chrome.runtime.sendMessage({
      action: 'analysisError',
      error: error.message
    });
  }
}