console.log('AI-based Day Trading Insights content script loaded');

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'captureScreen') {
    // Send acknowledgment back
    sendResponse({status: 'capturing'});
    return true;
  }
});