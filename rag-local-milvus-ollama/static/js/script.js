document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');

    // Configure marked options
    marked.setOptions({
        breaks: true,         // Add <br> on single line breaks
        gfm: true,            // GitHub Flavored Markdown
        headerIds: false,     // Don't add ids to headers
        highlight: function(code, language) {
            if (language && hljs.getLanguage(language)) {
                return hljs.highlight(code, { language }).value;
            }
            return hljs.highlightAuto(code).value;
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        userInput.value = '';
        
        // Add loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message system loading';
        loadingDiv.innerHTML = '<div class="spinner"></div><div>Thinking...</div>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        try {
            // Send message to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message
                })
            });
            
            const data = await response.json();
            
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);
            
            // Add system response with markdown formatting
            addMessage(data.response, 'system', true);
            
        } catch (error) {
            console.error('Error:', error);
            chatMessages.removeChild(loadingDiv);
            addMessage('Sorry, something went wrong. Please try again.', 'system');
        }
    });

    function addMessage(text, sender, isMarkdown = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        // Create container for markdown content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'markdown-content';
        
        if (isMarkdown && sender === 'system') {
            // Parse and render markdown for system messages
            contentDiv.innerHTML = marked.parse(text);
            
            // Initialize syntax highlighting for code blocks
            setTimeout(() => {
                messageDiv.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }, 0);
        } else {
            // For user messages or when not using markdown
            contentDiv.textContent = text;
        }
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
