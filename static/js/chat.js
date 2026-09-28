document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const chatToggleBtn = document.getElementById('chat-toggle-btn');
    const chatWidget = document.getElementById('chat-widget');
    const chatCloseBtn = document.getElementById('chat-close-btn');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const chatSendBtn = document.getElementById('chat-send-btn');

    // State
    let isChatOpen = false;
    let chatHistory = []; // [{role: 'user'|'assistant', content: '...'}]
    let isWaitingForResponse = false;

    // Toggle Chat Widget
    function toggleChat() {
        isChatOpen = !isChatOpen;
        
        if (isChatOpen) {
            // Open animation classes
            chatWidget.classList.remove('opacity-0', 'scale-95', 'pointer-events-none');
            chatWidget.classList.add('opacity-100', 'scale-100');
            // Focus input after animation
            setTimeout(() => chatInput.focus(), 300);
        } else {
            // Close animation classes
            chatWidget.classList.remove('opacity-100', 'scale-100');
            chatWidget.classList.add('opacity-0', 'scale-95', 'pointer-events-none');
        }
    }

    chatToggleBtn.addEventListener('click', toggleChat);
    chatCloseBtn.addEventListener('click', toggleChat);

    // Escape key to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isChatOpen) {
            toggleChat();
        }
    });

    // Handle Form Submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message || isWaitingForResponse) return;
        
        // Add User Message to UI
        addUserMessage(message);
        
        // Clear input and disable form
        chatInput.value = '';
        setFormState(true);
        
        // Add thinking indicator
        const thinkingId = addThinkingIndicator();
        
        try {
            // Stream the response from the server
            await fetchChatResponseStream(message, thinkingId);
        } catch (error) {
            console.error("Chat Error:", error);
            removeElement(thinkingId);
            addErrorMessage("Sorry, I'm having trouble connecting to the server. Please try again later.");
        } finally {
            setFormState(false);
            chatInput.focus();
        }
    });

    async function fetchChatResponseStream(message, thinkingId) {
        // Prepare request body
        const requestBody = {
            message: message,
            history: chatHistory
        };
        
        // Push user message to history
        chatHistory.push({ role: 'user', content: message });
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Remove thinking indicator
        removeElement(thinkingId);
        
        // Create an element for the assistant's response
        const { messageContainer, contentContainer } = createAssistantMessageElement();
        
        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullResponse = "";
        let buffer = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Server-Sent Events parsing with buffer
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep the last incomplete line in the buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        break;
                    }
                    try {
                        const parsedData = JSON.parse(data);
                        fullResponse += parsedData;
                        // Handle embedded forms
                        const formStartIndex = fullResponse.indexOf("<<<FORM_START>>>");
                        const formEndIndex = fullResponse.indexOf("<<<FORM_END>>>");
                        
                        let displayHTML = fullResponse;
                        if (formStartIndex !== -1) {
                            if (formEndIndex !== -1) {
                                const beforeForm = fullResponse.substring(0, formStartIndex);
                                const afterForm = fullResponse.substring(formEndIndex + 14);
                                const jsonStr = fullResponse.substring(formStartIndex + 16, formEndIndex).trim();
                                
                                try {
                                    const formSchema = JSON.parse(jsonStr);
                                    const formHTML = renderFormSchema(formSchema);
                                    displayHTML = marked.parse(beforeForm) + formHTML + marked.parse(afterForm);
                                } catch(e) {
                                    console.error("Invalid form JSON", e);
                                    displayHTML = marked.parse(fullResponse.replace("<<<FORM_START>>>", "").replace("<<<FORM_END>>>", ""));
                                }
                            } else {
                                displayHTML = marked.parse(fullResponse.substring(0, formStartIndex)) + '<div class="mt-4 p-4 border border-slate-200 rounded-xl bg-slate-50 text-sm text-slate-500 flex items-center gap-3"><div class="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin"></div> Generating form...</div>';
                            }
                        } else {
                            displayHTML = marked.parse(fullResponse);
                        }
                        
                        contentContainer.innerHTML = displayHTML;
                        scrollToBottom();
                    } catch (e) {
                        console.error("Error parsing SSE data", e);
                    }
                }
            }
        }
        
        // Add complete response to history
        chatHistory.push({ role: 'assistant', content: fullResponse });
    }

    // --- UI Helpers ---

    function addUserMessage(text) {
        const messageHTML = `
            <div class="flex items-start gap-2.5 justify-end max-w-[90%] self-end">
                <div class="bg-primary text-white p-3.5 rounded-2xl rounded-tr-sm shadow-sm text-[15px] leading-relaxed break-words whitespace-normal break-all sm:break-normal">
                    ${escapeHTML(text)}
                </div>
            </div>
        `;
        chatMessages.insertAdjacentHTML('beforeend', messageHTML);
        scrollToBottom();
    }

    function createAssistantMessageElement() {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'flex items-start gap-2.5 max-w-[90%]';
        
        messageContainer.innerHTML = `
            <div class="w-8 h-8 rounded-full bg-sky-100 flex-shrink-0 flex items-center justify-center text-[15px] shadow-sm border border-sky-200">🤖</div>
            <div class="flex flex-col gap-1 w-full overflow-hidden">
                <span class="text-xs text-slate-500 font-medium ml-1">AI Assistant</span>
                <div class="assistant-content bg-white p-3.5 rounded-2xl rounded-tl-sm shadow-sm border border-slate-100 text-slate-700 text-[15px] leading-relaxed markdown-body w-full break-words whitespace-normal">
                    <span class="inline-block w-1.5 h-4 bg-primary animate-pulse ml-1 align-middle"></span>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(messageContainer);
        scrollToBottom();
        
        const contentContainer = messageContainer.querySelector('.assistant-content');
        return { messageContainer, contentContainer };
    }

    function addThinkingIndicator() {
        const id = 'thinking-' + Date.now();
        const html = `
            <div id="${id}" class="flex items-start gap-2.5 max-w-[90%]">
                <div class="w-8 h-8 rounded-full bg-sky-100 flex-shrink-0 flex items-center justify-center text-sm shadow-sm border border-sky-200">🤖</div>
                <div class="bg-white p-4 rounded-2xl rounded-tl-sm shadow-sm border border-slate-100 flex items-center h-[46px]">
                    <div class="dot-flashing ml-3"></div>
                </div>
            </div>
        `;
        chatMessages.insertAdjacentHTML('beforeend', html);
        scrollToBottom();
        return id;
    }

    function addErrorMessage(text) {
        const html = `
            <div class="flex items-start gap-2.5 max-w-[90%] mx-auto my-2">
                <div class="bg-red-50 text-red-600 p-3 rounded-xl border border-red-100 text-xs text-center w-full shadow-sm">
                    ⚠️ ${text}
                </div>
            </div>
        `;
        chatMessages.insertAdjacentHTML('beforeend', html);
        scrollToBottom();
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function setFormState(disabled) {
        isWaitingForResponse = disabled;
        chatInput.disabled = disabled;
        chatSendBtn.disabled = disabled;
        
        if (disabled) {
            chatInput.classList.add('bg-slate-100', 'cursor-not-allowed');
            chatSendBtn.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            chatInput.classList.remove('bg-slate-100', 'cursor-not-allowed');
            chatSendBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }

    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});

// --- Bot Form Handling ---
function renderFormSchema(schema) {
    let fieldsHtml = '';
    
    (schema.fields || []).forEach(f => {
        let inputHtml = '';
        const requiredAttr = f.required ? 'required' : '';
        const commonClasses = "w-full bg-slate-50 border border-slate-200 text-slate-800 rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all text-[15px]";
        
        if (f.type === 'textarea') {
            inputHtml = `<textarea name="${f.name}" placeholder="${f.placeholder || ''}" class="${commonClasses} resize-none h-24" ${requiredAttr}></textarea>`;
        } else if (f.type === 'select') {
            let optionsHtml = (f.options || []).map(opt => `<option value="${opt}">${opt}</option>`).join('');
            inputHtml = `<select name="${f.name}" class="${commonClasses}" ${requiredAttr}>
                <option value="" disabled selected>Select an option...</option>
                ${optionsHtml}
            </select>`;
        } else if (f.type === 'radio' || f.type === 'checkbox') {
            inputHtml = `<div class="flex flex-col gap-2">`;
            (f.options || []).forEach(opt => {
                inputHtml += `
                <label class="flex items-center gap-2 text-[14px] text-slate-700 cursor-pointer">
                    <input type="${f.type}" name="${f.name}" value="${opt}" class="text-primary focus:ring-primary w-4 h-4" ${requiredAttr}>
                    ${opt}
                </label>`;
            });
            inputHtml += `</div>`;
        } else {
            inputHtml = `<input type="${f.type || 'text'}" name="${f.name}" placeholder="${f.placeholder || ''}" class="${commonClasses}" ${requiredAttr}>`;
        }

        fieldsHtml += `
            <div class="mb-4">
                <label class="block text-xs font-bold text-slate-600 uppercase tracking-wider mb-1.5">${f.label} ${f.required ? '<span class="text-red-500">*</span>' : ''}</label>
                ${inputHtml}
            </div>
        `;
    });

    return `
        <div class="bg-white border border-slate-200 rounded-2xl shadow-md my-4 overflow-hidden w-full max-w-sm" data-form-id="${schema.form_id}">
            <div class="bg-slate-50 border-b border-slate-200 px-5 py-3 flex items-center gap-2">
                <span class="text-xl">📋</span>
                <div>
                    <h4 class="font-bold text-slate-800 text-[15px] leading-tight">${schema.title || 'Form'}</h4>
                    ${schema.description ? `<p class="text-[12px] text-slate-500 mt-0.5">${schema.description}</p>` : ''}
                </div>
            </div>
            <form class="p-5" onsubmit="window.handleBotFormSubmit(event, '${schema.form_id}')">
                ${fieldsHtml}
                <div class="mt-6">
                    <button type="submit" class="w-full bg-primary hover:bg-sky-600 text-white font-medium py-3 rounded-xl transition-colors shadow-sm flex items-center justify-center gap-2 form-submit-btn">
                        <span>${schema.submit_label || 'Submit'}</span>
                    </button>
                </div>
                <div class="form-feedback hidden mt-4 text-sm text-center p-3 rounded-xl"></div>
            </form>
        </div>
    `;
}

window.handleBotFormSubmit = async function(event, formId) {
    event.preventDefault();
    const form = event.target;
    const submitBtn = form.querySelector('.form-submit-btn');
    const feedbackEl = form.querySelector('.form-feedback');
    
    // Collect data
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // UI Loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<div class="w-5 h-5 rounded-full border-2 border-white border-t-transparent animate-spin"></div><span>Submitting...</span>';
    feedbackEl.classList.add('hidden');
    
    try {
        const response = await fetch('/api/forms/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ form_id: formId, data: data })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to submit form');
        }
        
        // Success UI
        form.innerHTML = `
            <div class="text-center py-6">
                <div class="w-16 h-16 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-3xl mx-auto mb-4">✓</div>
                <h4 class="font-bold text-slate-800 text-lg mb-2">Submitted Successfully</h4>
                <p class="text-slate-600 text-sm">Thanks! Our team will get back to you shortly.</p>
            </div>
        `;
        
    } catch (error) {
        // Error UI
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Try Again</span>';
        feedbackEl.textContent = '⚠️ ' + error.message;
        feedbackEl.className = 'form-feedback mt-4 text-sm text-center p-3 rounded-xl bg-red-50 text-red-600 border border-red-100 block';
    }
};
