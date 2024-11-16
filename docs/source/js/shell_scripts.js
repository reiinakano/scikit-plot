document.addEventListener('DOMContentLoaded', function() {
    // Function to copy code to clipboard
    function copyCode(event) {
        const button = event.currentTarget;
        const targetId = button.getAttribute('data-copy-target');
        const codeBlock = document.getElementById(targetId);

        if (codeBlock) {
            const codeText = codeBlock.textContent.trim();

            // Try Clipboard API first
            if (navigator.clipboard && navigator.clipboard.writeText) {
                handleClipboardCopy(codeText, button);
            } else {
                // Fallback method if Clipboard API is not available
                fallbackCopyText(codeText, button);
            }
        } else {
            console.warn("Code block not found.");
        }
    }

    // Handle Clipboard API copy process
    function handleClipboardCopy(text, button) {
        navigator.clipboard.writeText(text)
            .then(() => showCopySuccess(button))
            .catch(() => fallbackCopyText(text, button));
    }

    // Show copy success feedback on the button
    function showCopySuccess(button) {
        const originalText = button.innerHTML;  // "&#128203; Copy"
        button.innerHTML = "Copied!";  // Change to 'Copied' icon
        button.style.color = 'green';

        setTimeout(() => {
            // Revert back to original state after 2 seconds
            button.innerHTML = originalText;
            button.style.color = '#007bff';
        }, 2000);
    }

    // Fallback method using a temporary textarea for unsupported Clipboard API
    function fallbackCopyText(text, button) {
        const tempTextArea = document.createElement('textarea');
        tempTextArea.value = text;
        document.body.appendChild(tempTextArea);
        tempTextArea.select();

        try {
            document.execCommand('copy');  // Try to copy using execCommand
            showCopySuccess(button);  // Show success feedback
        } catch (err) {
            console.error('Fallback copy failed', err);
            alert('Failed to copy text. Please try again.');
        }

        document.body.removeChild(tempTextArea);  // Clean up the temporary textarea
    }

    // Attach the copy function to all buttons with the class 'copy-btn'
    document.querySelectorAll('.copy-btn').forEach(button => {
       button.addEventListener('click', copyCode);
    });
});