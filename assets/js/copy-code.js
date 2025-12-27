document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('div.highlighter-rouge pre').forEach((pre) => {
    const btn = document.createElement('button');
    btn.textContent = 'Copy';
    btn.className = 'copy-code-button';
    btn.addEventListener('click', () => {
      const text = pre.innerText;
      navigator.clipboard.writeText(text);
      const old = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => (btn.textContent = old), 1500);
    });
    const wrapper = pre.parentElement;
    wrapper.style.position = 'relative';
    wrapper.prepend(btn);
  });
});
