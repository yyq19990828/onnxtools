document.addEventListener('DOMContentLoaded', async () => {
  document.querySelectorAll('.mermaid').forEach(el => {
    el.setAttribute('data-mermaid-src', el.textContent)
  })

  const getTheme = () =>
    document.body.getAttribute('data-md-color-scheme') === 'slate' ? 'dark' : 'default'

  const render = async () => {
    document.querySelectorAll('.mermaid').forEach(el => {
      const src = el.getAttribute('data-mermaid-src')
      if (src) {
        el.removeAttribute('data-processed')
        el.innerHTML = ''
        el.textContent = src
      }
    })
    mermaid.initialize({ startOnLoad: false, theme: getTheme() })
    await mermaid.run({ querySelector: '.mermaid' })
  }

  await render()

  new MutationObserver(render).observe(document.body, {
    attributes: true,
    attributeFilter: ['data-md-color-scheme']
  })
})
