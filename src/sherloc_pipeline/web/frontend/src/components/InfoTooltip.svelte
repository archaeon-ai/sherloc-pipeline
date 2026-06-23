<script lang="ts">
  export let text: string = '';
  let open = false;
  let btnEl: HTMLButtonElement;
  // Viewport-relative popup coordinates. Computed on open from the button's
  // bounding rect so the popup escapes ancestor `overflow: hidden` containers
  // (the .step-card in the workbench clips an absolutely-positioned popup).
  let popupTop = 0;
  let popupLeft = 0;

  const POPUP_WIDTH = 340;
  // Must match the CSS max-height on .info-popup. The popup scrolls
  // internally if content is longer than this.
  const POPUP_MAX_HEIGHT = 480;
  const POPUP_OFFSET_Y = 6;
  const VIEWPORT_PAD = 8;

  function placePopup() {
    if (!btnEl) return;
    const r = btnEl.getBoundingClientRect();
    // Horizontal: align popup's left edge to button's left, clamped to viewport.
    const maxLeft = window.innerWidth - POPUP_WIDTH - VIEWPORT_PAD;
    popupLeft = Math.max(VIEWPORT_PAD, Math.min(r.left, maxLeft));
    // Vertical: place popup below the button by default, but clamp upward if
    // it would extend past the viewport bottom. If the button is near the
    // bottom of the screen, this effectively flips the popup above the button.
    const desiredTop = r.bottom + POPUP_OFFSET_Y;
    const maxTop = window.innerHeight - VIEWPORT_PAD - POPUP_MAX_HEIGHT;
    popupTop = Math.max(VIEWPORT_PAD, Math.min(desiredTop, maxTop));
  }

  function toggle(e: MouseEvent) {
    e.stopPropagation();
    if (!open) placePopup();
    open = !open;
  }

  function close() {
    open = false;
  }
</script>

<span class="info-tooltip-wrapper">
  <button bind:this={btnEl} class="info-btn" on:click={toggle} title="Method reference">?</button>
  {#if open}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="info-backdrop" on:click={close}></div>
    <div class="info-popup" style="top: {popupTop}px; left: {popupLeft}px;">
      {@html text}
    </div>
  {/if}
</span>

<style>
  .info-tooltip-wrapper {
    position: relative;
    display: inline-flex;
    align-items: center;
  }

  .info-btn {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 1px solid var(--color-border-strong, #94a3b8);
    background: var(--color-surface, #fff);
    color: var(--color-text-secondary, #64748b);
    font-size: 0.7rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    line-height: 1;
  }

  .info-btn:hover {
    border-color: var(--color-primary, #2563eb);
    color: var(--color-primary, #2563eb);
  }

  .info-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 99;
  }

  .info-popup {
    /* Viewport-relative so the popup escapes any ancestor `overflow: hidden`
       (e.g. the .step-card in ProcessingChain). top / left are set inline
       from the button's bounding rect at open time. */
    position: fixed;
    z-index: 100;
    width: 340px;
    /* Must match POPUP_MAX_HEIGHT in the script — placePopup() reserves this
       much space when clamping popupTop into the viewport. */
    max-height: 480px;
    overflow-y: auto;
    padding: 12px;
    background: var(--color-surface, #fff);
    border: 1px solid var(--color-border, #e2e8f0);
    border-radius: var(--radius-md, 8px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    font-size: 0.78rem;
    line-height: 1.5;
    color: var(--color-text, #1e293b);
  }

  .info-popup :global(b) {
    font-weight: 600;
  }

  .info-popup :global(a) {
    color: var(--color-primary, #2563eb);
    text-decoration: none;
  }

  .info-popup :global(a:hover) {
    text-decoration: underline;
  }

  .info-popup :global(.ref-title) {
    font-weight: 600;
    margin-bottom: 4px;
  }

  .info-popup :global(.ref-section) {
    margin-top: 8px;
    font-size: 0.78rem;
  }

  .info-popup :global(.ref-cite) {
    font-size: 0.72rem;
    color: var(--color-text-secondary, #64748b);
    margin-top: 8px;
  }
</style>
