import React, { useEffect, useRef, useState } from 'react';

// Viewport gate: true once the element is within `rootMargin` of the viewport.
// Used to defer heavy work (mounting Plotly, highlighting code) for off-screen
// cards so the number of LIVE heavy widgets stays bounded by what's visible —
// this is what keeps the page responsive as outputs accumulate. Stays true once
// seen (we don't tear widgets back down) so scrolling back is instant; the cap
// is on how many ever mount *at once* during the initial reveal, not lifetime.
export function useInView<T extends Element>(rootMargin = '800px'): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || inView) return;
    // Fallback for environments without IntersectionObserver (e.g. JSDOM): reveal
    // immediately. Synchronous setState is intentional and one-shot here.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (typeof IntersectionObserver === 'undefined') { setInView(true); return; }
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setInView(true);
      },
      { rootMargin }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [rootMargin, inView]);
  return [ref, inView];
}
