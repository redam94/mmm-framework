/**
 * MMM Framework - Shared Components
 * Dynamically injects navigation and footer into pages
 */

(function() {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================
    // Grouped navigation: Learn / Methodology / Platform / Proof / Project.
    // Top-level entries with `href` render as plain links; entries with
    // `items` render as hover/click dropdowns (flattened in the mobile menu).
    const NAV_GROUPS = [
        { href: 'index.html', label: 'Home' },
        {
            label: 'Learn',
            items: [
                { href: 'getting-started.html', label: 'Getting Started' },
                { href: 'modeling-guide.html', label: 'Modeling Guide' },
                { href: 'real-data-guide.html', label: 'Real-Data Guide' },
                { href: 'interpreting-results.html', label: 'Interpreting Results' },
                { href: 'business-stakeholders.html', label: 'For Business' },
                { href: 'glossary.html', label: 'Glossary' },
                { href: 'faq.html', label: 'FAQ' }
            ]
        },
        {
            label: 'Methodology',
            items: [
                { href: 'causal-inference.html', label: 'Causal Inference' },
                { href: 'bayesian-workflow.html', label: 'Bayesian Workflow' },
                { href: 'variable-selection.html', label: 'Variable Selection' },
                { href: 'scientific-modeling.html', label: 'Scientific Modeling' },
                { href: 'measurement-calibration.html', label: 'Calibration Loop' },
                { href: 'identification-assumptions.html', label: 'Identification Assumptions' },
                { href: 'technical-guide.html', label: 'Technical Guide' }
            ]
        },
        {
            label: 'Platform',
            items: [
                { href: 'platform-overview.html', label: 'Platform Overview' },
                { href: 'data-requirements.html', label: 'Data Requirements & Runtime' },
                { href: 'security.html', label: 'Security & AI Governance' }
            ]
        },
        {
            label: 'Proof',
            items: [
                { href: 'demos.html', label: 'Demos & Reports' },
                { href: 'pressure-testing.html', label: 'Pressure Testing' },
                { href: 'mmm-walkthrough.html', label: 'MMM Walkthrough' },
                { href: 'mmm-example-report.html', label: 'Example Report' },
                { href: 'artifacts/index.html', label: 'Consultant Artifacts' }
            ]
        },
        {
            label: 'Project',
            items: [
                { href: 'about.html', label: 'About' },
                { href: 'evaluator.html', label: 'For Evaluators' },
                { href: 'changelog.html', label: 'Changelog' },
                { href: 'https://github.com/redam94/mmm-framework', label: 'GitHub', external: true }
            ]
        }
    ];

    // Flat list (footer + anything that wants one link per page)
    const NAV_LINKS = NAV_GROUPS.reduce((acc, g) => {
        if (g.href) acc.push(g);
        else acc.push(...g.items);
        return acc;
    }, []);

    // =========================================================================
    // Utility Functions
    // =========================================================================
    
    /**
     * Get the current page filename from the URL
     */
    function getCurrentPage() {
        const path = window.location.pathname;
        const page = path.substring(path.lastIndexOf('/') + 1) || 'index.html';
        return page;
    }

    /**
     * Check if a link is active (current page)
     */
    // Pages that should highlight "Demos & Reports" in the nav
    const DEMO_PAGES = [
        'demos.html',
        'scientific-workflow-demo.html',
        'scientific-workflow-simple.html',
        'workflow-budget-optimization.html',
        'workflow-channel-effectiveness.html',
        'workflow-forecasting.html',
        'workflow-calibration-decisions.html',
        'mmm-example-report.html',
        // Notebook guide series (pressure testing / aurora / workshop / math)
        'pressure-testing.html',
        'stress-00-rosy-picture.html',
        'stress-01-carryover-shape.html',
        'stress-02-time-structure.html',
        'stress-03-confounding-selection.html',
        'stress-04-extension-traps.html',
        'stress-05-gauntlet.html',
        'stress-06-geo-hierarchy.html',
        'mmm-walkthrough.html',
        'aurora-00-overview.html',
        'aurora-01-causality.html',
        'aurora-02-base-mmm.html',
        'aurora-03-extended-mmm.html',
        'aurora-04-reporting.html',
        'aurora-05-unified-workflow.html',
        'causal-features-showcase.html',
        'workshop-00-thinking-in-distributions.html',
        'workshop-01-priors.html',
        'workshop-02-sampling.html',
        'workshop-03-first-mmm.html',
        'workshop-04-reading-the-posterior.html',
        'workshop-05-from-draws-to-decisions.html',
        'math-00-overview.html',
        'math-01-adstock.html',
        'math-02-saturation.html',
        'math-03-seasonality-trend.html',
        'math-04-bayesian-model.html',
        'math-05-calibration.html',
        'math-06-extensions.html'
    ];

    function isActive(href) {
        const currentPage = getCurrentPage();
        // Handle both exact match and index.html for root
        if (href === 'index.html' && (currentPage === '' || currentPage === 'index.html')) {
            return true;
        }
        // Highlight "Demos & Reports" for all demo sub-pages
        if (href === 'demos.html' && DEMO_PAGES.includes(currentPage)) {
            return true;
        }
        return currentPage === href;
    }

    // =========================================================================
    // Navigation Component
    // =========================================================================
    
    function renderLink(link) {
        const activeClass = isActive(link.href) ? ' class="active"' : '';
        const target = link.external ? ' target="_blank" rel="noopener"' : '';
        return `<li><a href="${link.href}"${activeClass}${target}>${link.label}</a></li>`;
    }

    function groupIsActive(group) {
        return group.items.some(item => isActive(item.href));
    }

    function createNavigation() {
        const nav = document.createElement('nav');
        nav.className = 'site-nav';

        const linksHtml = NAV_GROUPS.map(entry => {
            if (entry.href) return renderLink(entry);
            const activeClass = groupIsActive(entry) ? ' active' : '';
            const itemsHtml = entry.items.map(renderLink).join('\n                    ');
            return `<li class="nav-group">
                <button class="nav-group-btn${activeClass}" aria-expanded="false" aria-haspopup="true">${entry.label}<span class="caret" aria-hidden="true">▾</span></button>
                <ul class="nav-dropdown">
                    ${itemsHtml}
                </ul>
            </li>`;
        }).join('\n                ');

        nav.innerHTML = `
        <div class="nav-content">
            <a href="index.html" class="logo">MMM Framework</a>
            <button class="mobile-menu-btn" aria-label="Toggle menu">
                <span></span>
                <span></span>
                <span></span>
            </button>
            <ul class="nav-links">
                ${linksHtml}
            </ul>
        </div>
        `;

        // Mobile menu toggle
        const menuBtn = nav.querySelector('.mobile-menu-btn');
        const navLinks = nav.querySelector('.nav-links');

        menuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            menuBtn.classList.toggle('active');
        });

        // Dropdown groups: click toggles (touch devices); hover/focus is CSS.
        function closeGroups(except) {
            nav.querySelectorAll('.nav-group.open').forEach(g => {
                if (g !== except) {
                    g.classList.remove('open');
                    const b = g.querySelector('.nav-group-btn');
                    if (b) b.setAttribute('aria-expanded', 'false');
                }
            });
        }

        nav.querySelectorAll('.nav-group-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const li = btn.parentElement;
                const willOpen = !li.classList.contains('open');
                closeGroups(li);
                li.classList.toggle('open', willOpen);
                btn.setAttribute('aria-expanded', String(willOpen));
            });
        });

        document.addEventListener('click', (e) => {
            if (!nav.contains(e.target)) closeGroups();
        });

        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                menuBtn.classList.remove('active');
                closeGroups();
            });
        });

        return nav;
    }

    // =========================================================================
    // Footer Component
    // =========================================================================
    
    function createFooter() {
        const footer = document.createElement('footer');
        
        const linksHtml = NAV_LINKS.map(link => {
            const target = link.external ? ' target="_blank"' : '';
            return `<a href="${link.href}"${target}>${link.label}</a>`;
        }).join('\n                ');

        footer.innerHTML = `
        <div class="footer-content">
            <span class="logo">MMM Framework</span>
            <div class="footer-links">
                ${linksHtml}
            </div>
        </div>
        <div class="footer-meta">
            <span>Apache-2.0 licensed</span>
            <span>v0.1.0 (Beta, in development)</span>
            <a href="changelog.html">Changelog &amp; API stability</a>
            <a href="evaluator.html">For evaluators</a>
            <a href="https://github.com/redam94/mmm-framework" target="_blank" rel="noopener">Source on GitHub</a>
        </div>
        `;

        return footer;
    }

    // =========================================================================
    // Scroll Animations
    // =========================================================================
    
    function initScrollAnimations() {
        const fadeElements = document.querySelectorAll('.fade-in');
        
        if (fadeElements.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        fadeElements.forEach(el => observer.observe(el));
    }

    // =========================================================================
    // Sidebar Active State (for pages with sidebar navigation)
    // =========================================================================
    
    function initSidebarActiveState() {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        const sections = document.querySelectorAll('.main-content [id]');
        const sidebarLinks = sidebar.querySelectorAll('a[href^="#"]');

        if (sections.length === 0 || sidebarLinks.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    sidebarLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, {
            threshold: 0,
            rootMargin: '-100px 0px -66% 0px'
        });

        sections.forEach(section => observer.observe(section));
    }

    // =========================================================================
    // Collapsible Sidebar (narrow screens)
    // =========================================================================

    function initCollapsibleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        // Wrap existing sidebar-nav content in a wrapper div
        const navElements = sidebar.querySelectorAll('.sidebar-nav, h3');
        if (navElements.length === 0) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'sidebar-nav-wrapper';
        navElements.forEach(el => wrapper.appendChild(el));

        // Create toggle button
        const toggle = document.createElement('button');
        toggle.className = 'sidebar-toggle';
        toggle.textContent = 'Table of Contents';
        toggle.setAttribute('aria-expanded', 'true');
        toggle.setAttribute('aria-label', 'Toggle table of contents');

        // Insert toggle and wrapper into sidebar
        sidebar.appendChild(toggle);
        sidebar.appendChild(wrapper);

        // Start collapsed on narrow screens
        function checkWidth() {
            if (window.innerWidth <= 1024) {
                sidebar.classList.add('collapsed');
                toggle.setAttribute('aria-expanded', 'false');
            } else {
                sidebar.classList.remove('collapsed');
                toggle.setAttribute('aria-expanded', 'true');
            }
        }

        checkWidth();
        window.addEventListener('resize', checkWidth);

        // Toggle on click
        toggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            const expanded = !sidebar.classList.contains('collapsed');
            toggle.setAttribute('aria-expanded', String(expanded));
        });

        // Collapse after clicking a link on narrow screens
        wrapper.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 1024) {
                    sidebar.classList.add('collapsed');
                    toggle.setAttribute('aria-expanded', 'false');
                }
            });
        });
    }

    // =========================================================================
    // Plotly headroom guard
    // -------------------------------------------------------------------------
    // Many guide pages render Plotly charts with tight top margins. Two
    // failure modes are handled after each plot mounts:
    //   1. titles / top annotations clipping against the SVG edge, and
    //   2. the title overlapping a top-anchored legend or annotation.
    // Both are resolved by pinning the title to the container top and widening
    // margin.t by exactly the measured shortfall. Re-runs converge and are
    // capped so a pathological chart can never loop.
    // =========================================================================

    function ensurePlotHeadroom(gd) {
        if (!window.Plotly || !gd._fullLayout) return;
        if (gd.dataset.headroomOk === '1') return;
        const tries = Number(gd.dataset.headroomTries || 0);
        if (tries >= 4) return;
        const rect = gd.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return; // hidden (e.g. inactive tab)

        // 1. Clipping against the container top
        let clip = 0;
        let titleClipped = false;
        gd.querySelectorAll('text.gtitle, g.annotation').forEach(node => {
            const r = node.getBoundingClientRect();
            if (r.height > 0 && r.top < rect.top) {
                clip = Math.max(clip, rect.top - r.top);
                if (node.classList.contains('gtitle')) titleClipped = true;
            }
        });

        // 2. Title overlapping a legend or annotation in the top margin
        let overlap = 0;
        const title = gd.querySelector('text.gtitle');
        if (title) {
            const tr = title.getBoundingClientRect();
            if (tr.height > 0) {
                gd.querySelectorAll('g.legend, g.annotation').forEach(node => {
                    const r = node.getBoundingClientRect();
                    const intersects = tr.left < r.right && r.left < tr.right &&
                                       tr.top < r.bottom && r.top < tr.bottom;
                    if (r.height > 0 && intersects) {
                        overlap = Math.max(overlap,
                            Math.min(tr.bottom, r.bottom) - Math.max(tr.top, r.top));
                    }
                });
            }
        }

        if (clip > 0.5 || overlap > 0.5) {
            gd.dataset.headroomTries = String(tries + 1);
            const fl = gd._fullLayout;
            const t = (fl.margin && fl.margin.t) || 0;
            const update = { 'margin.t': Math.ceil(t + clip + overlap + 6) };
            if (overlap > 0.5) {
                // Pin the title near the container top so the widened margin
                // separates it from the legend instead of re-centering both.
                update['title.y'] = 1 - 8 / rect.height;
                update['title.yanchor'] = 'top';
            }
            // A title pinned at a numeric container y ignores margin changes —
            // lower the title itself by the clipped amount instead.
            const tt = fl.title || {};
            if (titleClipped && typeof tt.y === 'number') {
                update['title.y'] = Math.max(0, tt.y - (clip + 3) / rect.height);
                update['title.yanchor'] = 'top';
            }
            try {
                window.Plotly.relayout(gd, update);
            } catch (e) { /* never break the page over a margin */ }
        } else {
            gd.dataset.headroomOk = '1';
        }
    }

    function checkAllPlots() {
        document.querySelectorAll('.js-plotly-plot').forEach(ensurePlotHeadroom);
    }

    function initPlotHeadroomGuard() {
        // Initial passes: charts render at parse time and shortly after load
        window.addEventListener('load', () => {
            checkAllPlots();
            setTimeout(checkAllPlots, 1200);
            setTimeout(checkAllPlots, 3500);
        });

        // Lazily rendered charts (tabs, scroll-triggered draws) add DOM nodes;
        // debounce a re-check whenever the document grows new plot content.
        let pending = null;
        const observer = new MutationObserver(() => {
            if (pending) return;
            pending = setTimeout(() => {
                pending = null;
                checkAllPlots();
            }, 350);
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    // =========================================================================
    // Initialize
    // =========================================================================

    function ensureMainTarget() {
        // Pick the most sensible main-content anchor on this page so the
        // skip link has somewhere to land. Prefer an explicit id, then
        // <main>, then .main-content, then the first <section>.
        if (document.getElementById('main-content')) return;
        const candidate = document.querySelector('main, .main-content, section');
        if (candidate) candidate.id = 'main-content';
    }

    function createSkipLink() {
        const a = document.createElement('a');
        a.className = 'skip-link';
        a.href = '#main-content';
        a.textContent = 'Skip to main content';
        return a;
    }

    function init() {
        // Make sure there's a target for the skip link
        ensureMainTarget();

        // Insert skip link first so it's the first focusable element
        const skip = createSkipLink();
        document.body.insertBefore(skip, document.body.firstChild);

        // Insert navigation after the skip link
        const nav = createNavigation();
        document.body.insertBefore(nav, skip.nextSibling);

        // Insert footer at the end of body
        const footer = createFooter();
        document.body.appendChild(footer);

        // Initialize features
        initScrollAnimations();
        initSidebarActiveState();
        initCollapsibleSidebar();
        initPlotHeadroomGuard();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();