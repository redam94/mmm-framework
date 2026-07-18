/**
 * MMM Framework - Shared Components
 * Dynamically injects navigation and footer into pages
 */

(function() {
    'use strict';

    // =========================================================================
    // Theme (light / dark) — resolved synchronously so the first paint uses
    // the right palette. localStorage wins; otherwise the OS preference.
    // =========================================================================
    const THEME_KEY = 'mmm-docs-theme';

    function storedTheme() {
        try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
    }

    function systemTheme() {
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
            ? 'dark' : 'light';
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
    }

    applyTheme(storedTheme() || systemTheme());

    // Follow OS changes while the user hasn't picked explicitly.
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!storedTheme()) applyTheme(e.matches ? 'dark' : 'light');
        });
    }

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
                { href: 'reading-the-report.html', label: 'Reading the Report' },
                { href: 'migration-guide.html', label: 'Migration Guide' },
                { href: 'business-stakeholders.html', label: 'For Business' },
                { href: 'glossary.html', label: 'Glossary' },
                { href: 'faq.html', label: 'FAQ' },
                { href: 'troubleshooting.html', label: 'Troubleshooting' }
            ]
        },
        {
            label: 'Tutorials',
            items: [
                { href: 'workshop-00-thinking-in-distributions.html', label: 'Workshop (Beginner)' },
                { href: 'aurora-00-overview.html', label: 'Aurora — Framework Tour' },
                { href: 'causal-00-the-ladder.html', label: 'Causal Inference Series' },
                { href: 'stress-00-rosy-picture.html', label: 'Pressure-Test Series' },
                { href: 'math-00-overview.html', label: 'Mathematics Series' },
                { href: 'demos.html#recommended-paths', label: 'Recommended paths →' }
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
                { href: 'continuous-learning.html', label: 'Continuous Learning' },
                { href: 'continuous-learning-math.html', label: 'Continuous Learning · Math' },
                { href: 'identification-assumptions.html', label: 'Identification Assumptions' },
                { href: 'technical-guide.html', label: 'Technical Guide' }
            ]
        },
        {
            label: 'Platform',
            items: [
                { href: 'platform-overview.html', label: 'Platform Overview' },
                { href: 'model-garden.html', label: 'Model Garden & Atelier' },
                { href: 'pricing.html', label: 'Pricing' },
                { href: 'data-requirements.html', label: 'Data Requirements & Runtime' },
                { href: 'data-prep-cookbook.html', label: 'Data-Prep Cookbook' },
                { href: 'data-connections.html', label: 'Data Connections' },
                { href: 'trust.html', label: 'Trust & Security' },
                { href: 'security.html', label: 'Security & AI Governance' },
                { href: 'responsible-disclosure.html', label: 'Responsible Disclosure' }
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
        { href: 'blog.html', label: 'Research' },
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

    const GITHUB_REPO = 'https://github.com/redam94/mmm-framework';

    // Ordered page series → automatic Previous/Next cards above the footer.
    const SERIES = {
        'Math series': [
            ['math-00-overview.html', 'The Generative Model'],
            ['math-01-adstock.html', 'The Mathematics of Adstock'],
            ['math-02-saturation.html', 'The Mathematics of Saturation'],
            ['math-03-seasonality-trend.html', 'Seasonality & Trend'],
            ['math-04-bayesian-model.html', 'The Bayesian Model'],
            ['math-05-calibration.html', 'Calibration'],
            ['math-06-extensions.html', 'Extensions']
        ],
        'Workshop series': [
            ['workshop-00-thinking-in-distributions.html', 'Thinking in Distributions'],
            ['workshop-01-priors.html', 'Priors'],
            ['workshop-02-sampling.html', 'Sampling'],
            ['workshop-03-first-mmm.html', 'Your First MMM'],
            ['workshop-04-reading-the-posterior.html', 'Reading the Posterior'],
            ['workshop-05-from-draws-to-decisions.html', 'From Draws to Decisions']
        ],
        'Pressure-test series': [
            ['stress-00-rosy-picture.html', 'The Rosy Picture'],
            ['stress-01-carryover-shape.html', 'Carryover & Shape'],
            ['stress-02-time-structure.html', 'Time Structure'],
            ['stress-03-confounding-selection.html', 'Confounding & Selection'],
            ['stress-04-extension-traps.html', 'Extension Traps'],
            ['stress-05-gauntlet.html', 'The Gauntlet'],
            ['stress-06-geo-hierarchy.html', 'Geo Hierarchy']
        ],
        'Aurora tour': [
            ['aurora-00-overview.html', 'Overview'],
            ['aurora-01-causality.html', 'Causality'],
            ['aurora-02-base-mmm.html', 'The Base MMM'],
            ['aurora-03-extended-mmm.html', 'Extended Models'],
            ['aurora-04-reporting.html', 'Reporting'],
            ['aurora-05-unified-workflow.html', 'The Unified Workflow']
        ],
        'Causal inference series': [
            ['causal-00-the-ladder.html', 'The Ladder of Evidence'],
            ['causal-01-confounding-adjustment.html', 'Confounding & Adjustment'],
            ['causal-02-mmm-as-causal-model.html', 'The MMM as a Causal Model'],
            ['causal-03-structural-mediation.html', 'Structural Mediation'],
            ['causal-04-latent-confounders.html', 'Latent Confounders'],
            ['causal-05-measuring-one-experiment.html', 'Measuring One Experiment'],
            ['causal-06-calibrating-the-model.html', 'Calibrating the Model'],
            ['causal-07-many-experiments.html', 'Many Experiments'],
            ['causal-08-designing-next-experiment.html', 'Designing the Next Experiment'],
            ['causal-09-measurement-program.html', 'The Measurement Program'],
            ['causal-10-closed-loop.html', 'The Closed Loop']
        ],
        'Decision workflows': [
            ['workflow-channel-effectiveness.html', 'Channel Effectiveness'],
            ['workflow-budget-optimization.html', 'Budget Optimization'],
            ['workflow-forecasting.html', 'Forecasting'],
            ['workflow-calibration-decisions.html', 'Calibration Decisions']
        ],
        'Measurement research': [
            ['blog-activity-bias.html', 'Activity Bias in Ad Measurement'],
            ['blog-causal-estimates-observational.html', 'Causal Estimates Without Experiments'],
            ['blog-table-2-fallacy.html', 'The Table 2 Fallacy'],
            ['blog-attribution-incrementality.html', 'Attribution Is Not Incrementality'],
            ['blog-geo-experiments-tbr.html', 'Geo Experiments & TBR'],
            ['blog-synthetic-control.html', 'Synthetic Control, Done Right'],
            ['blog-staggered-did.html', 'Modern Staggered DiD'],
            ['blog-pretrends-testing.html', 'Pretest With Caution: Parallel Trends'],
            ['blog-causalimpact-bsts.html', 'CausalImpact & Structural Time Series'],
            ['blog-bayesian-mmm-carryover-shape.html', 'Carryover & Shape in Bayesian MMM'],
            ['blog-table-2-fallacy-mmm.html', 'The Table 2 Fallacy in MMMs'],
            ['blog-surrogate-outcomes.html', 'Surrogate Outcomes for the Long Run'],
            ['blog-carryover-experiment-timing.html', 'Adstock & Experiment Timing'],
            ['blog-modeling-pitfalls.html', 'Statistical Modeling Pitfalls'],
            ['blog-multiple-comparisons.html', 'Multiple Comparisons & Model Selection'],
            ['blog-lindley-to-dad.html', 'Lindley to Deep Adaptive Design'],
            ['blog-geo-holdout-eig.html', 'A Geo-Holdout as Bayesian Design'],
            ['blog-bed-bo-bandits.html', 'BED vs. BO vs. Bandits'],
            ['blog-thompson-sampling.html', 'Thompson Sampling in Practice'],
            ['blog-continuous-learning-interactions.html', 'Continuous Learning with Interactions']
        ]
    };

    // Audience tier per page (three-tier editorial policy). Pages not listed
    // get no tier chip. Reading time is computed for any page with a chip row.
    const TIER_OVERVIEW = 'overview', TIER_ANALYST = 'analyst', TIER_TECHNICAL = 'technical';
    const PAGE_TIERS = {
        'getting-started.html': TIER_OVERVIEW,
        'troubleshooting.html': TIER_OVERVIEW,
        'about.html': TIER_OVERVIEW,
        'business-stakeholders.html': TIER_OVERVIEW,
        'faq.html': TIER_OVERVIEW,
        'glossary.html': TIER_OVERVIEW,
        'demos.html': TIER_OVERVIEW,
        'changelog.html': TIER_OVERVIEW,
        'platform-overview.html': TIER_OVERVIEW,
        'pricing.html': TIER_OVERVIEW,
        'trust.html': TIER_OVERVIEW,
        'data-connections.html': TIER_OVERVIEW,
        'security.html': TIER_OVERVIEW,
        'responsible-disclosure.html': TIER_OVERVIEW,

        'modeling-guide.html': TIER_ANALYST,
        'real-data-guide.html': TIER_ANALYST,
        'interpreting-results.html': TIER_ANALYST,
        'reading-the-report.html': TIER_ANALYST,
        'data-prep-cookbook.html': TIER_ANALYST,
        'migration-guide.html': TIER_ANALYST,
        'causal-inference.html': TIER_ANALYST,
        'bayesian-workflow.html': TIER_ANALYST,
        'variable-selection.html': TIER_ANALYST,
        'scientific-modeling.html': TIER_ANALYST,
        'measurement-calibration.html': TIER_ANALYST,
        'continuous-learning.html': TIER_ANALYST,
        'identification-assumptions.html': TIER_ANALYST,
        'pressure-testing.html': TIER_ANALYST,
        'data-requirements.html': TIER_ANALYST,
        'evaluator.html': TIER_ANALYST,
        'model-garden.html': TIER_ANALYST,
        'mmm-walkthrough.html': TIER_ANALYST,
        'mmm-example-report.html': TIER_ANALYST,
        'causal-features-showcase.html': TIER_ANALYST,
        'scientific-workflow-demo.html': TIER_ANALYST,
        'scientific-workflow-simple.html': TIER_ANALYST,
        'workflow-budget-optimization.html': TIER_ANALYST,
        'workflow-channel-effectiveness.html': TIER_ANALYST,
        'workflow-forecasting.html': TIER_ANALYST,
        'workflow-calibration-decisions.html': TIER_ANALYST,
        'workshop-00-thinking-in-distributions.html': TIER_ANALYST,
        'workshop-01-priors.html': TIER_ANALYST,
        'workshop-02-sampling.html': TIER_ANALYST,
        'workshop-03-first-mmm.html': TIER_ANALYST,
        'workshop-04-reading-the-posterior.html': TIER_ANALYST,
        'workshop-05-from-draws-to-decisions.html': TIER_ANALYST,
        'stress-00-rosy-picture.html': TIER_ANALYST,
        'stress-01-carryover-shape.html': TIER_ANALYST,
        'stress-02-time-structure.html': TIER_ANALYST,
        'stress-03-confounding-selection.html': TIER_ANALYST,
        'stress-04-extension-traps.html': TIER_ANALYST,
        'stress-05-gauntlet.html': TIER_ANALYST,
        'stress-06-geo-hierarchy.html': TIER_ANALYST,
        'aurora-00-overview.html': TIER_ANALYST,
        'aurora-01-causality.html': TIER_ANALYST,
        'aurora-02-base-mmm.html': TIER_ANALYST,
        'aurora-03-extended-mmm.html': TIER_ANALYST,
        'aurora-04-reporting.html': TIER_ANALYST,
        'aurora-05-unified-workflow.html': TIER_ANALYST,
        'causal-00-the-ladder.html': TIER_ANALYST,
        'causal-01-confounding-adjustment.html': TIER_ANALYST,
        'causal-02-mmm-as-causal-model.html': TIER_ANALYST,
        'causal-03-structural-mediation.html': TIER_ANALYST,
        'causal-04-latent-confounders.html': TIER_ANALYST,
        'causal-05-measuring-one-experiment.html': TIER_ANALYST,
        'causal-06-calibrating-the-model.html': TIER_ANALYST,
        'causal-07-many-experiments.html': TIER_ANALYST,
        'causal-08-designing-next-experiment.html': TIER_ANALYST,
        'causal-09-measurement-program.html': TIER_ANALYST,
        'causal-10-closed-loop.html': TIER_ANALYST,

        'blog.html': TIER_OVERVIEW,
        'blog-activity-bias.html': TIER_ANALYST,
        'blog-causal-estimates-observational.html': TIER_ANALYST,
        'blog-table-2-fallacy.html': TIER_ANALYST,
        'blog-table-2-fallacy-mmm.html': TIER_ANALYST,
        'blog-attribution-incrementality.html': TIER_ANALYST,
        'blog-synthetic-control.html': TIER_ANALYST,
        'blog-carryover-experiment-timing.html': TIER_ANALYST,
        'blog-modeling-pitfalls.html': TIER_ANALYST,
        'blog-multiple-comparisons.html': TIER_ANALYST,
        'blog-bed-bo-bandits.html': TIER_ANALYST,
        'blog-thompson-sampling.html': TIER_ANALYST,
        'blog-continuous-learning-interactions.html': TIER_ANALYST,

        'blog-geo-experiments-tbr.html': TIER_TECHNICAL,
        'blog-staggered-did.html': TIER_TECHNICAL,
        'blog-pretrends-testing.html': TIER_TECHNICAL,
        'blog-surrogate-outcomes.html': TIER_TECHNICAL,
        'blog-causalimpact-bsts.html': TIER_TECHNICAL,
        'blog-bayesian-mmm-carryover-shape.html': TIER_TECHNICAL,
        'blog-lindley-to-dad.html': TIER_TECHNICAL,
        'blog-geo-holdout-eig.html': TIER_TECHNICAL,

        'technical-guide.html': TIER_TECHNICAL,
        'continuous-learning-math.html': TIER_TECHNICAL,
        'math-00-overview.html': TIER_TECHNICAL,
        'math-01-adstock.html': TIER_TECHNICAL,
        'math-02-saturation.html': TIER_TECHNICAL,
        'math-03-seasonality-trend.html': TIER_TECHNICAL,
        'math-04-bayesian-model.html': TIER_TECHNICAL,
        'math-05-calibration.html': TIER_TECHNICAL,
        'math-06-extensions.html': TIER_TECHNICAL
    };

    const TIER_LABELS = {
        overview: 'Overview · plain language',
        analyst: 'Analyst level',
        technical: 'Technical · math leads'
    };

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
        'math-06-extensions.html',
        'causal-00-the-ladder.html',
        'causal-01-confounding-adjustment.html',
        'causal-02-mmm-as-causal-model.html',
        'causal-03-structural-mediation.html',
        'causal-04-latent-confounders.html',
        'causal-05-measuring-one-experiment.html',
        'causal-06-calibrating-the-model.html',
        'causal-07-many-experiments.html',
        'causal-08-designing-next-experiment.html',
        'causal-09-measurement-program.html',
        'causal-10-closed-loop.html'
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
        // Highlight "Research" for the blog index and every blog post
        if (href === 'blog.html' && (currentPage === 'blog.html' || currentPage.startsWith('blog-'))) {
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

        const isMac = /Mac|iPhone|iPad/.test(navigator.platform || '');
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
            <div class="nav-tools">
                <button class="nav-tool-btn search-trigger" aria-label="Search the documentation">
                    ${SEARCH_ICON}
                    <span class="search-kbd">${isMac ? '⌘' : 'Ctrl'} K</span>
                </button>
                <button class="nav-tool-btn theme-toggle" aria-label="Switch between light and dark theme">
                    ${currentThemeIcon()}
                </button>
            </div>
        </div>
        `;

        nav.querySelector('.search-trigger').addEventListener('click', () => {
            openSearch();
        });

        const themeBtn = nav.querySelector('.theme-toggle');
        themeBtn.addEventListener('click', () => {
            const next = document.documentElement.getAttribute('data-theme') === 'dark'
                ? 'light' : 'dark';
            applyTheme(next);
            try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* private mode */ }
            themeBtn.innerHTML = currentThemeIcon();
        });

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
            <span>v0.2.0 (Beta, in development)</span>
            <a href="changelog.html">Changelog &amp; API stability</a>
            <a href="evaluator.html">For evaluators</a>
            <a href="https://github.com/redam94/mmm-framework" target="_blank" rel="noopener">Source on GitHub</a>
            <span class="footer-page-actions">
                <a href="${GITHUB_REPO}/edit/main/docs/${getCurrentPage()}" target="_blank" rel="noopener">Edit this page</a>
                <a href="${GITHUB_REPO}/issues/new?title=${encodeURIComponent('Docs: ' + getCurrentPage())}" target="_blank" rel="noopener">Report an issue</a>
            </span>
        </div>
        `;

        return footer;
    }

    // =========================================================================
    // Icons & small helpers
    // =========================================================================

    const SEARCH_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><line x1="21" y1="21" x2="16.5" y2="16.5"></line></svg>';
    const SUN_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5"></circle><line x1="12" y1="2" x2="12" y2="4.5"></line><line x1="12" y1="19.5" x2="12" y2="22"></line><line x1="2" y1="12" x2="4.5" y2="12"></line><line x1="19.5" y1="12" x2="22" y2="12"></line><line x1="4.9" y1="4.9" x2="6.7" y2="6.7"></line><line x1="17.3" y1="17.3" x2="19.1" y2="19.1"></line><line x1="4.9" y1="19.1" x2="6.7" y2="17.3"></line><line x1="17.3" y1="6.7" x2="19.1" y2="4.9"></line></svg>';
    const MOON_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path></svg>';
    const COPY_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"></rect><path d="M5 15V5a2 2 0 0 1 2-2h10"></path></svg>';

    function currentThemeIcon() {
        // Show the theme you would switch TO.
        return document.documentElement.getAttribute('data-theme') === 'dark'
            ? SUN_ICON : MOON_ICON;
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // =========================================================================
    // Search palette (Cmd/Ctrl-K) over shared/search-index.json
    // =========================================================================

    let searchOverlay = null;
    let searchIndex = null;       // null = not loaded, [] = loading failed
    let searchSelected = 0;
    let searchResults = [];

    function buildSearchOverlay() {
        if (searchOverlay) return searchOverlay;

        searchOverlay = document.createElement('div');
        searchOverlay.className = 'search-overlay';
        searchOverlay.setAttribute('role', 'dialog');
        searchOverlay.setAttribute('aria-modal', 'true');
        searchOverlay.setAttribute('aria-label', 'Search the documentation');
        searchOverlay.innerHTML = `
            <div class="search-modal">
                <div class="search-input-row">
                    ${SEARCH_ICON}
                    <input type="text" class="search-input" placeholder="Search the docs…"
                           aria-label="Search query" autocomplete="off" spellcheck="false">
                </div>
                <ul class="search-results" role="listbox"></ul>
                <div class="search-hint">Type to search across every page — titles, sections and text.</div>
                <div class="search-footer">
                    <span><kbd>↑</kbd> <kbd>↓</kbd> navigate</span>
                    <span><kbd>Enter</kbd> open</span>
                    <span><kbd>Esc</kbd> close</span>
                </div>
            </div>`;

        const input = searchOverlay.querySelector('.search-input');

        searchOverlay.addEventListener('click', (e) => {
            if (e.target === searchOverlay) closeSearch();
        });

        input.addEventListener('input', () => runSearch(input.value));

        input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                if (!searchResults.length) return;
                searchSelected = (searchSelected + (e.key === 'ArrowDown' ? 1 : -1)
                    + searchResults.length) % searchResults.length;
                paintSelection();
            } else if (e.key === 'Enter') {
                const current = searchResults[searchSelected];
                if (current) window.location.href = current.href;
            } else if (e.key === 'Escape') {
                closeSearch();
            }
        });

        document.body.appendChild(searchOverlay);
        return searchOverlay;
    }

    function openSearch() {
        const overlay = buildSearchOverlay();
        overlay.classList.add('open');
        const input = overlay.querySelector('.search-input');
        input.value = '';
        runSearch('');
        input.focus();
        ensureSearchIndex();
    }

    function closeSearch() {
        if (searchOverlay) searchOverlay.classList.remove('open');
    }

    function ensureSearchIndex() {
        if (searchIndex !== null) return Promise.resolve(searchIndex);
        return fetch('shared/search-index.json')
            .then(r => (r.ok ? r.json() : []))
            .then(idx => { searchIndex = idx; return idx; })
            .catch(() => { searchIndex = []; return searchIndex; });
    }

    function searchSnippet(body, term) {
        const at = body.toLowerCase().indexOf(term);
        if (at < 0) return escapeHtml(body.slice(0, 130));
        const start = Math.max(0, at - 50);
        const raw = (start > 0 ? '…' : '') + body.slice(start, at + term.length + 90);
        const rel = at - start + (start > 0 ? 1 : 0);
        return escapeHtml(raw.slice(0, rel)) + '<mark>'
            + escapeHtml(raw.slice(rel, rel + term.length)) + '</mark>'
            + escapeHtml(raw.slice(rel + term.length));
    }

    function runSearch(query) {
        const overlay = buildSearchOverlay();
        const list = overlay.querySelector('.search-results');
        const hint = overlay.querySelector('.search-hint');
        const q = query.trim().toLowerCase();

        if (!q) {
            list.innerHTML = '';
            hint.style.display = '';
            hint.textContent = 'Type to search across every page — titles, sections and text.';
            searchResults = [];
            return;
        }

        ensureSearchIndex().then(() => {
            const scored = [];
            (searchIndex || []).forEach(page => {
                const title = page.t || '';
                const body = page.b || '';
                let best = null;

                if (title.toLowerCase().includes(q)) {
                    best = { score: 100 + (title.toLowerCase().startsWith(q) ? 20 : 0),
                             href: page.p, title, section: '',
                             snippet: escapeHtml(page.d || body.slice(0, 130)) };
                }

                (page.h || []).forEach(([id, heading]) => {
                    if (heading.toLowerCase().includes(q)) {
                        const cand = { score: 60, href: page.p + (id ? '#' + id : ''),
                                       title, section: heading,
                                       snippet: escapeHtml(page.d || '') };
                        if (!best || cand.score > best.score) best = cand;
                    }
                });

                if (!best && body.toLowerCase().includes(q)) {
                    best = { score: 25, href: page.p, title, section: '',
                             snippet: searchSnippet(body, q) };
                }

                if (best) scored.push(best);
            });

            scored.sort((a, b) => b.score - a.score);
            searchResults = scored.slice(0, 12);
            searchSelected = 0;

            if (!searchResults.length) {
                list.innerHTML = '';
                hint.style.display = '';
                hint.textContent = searchIndex && searchIndex.length
                    ? 'No matches. Try a shorter term — e.g. "adstock", "priors", "ROI".'
                    : 'The search index has not been built for this copy of the docs.';
                return;
            }

            hint.style.display = 'none';
            list.innerHTML = searchResults.map((r, i) => `
                <li class="search-result${i === searchSelected ? ' selected' : ''}" role="option">
                    <a href="${escapeHtml(r.href)}">
                        <div class="result-title">${escapeHtml(r.title)}${r.section
                            ? ` <span class="result-section">· ${escapeHtml(r.section)}</span>` : ''}</div>
                        <div class="result-snippet">${r.snippet}</div>
                    </a>
                </li>`).join('');
        });
    }

    function paintSelection() {
        if (!searchOverlay) return;
        searchOverlay.querySelectorAll('.search-result').forEach((li, i) => {
            li.classList.toggle('selected', i === searchSelected);
            if (i === searchSelected) li.scrollIntoView({ block: 'nearest' });
        });
    }

    function initSearchShortcuts() {
        document.addEventListener('keydown', (e) => {
            const inField = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || ''))
                || e.target.isContentEditable;
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                if (searchOverlay && searchOverlay.classList.contains('open')) closeSearch();
                else openSearch();
            } else if (e.key === '/' && !inField
                       && !(searchOverlay && searchOverlay.classList.contains('open'))) {
                e.preventDefault();
                openSearch();
            } else if (e.key === 'Escape') {
                closeSearch();
            }
        });
    }

    // =========================================================================
    // Copy buttons on every <pre> block
    // =========================================================================

    function initCopyButtons() {
        document.querySelectorAll('pre').forEach(pre => {
            if (pre.closest('.code-block-wrap')) return;
            if (pre.querySelector('.code-copy-btn')) return;

            const wrap = document.createElement('div');
            wrap.className = 'code-block-wrap';
            pre.parentNode.insertBefore(wrap, pre);
            wrap.appendChild(pre);

            const btn = document.createElement('button');
            btn.className = 'code-copy-btn';
            btn.type = 'button';
            btn.setAttribute('aria-label', 'Copy code to clipboard');
            btn.innerHTML = COPY_ICON + '<span>Copy</span>';
            btn.addEventListener('click', () => {
                const text = pre.innerText.replace(/\n$/, '');
                const done = () => {
                    btn.classList.add('copied');
                    btn.querySelector('span').textContent = 'Copied';
                    setTimeout(() => {
                        btn.classList.remove('copied');
                        btn.querySelector('span').textContent = 'Copy';
                    }, 1800);
                };
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(done).catch(() => {});
                }
            });
            wrap.appendChild(btn);
        });
    }

    // =========================================================================
    // Series navigation (Previous / Next cards above the footer)
    // =========================================================================

    function initSeriesNav(footer) {
        const page = getCurrentPage();
        for (const [seriesName, pages] of Object.entries(SERIES)) {
            const idx = pages.findIndex(([href]) => href === page);
            if (idx === -1) continue;

            const nav = document.createElement('nav');
            nav.className = 'series-nav';
            nav.setAttribute('aria-label', seriesName + ' navigation');

            const prev = pages[idx - 1];
            const next = pages[idx + 1];

            if (prev) {
                nav.innerHTML += `
                    <a class="series-nav-link prev" href="${prev[0]}">
                        <span class="series-nav-label">← Previous · ${escapeHtml(seriesName)}</span>
                        <span class="series-nav-title">${escapeHtml(prev[1])}</span>
                    </a>`;
            }
            if (next) {
                nav.innerHTML += `
                    <a class="series-nav-link next" href="${next[0]}">
                        <span class="series-nav-label">Next · ${escapeHtml(seriesName)} →</span>
                        <span class="series-nav-title">${escapeHtml(next[1])}</span>
                    </a>`;
            }

            if (nav.children.length) {
                document.body.insertBefore(nav, footer);
            }
            return;
        }
    }

    // =========================================================================
    // Page meta chips: audience tier + estimated reading time
    // =========================================================================

    function initPageMetaChips() {
        const page = getCurrentPage();
        const tier = PAGE_TIERS[page];
        if (!tier) return;
        if (document.querySelector('.page-meta-chips')) return;

        const h1 = document.querySelector('h1');
        if (!h1) return;

        const content = document.querySelector('.main-content, main') || document.body;
        const words = (content.textContent || '').trim().split(/\s+/).length;
        const minutes = Math.max(1, Math.round(words / 220));

        const chips = document.createElement('div');
        chips.className = 'page-meta-chips';
        chips.innerHTML = `
            <span class="meta-chip tier-${tier}">${TIER_LABELS[tier]}</span>
            <span class="meta-chip">≈ ${minutes} min read</span>`;
        h1.insertAdjacentElement('afterend', chips);
    }

    // =========================================================================
    // Glossary tooltips — auto-link the first occurrence of glossary terms
    // =========================================================================

    function initGlossaryTerms() {
        const page = getCurrentPage();
        if (page === 'glossary.html' || page === 'index.html') return;

        fetch('shared/glossary.json')
            .then(r => (r.ok ? r.json() : []))
            .then(terms => {
                if (!Array.isArray(terms) || !terms.length) return;
                // Longest terms first so "credible interval" wins over "interval".
                terms.sort((a, b) => b.term.length - a.term.length);

                const root = document.querySelector('.main-content, main') || document.body;
                const paragraphs = Array.from(root.querySelectorAll('p')).slice(0, 80)
                    .filter(p => !p.closest('pre, code, a, .no-gloss, nav, footer, .sidebar'));

                const linked = new Set();
                let total = 0;
                const MAX_LINKS = 10;

                for (const { id, term, def } of terms) {
                    if (total >= MAX_LINKS) break;
                    if (linked.has(term.toLowerCase())) continue;
                    const re = new RegExp('\\b(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')\\b', 'i');

                    for (const p of paragraphs) {
                        const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT, {
                            acceptNode(node) {
                                if (node.parentElement.closest('a, code, pre, .gloss-term, .katex')) {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return re.test(node.nodeValue)
                                    ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
                            }
                        });
                        const node = walker.nextNode();
                        if (!node) continue;

                        const m = node.nodeValue.match(re);
                        const before = node.nodeValue.slice(0, m.index);
                        const after = node.nodeValue.slice(m.index + m[1].length);
                        const link = document.createElement('a');
                        link.className = 'gloss-term';
                        link.href = 'glossary.html#' + id;
                        link.setAttribute('data-tip', def);
                        link.textContent = m[1];
                        const frag = document.createDocumentFragment();
                        if (before) frag.appendChild(document.createTextNode(before));
                        frag.appendChild(link);
                        if (after) frag.appendChild(document.createTextNode(after));
                        node.parentNode.replaceChild(frag, node);

                        linked.add(term.toLowerCase());
                        total += 1;
                        break;
                    }
                }
            })
            .catch(() => { /* glossary tooltips are progressive enhancement */ });
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

        // Series prev/next cards sit directly above the footer
        initSeriesNav(footer);

        // Initialize features
        initScrollAnimations();
        initSidebarActiveState();
        initCollapsibleSidebar();
        initPlotHeadroomGuard();
        initSearchShortcuts();
        initCopyButtons();
        initPageMetaChips();
        initGlossaryTerms();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();