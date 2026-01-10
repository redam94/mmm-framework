/**
 * MMM Framework - Shared Components
 * Dynamically injects navigation and footer into pages
 */

(function() {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================
    const NAV_LINKS = [
        { href: 'index.html', label: 'Home' },
        { href: 'getting-started.html', label: 'Getting Started' },
        { href: 'about.html', label: 'About' },
        { href: 'variable-selection.html', label: 'Variable Selection' },
        { href: 'technical-guide.html', label: 'Technical Guide' },
        { href: 'bayesian-workflow.html', label: 'Bayesian Workflow' },
        { href: 'causal-inference.html', label: 'Causal Inference' },
        { href: 'scientific-modeling.html', label: 'Scientific Modeling' },
        { href: 'faq.html', label: 'FAQ' },
        { href: 'https://github.com/redam94/mmm-framework', label: 'GitHub', external: true }
    ];

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
    function isActive(href) {
        const currentPage = getCurrentPage();
        // Handle both exact match and index.html for root
        if (href === 'index.html' && (currentPage === '' || currentPage === 'index.html')) {
            return true;
        }
        return currentPage === href;
    }

    // =========================================================================
    // Navigation Component
    // =========================================================================
    
    function createNavigation() {
        const nav = document.createElement('nav');
        
        const linksHtml = NAV_LINKS.map(link => {
            const activeClass = isActive(link.href) ? ' class="active"' : '';
            const target = link.external ? ' target="_blank"' : '';
            return `<li><a href="${link.href}"${activeClass}${target}>${link.label}</a></li>`;
        }).join('\n                ');

        nav.innerHTML = `
        <div class="container nav-content">
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

        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                menuBtn.classList.remove('active');
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
        <div class="container footer-content">
            <span class="logo">MMM Framework</span>
            <div class="footer-links">
                ${linksHtml}
            </div>
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
    // Initialize
    // =========================================================================
    
    function init() {
        // Insert navigation at the start of body
        const nav = createNavigation();
        document.body.insertBefore(nav, document.body.firstChild);

        // Insert footer at the end of body
        const footer = createFooter();
        document.body.appendChild(footer);

        // Initialize features
        initScrollAnimations();
        initSidebarActiveState();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();