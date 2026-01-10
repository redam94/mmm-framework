/**
 * MMM Framework - Shared Components
 * Dynamically injects navigation and footer into pages
 * Includes mobile hamburger menu and collapsible sidebar
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

    /**
     * Check if we're on mobile/tablet
     */
    function isMobile() {
        return window.innerWidth <= 1024;
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
            <button class="mobile-menu-btn" aria-label="Toggle menu" aria-expanded="false">
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
            const isExpanded = navLinks.classList.toggle('active');
            menuBtn.classList.toggle('active');
            menuBtn.setAttribute('aria-expanded', isExpanded);
            
            // Prevent body scroll when menu is open
            document.body.classList.toggle('menu-open', isExpanded);
        });

        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                menuBtn.classList.remove('active');
                menuBtn.setAttribute('aria-expanded', 'false');
                document.body.classList.remove('menu-open');
            });
        });

        // Close menu on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && navLinks.classList.contains('active')) {
                navLinks.classList.remove('active');
                menuBtn.classList.remove('active');
                menuBtn.setAttribute('aria-expanded', 'false');
                document.body.classList.remove('menu-open');
            }
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
    // Sidebar Toggle Component (for pages with sidebar)
    // =========================================================================
    
    function createSidebarToggle() {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        // Create toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'sidebar-toggle';
        toggleBtn.setAttribute('aria-label', 'Toggle table of contents');
        toggleBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
        `;

        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';

        // Add to DOM
        document.body.appendChild(toggleBtn);
        document.body.appendChild(overlay);

        // Toggle sidebar
        function toggleSidebar() {
            const isActive = sidebar.classList.toggle('active');
            overlay.classList.toggle('active', isActive);
            document.body.classList.toggle('menu-open', isActive);
            
            // Update button icon
            if (isActive) {
                toggleBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                `;
            } else {
                toggleBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="3" y1="12" x2="21" y2="12"></line>
                        <line x1="3" y1="6" x2="21" y2="6"></line>
                        <line x1="3" y1="18" x2="21" y2="18"></line>
                    </svg>
                `;
            }
        }

        toggleBtn.addEventListener('click', toggleSidebar);
        overlay.addEventListener('click', toggleSidebar);

        // Close sidebar when clicking a link (on mobile)
        sidebar.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                if (isMobile()) {
                    sidebar.classList.remove('active');
                    overlay.classList.remove('active');
                    document.body.classList.remove('menu-open');
                    toggleBtn.innerHTML = `
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="3" y1="12" x2="21" y2="12"></line>
                            <line x1="3" y1="6" x2="21" y2="6"></line>
                            <line x1="3" y1="18" x2="21" y2="18"></line>
                        </svg>
                    `;
                }
            });
        });

        // Close on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && sidebar.classList.contains('active')) {
                toggleSidebar();
            }
        });

        // Handle resize - close sidebar when returning to desktop
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (!isMobile() && sidebar.classList.contains('active')) {
                    sidebar.classList.remove('active');
                    overlay.classList.remove('active');
                    document.body.classList.remove('menu-open');
                }
            }, 100);
        });
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
    // Smooth Scroll for Anchor Links
    // =========================================================================
    
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    e.preventDefault();
                    const navHeight = document.querySelector('nav')?.offsetHeight || 0;
                    const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;
                    
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
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

        // Initialize sidebar toggle for pages with sidebars
        createSidebarToggle();

        // Initialize features
        initScrollAnimations();
        initSidebarActiveState();
        initSmoothScroll();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();