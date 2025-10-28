/**
 * GIFT Framework v2.0 - Dashboard Tab Controller
 * Handles tab switching and terminal interface interactions
 */

(function() {
    'use strict';
    
    // Wait for DOM to load
    document.addEventListener('DOMContentLoaded', initDashboard);
    
    function initDashboard() {
        setupTabNavigation();
        setupKeyboardShortcuts();
        logInitialization();
    }
    
    /**
     * Setup tab navigation system
     */
    function setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                const targetTab = this.getAttribute('data-tab');
                switchTab(targetTab, tabButtons, tabContents);
            });
        });
    }
    
    /**
     * Switch active tab
     */
    function switchTab(targetTab, tabButtons, tabContents) {
        // Remove active class from all buttons and contents
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to selected button and content
        const selectedButton = document.querySelector(`[data-tab="${targetTab}"]`);
        const selectedContent = document.getElementById(targetTab);
        
        if (selectedButton && selectedContent) {
            selectedButton.classList.add('active');
            selectedContent.classList.add('active');
            
            // Log tab switch
            console.log(`[GIFT] Tab switched to: ${targetTab.toUpperCase()}`);
        }
    }
    
    /**
     * Setup keyboard shortcuts for tab navigation
     * 1, 2, 3 keys for quick tab switching
     */
    function setupKeyboardShortcuts() {
        const tabMap = {
            '1': 'e8-roots',
            '2': 'precision',
            '3': 'reduction'
        };
        
        document.addEventListener('keydown', function(event) {
            // Only trigger if not typing in an input field
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                return;
            }
            
            const targetTab = tabMap[event.key];
            if (targetTab) {
                event.preventDefault();
                const tabButtons = document.querySelectorAll('.tab-button');
                const tabContents = document.querySelectorAll('.tab-content');
                switchTab(targetTab, tabButtons, tabContents);
            }
        });
        
        console.log('[GIFT] Keyboard shortcuts enabled: Press 1, 2, or 3 to switch tabs');
    }
    
    /**
     * Log initialization to console
     */
    function logInitialization() {
        console.log('========================================');
        console.log('GIFT FRAMEWORK v2.0 - VISUALIZATION TERMINAL');
        console.log('========================================');
        console.log('System: INITIALIZED');
        console.log('Visualizations: 3 LOADED');
        console.log('Precision: 0.13% MEAN DEVIATION');
        console.log('Observables: 34 TRACKED');
        console.log('========================================');
        console.log('Commands:');
        console.log('  Press 1: E8 Root System');
        console.log('  Press 2: Precision Analysis');
        console.log('  Press 3: Dimensional Reduction Flow');
        console.log('========================================');
    }
    
    /**
     * Optional: Add performance monitoring
     */
    window.addEventListener('load', function() {
        const loadTime = performance.now();
        console.log(`[GIFT] Dashboard loaded in ${loadTime.toFixed(2)}ms`);
    });
    
    /**
     * Optional: Monitor iframe load status
     */
    function monitorIframeLoading() {
        const iframes = document.querySelectorAll('.viz-frame');
        
        iframes.forEach((iframe, index) => {
            iframe.addEventListener('load', function() {
                console.log(`[GIFT] Visualization ${index + 1} loaded successfully`);
            });
            
            iframe.addEventListener('error', function() {
                console.error(`[GIFT] Error loading visualization ${index + 1}`);
            });
        });
    }
    
    // Initialize iframe monitoring
    monitorIframeLoading();
    
})();

