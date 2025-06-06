// Global variables
let currentPortfolio = null;
let processingModal = null;
let recommendationsModal = null;
let chatbotOpen = false;
let chatbotMinimized = false;

document.addEventListener('DOMContentLoaded', function () {
    // Add Chart.js if not already present
    if (!document.querySelector('script[src*="chart.js"]')) {
        const chartScript = document.createElement('script');
        chartScript.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        document.head.appendChild(chartScript);
    }

    // Verify that bootstrap is available
    if (typeof bootstrap === 'undefined') {
        console.error("Bootstrap library is not loaded! Adding it now...");
        const bootstrapScript = document.createElement('script');
        bootstrapScript.src = 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js';
        document.head.appendChild(bootstrapScript);
        
        // Wait for it to load before initializing modals
        bootstrapScript.onload = function() {
            console.log("Bootstrap loaded dynamically, initializing modals...");
            initializeModals();
        };
    } else {
        // Initialize modals
        initializeModals();
    }

    // Initialize interactive elements
    initializeTooltips();
    initializeFormBehavior();
    setupAnimations();
    initializeChatbot();

    // Set up the form submission event
    const form = document.getElementById('investmentForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    // Set up action buttons
    setupActionButtons();
});

/**
 * Initialize modals
 */
function initializeModals() {
    try {
        const processingModalEl = document.getElementById('processingModal');
        if (processingModalEl) {
            processingModal = new bootstrap.Modal(processingModalEl, {
                backdrop: 'static',
                keyboard: false
            });
            console.log("Processing modal initialized successfully");
        } else {
            console.error("Processing modal element not found");
        }

        const recommendationsModalEl = document.getElementById('recommendationsModal');
        if (recommendationsModalEl) {
            recommendationsModal = new bootstrap.Modal(recommendationsModalEl);
            console.log("Recommendations modal initialized successfully");
        } else {
            console.error("Recommendations modal element not found");
        }
    } catch (error) {
        console.error("Error initializing modals:", error);
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Set up animations and interactive effects
 */
function setupAnimations() {
    // Feature rings hover effect
    const featureRings = document.querySelectorAll('.feature-ring');
    featureRings.forEach(ring => {
        ring.addEventListener('mouseenter', function () {
            const icon = this.querySelector('i');
            icon.classList.add('animated');
            setTimeout(() => {
                icon.classList.remove('animated');
            }, 1000);
        });
    });

    // Card hover effects
    const cards = document.querySelectorAll('.app-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function () {
            this.classList.add('card-hover');
        });

        card.addEventListener('mouseleave', function () {
            this.classList.remove('card-hover');
        });
    });
}

/**
 * Initialize chatbot functionality
 */
function initializeChatbot() {
    document.addEventListener('DOMContentLoaded', function () {
        // Simple direct approach to get elements
        var chatToggle = document.getElementById('chatbot-toggle');
        var chatContainer = document.getElementById('chatbot-container');
        var closeChat = document.getElementById('close-chat');
        var sendButton = document.getElementById('send-message');
        var messageInput = document.getElementById('message-input');
        var messagesContainer = document.getElementById('chatbot-messages');

        console.log("Chat elements check:", {
            toggle: chatToggle !== null,
            container: chatContainer !== null,
            close: closeChat !== null,
            send: sendButton !== null,
            input: messageInput !== null,
            messages: messagesContainer !== null
        });

        if (!chatToggle || !chatContainer) {
            console.error("Critical chatbot elements missing");
            return;
        }

        // Add welcome message if container exists
        if (messagesContainer && messagesContainer.children.length === 0) {
            var welcomeMsg = document.createElement('div');
            welcomeMsg.className = 'chatbot-message bot';
            welcomeMsg.innerHTML = '<div class="message-content bot-message">Welcome to the Investment Assistant! How can I help you with your investment portfolio today?</div>';
            messagesContainer.appendChild(welcomeMsg);
        }

        // Super simple toggle approach - just add/remove display class
        chatToggle.onclick = function (e) {
            console.log("Chat toggle clicked!");
            chatContainer.classList.toggle('d-none');
            if (!chatContainer.classList.contains('d-none') && messageInput) {
                // Focus the input field when opened
                setTimeout(function () {
                    messageInput.focus();
                }, 100);
            }
            e.preventDefault();
        };

        // Close button
        if (closeChat) {
            closeChat.onclick = function () {
                console.log("Close chat clicked!");
                chatContainer.classList.add('d-none');
            };
        }

        // Send message functionality
        function sendChatMessage() {
            console.log("Send message function called");
            if (!messageInput || !messagesContainer) return;

            var message = messageInput.value.trim();
            if (!message) return;

            // Create and add user message
            var userMsg = document.createElement('div');
            userMsg.className = 'chatbot-message user';
            userMsg.innerHTML = '<div class="message-content user-message">' + message + '</div>';
            messagesContainer.appendChild(userMsg);

            // Clear input
            messageInput.value = '';

            // Add typing indicator
            var indicatorMsg = document.createElement('div');
            indicatorMsg.className = 'chatbot-message bot typing-indicator';
            indicatorMsg.innerHTML = '<div class="typing-dots"><span>.</span><span>.</span><span>.</span></div>';
            messagesContainer.appendChild(indicatorMsg);

            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            // Make API call to LLM backend
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
                .then(function (response) {
                    if (!response.ok) {
                        throw new Error('Chat API error: ' + response.status);
                    }
                    return response.json();
                })
                .then(function (data) {
                    // Remove typing indicator
                    messagesContainer.removeChild(indicatorMsg);

                    // Add bot response from API
                    var botMsg = document.createElement('div');
                    botMsg.className = 'chatbot-message bot';
                    botMsg.innerHTML = '<div class="message-content bot-message">' + data.response + '</div>';
                    messagesContainer.appendChild(botMsg);

                    // Scroll to bottom again
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                })
                .catch(function (error) {
                    console.error('Chat API error:', error);

                    // Remove typing indicator
                    messagesContainer.removeChild(indicatorMsg);

                    // Fall back to local response only if API fails
                    var fallbackMsg = document.createElement('div');
                    fallbackMsg.className = 'chatbot-message bot';
                    fallbackMsg.innerHTML = '<div class="message-content bot-message">Sorry, I\'m having trouble connecting to the server. Please try again later.</div>';
                    messagesContainer.appendChild(fallbackMsg);

                    // Scroll to bottom
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                });
        }

        // Send button click
        if (sendButton) {
            sendButton.onclick = function () {
                console.log("Send button clicked!");
                sendChatMessage();
            };
        }

        // Enter key in input field
        if (messageInput) {
            messageInput.onkeypress = function (e) {
                if (e.key === 'Enter') {
                    console.log("Enter pressed in chat input!");
                    sendChatMessage();
                    e.preventDefault();
                }
            };
        }
    });
}

// Call this immediately to set up event listeners
initializeChatbot();

/**
 * Get a local response for the chatbot when API fails
 * @param {string} message - The user's message
 * @returns {string} A local response
 */
function getLocalChatResponse(message) {
    message = message.toLowerCase();

    if (message.includes('hello') || message.includes('hi ') || message.includes('hey')) {
        return "Hello! How can I help you with your investment portfolio today?";
    } else if (message.includes('risk') || message.includes('tolerance')) {
        return "Risk tolerance reflects how much volatility you can handle. Conservative investors prefer stability, while aggressive investors can tolerate larger swings for potentially higher returns.";
    } else if (message.includes('fund') || message.includes('ticker')) {
        return "Investment funds hold assets like stocks, bonds, or commodities and often track an index. They provide diversification with a single investment.";
    } else if (message.includes('rebalance') || message.includes('rebalancing')) {
        return "Rebalancing is the process of realigning the weightings of your portfolio assets to maintain your desired level of asset allocation.";
    } else if (message.includes('allocation') || message.includes('diversif')) {
        return "Asset allocation is the strategy of dividing investments among different asset categories like stocks, bonds, and cash to optimize risk/return based on your goals and risk tolerance.";
    } else if (message.includes('stock') || message.includes('equity')) {
        return "Stocks represent ownership in a company. They typically offer higher potential returns but come with higher volatility compared to bonds or cash.";
    } else if (message.includes('bond')) {
        return "Bonds are loans to companies or governments that pay interest. They're generally less volatile than stocks but offer lower potential returns.";
    } else if (message.includes('thank')) {
        return "You're welcome! Feel free to ask if you have any other questions about investing.";
    } else {
        return "I'm not sure I understand. You can ask me about risk tolerance, investment funds, asset allocation, stocks, bonds, or rebalancing strategies.";
    }
}

/**
 * Initialize form field behaviors
 */
function initializeFormBehavior() {
    // Add animation class to inputs when they receive focus
    const formInputs = document.querySelectorAll('.form-control, .form-select');

    formInputs.forEach(input => {
        // Input animation
        input.addEventListener('focus', function () {
            this.classList.add('animate-input');
            this.closest('.app-input-group').classList.add('input-focus');
        });

        input.addEventListener('blur', function () {
            this.closest('.app-input-group').classList.remove('input-focus');
        });

        // Add "has-value" class when input has a value
        input.addEventListener('input', function () {
            if (this.value.trim() !== '') {
                this.classList.add('has-value');
            } else {
                this.classList.remove('has-value');
            }
        });

        // Check initial value
        if (input.value.trim() !== '') {
            input.classList.add('has-value');
        }
    });

    // Add click animation to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('mousedown', function () {
            this.classList.add('btn-clicked');
        });

        button.addEventListener('mouseup', function () {
            this.classList.remove('btn-clicked');
        });

        button.addEventListener('mouseleave', function () {
            this.classList.remove('btn-clicked');
        });
    });
}

/**
 * Set up action buttons for the portfolio results
 */
function setupActionButtons() {
    console.log("Setting up action buttons");
    
    // View Details Button - We'll set up the event listener when the portfolio is loaded
    const viewDetailsBtn = document.getElementById('viewDetailsBtn');
    if (viewDetailsBtn) {
        console.log("Found view details button:", viewDetailsBtn);
        // We'll attach the event listener in displayPortfolioResults
    } else {
        console.error("viewDetailsBtn element not found during setup");
    }

    // New Simulation Button
    const newSimulationBtn = document.getElementById('newSimulationBtn');
    if (newSimulationBtn) {
        newSimulationBtn.addEventListener('click', function() {
            console.log("New simulation button clicked");
            // Reset UI state
            document.getElementById('initialState').classList.remove('d-none');
            document.getElementById('portfolioResults').classList.add('d-none');
            
            // Clear form if needed
            const form = document.getElementById('investmentForm');
            if (form) {
                form.reset();
            }
            
            // Scroll back to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // Debug button for modal testing
    if (process.env.NODE_ENV === 'development') {
        const debugBtn = document.createElement('button');
        debugBtn.textContent = 'Test Modal';
        debugBtn.className = 'btn btn-sm btn-warning position-fixed bottom-0 end-0 m-2';
        debugBtn.addEventListener('click', function() {
            if (recommendationsModal) {
                console.log("Testing recommendations modal");
                recommendationsModal.show();
            } else {
                console.error("recommendationsModal is not initialized");
                // Try initializing it now
                const recommendationsModalEl = document.getElementById('recommendationsModal');
                if (recommendationsModalEl) {
                    recommendationsModal = new bootstrap.Modal(recommendationsModalEl);
                    recommendationsModal.show();
                } else {
                    console.error("Modal element not found");
                    alert("Modal element not found!");
                }
            }
        });
        document.body.appendChild(debugBtn);
    }
}

/**
 * Handle the form submission
 * @param {Event} event - The form submission event
 */
function handleFormSubmit(event) {
    event.preventDefault();

    // Validate the form
    if (!validateForm()) {
        // Shake the invalid inputs
        return;
    }

    // Add loading state
    const submitButton = event.target.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="spinner"></span><span>Processing...</span>';

    // Get form data
    const formData = getFormData();

    // Show ML processing modal
    processingModal.show();

    // Simulate ML model processing
    simulateMLProcessing(formData).then(portfolio => {
        // Hide processing modal
        processingModal.hide();

        // Hide initial state with fade out
        const initialState = document.getElementById('initialState');
        initialState.classList.add('fade-out');

        setTimeout(() => {
            initialState.classList.add('d-none');
            initialState.classList.remove('fade-out');

            // Show results with animation
            currentPortfolio = portfolio;
            displayPortfolioResults(portfolio);

            // Reset button state
            submitButton.disabled = false;
            submitButton.innerHTML = '<span class="btn-text">Generate Portfolio</span><i class="bi bi-arrow-right"></i>';
        }, 300);
    }).catch(error => {
        console.error('Error generating portfolio:', error);

        // Hide processing modal
        processingModal.hide();

        // Show error
        const errorAlert = document.getElementById('errorAlert');
        errorAlert.classList.remove('d-none');
        errorAlert.textContent = 'An error occurred while generating your portfolio. Please try again.';

        // Hide error after 5 seconds
        setTimeout(() => {
            errorAlert.classList.add('d-none');
        }, 5000);

        // Show initial state again
        initialState.classList.remove('d-none');
    });
}

/**
 * Simulate ML processing with progress updates
 * @param {Object} formData - The form data
 * @returns {Promise} A promise that resolves with the generated portfolio
 */
function simulateMLProcessing(formData) {
    return new Promise((resolve, reject) => {
        const progressBar = document.getElementById('processingProgress');
        const statusText = document.getElementById('processing-status');

        if (!progressBar || !statusText) {
            console.error("Processing elements not found");
            reject(new Error("UI elements not found"));
            return;
        }

        // Reset progress
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        statusText.textContent = 'Initializing portfolio generation...';

        // Prepare data for API call
        const apiData = {
            salary: formData.annualIncome,
            investment_goals: formData.investmentGoal,
            risk_tolerance: formData.riskTolerance,
            time_horizon: formData.timeHorizon,
            investment_amount: formData.investmentAmount
        };

        console.log('API data being sent:', JSON.stringify(apiData));

        // Update initial progress
        progressBar.style.width = '10%';
        progressBar.setAttribute('aria-valuenow', 10);

        // Portfolio generation steps
        const steps = [
            { percent: 15, message: 'Loading market data...' },
            { percent: 30, message: 'Analyzing market trends...' },
            { percent: 45, message: 'Calculating risk metrics...' },
            { percent: 60, message: 'Optimizing asset allocation...' },
            { percent: 75, message: 'Determining optimal diversification...' },
            { percent: 90, message: 'Generating investment recommendations...' }
        ];

        let currentStep = 0;
        const totalSteps = steps.length;

        // Process steps recursively
        const processStep = function () {
            // Update progress bar with current step
            if (currentStep < totalSteps) {
                const step = steps[currentStep];
                progressBar.style.width = step.percent + '%';
                progressBar.setAttribute('aria-valuenow', step.percent);
                statusText.textContent = step.message;

                currentStep++;
                setTimeout(processStep, 800);
            } else {
                // When all steps are done, call the API
                statusText.textContent = 'Finalizing portfolio...';

                // Make API call
                fetch('/portfolio', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(apiData)
                })
                    .then(response => {
                        console.log('API response status:', response.status);
                        if (!response.ok) {
                            return response.json().then(data => {
                                console.error('API error response:', data);
                                throw new Error(data.error || 'Error generating portfolio');
                            });
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Portfolio data received:', data);

                        // Update progress to 100%
                        progressBar.style.width = '100%';
                        progressBar.setAttribute('aria-valuenow', 100);
                        statusText.textContent = 'Portfolio generation complete!';

                        // Check if data contains expected properties
                        if (!data.user_profile || !data.asset_allocation || !data.specific_allocations) {
                            console.error('Invalid portfolio data structure:', data);
                            throw new Error('Invalid portfolio data structure received from server');
                        }

                        // Convert backend data format to frontend format
                        const portfolio = transformBackendDataToPortfolio(data);
                        console.log('Transformed portfolio:', portfolio);

                        // Wait a moment before resolving to show completion
                        setTimeout(() => {
                            resolve(portfolio);
                        }, 1000);
                    })
                    .catch(error => {
                        console.error('API error:', error);
                        statusText.textContent = 'Error: ' + error.message;
                        reject(error);
                    });
            }
        };

        // Start processing
        setTimeout(processStep, 800);
    });
}

/**
 * Transform backend data into the format expected by the frontend
 * @param {Object} backendData - Data from the backend API
 * @returns {Object} Transformed portfolio data
 */
function transformBackendDataToPortfolio(backendData) {
    console.log("Transforming backend data:", backendData);

    // Handle error or empty response
    if (!backendData || typeof backendData !== 'object') {
        console.error("Invalid backend data:", backendData);
        return {
            profile: {
                investmentAmount: 10000,
                investmentGoal: 'wealth',
                timeHorizon: 'medium',
                riskScore: 50
            },
            allocation: [
                { id: 'stocks', name: 'Stocks', percentage: 40, amount: 4000, color: getAssetColor('stocks') },
                { id: 'bonds', name: 'Bonds', percentage: 30, amount: 3000, color: getAssetColor('bonds') },
                { id: 'crypto', name: 'Crypto', percentage: 20, amount: 2000, color: getAssetColor('crypto') },
                { id: 'gold', name: 'Gold', percentage: 10, amount: 1000, color: getAssetColor('gold') }
            ],
            recommendations: sampleRecommendations()
        };
    }

    try {
        // Standardize backend data format for frontend consumption
        let portfolio = {};

        // Extract user profile data
        const userProfile = backendData.user_profile || {};
        portfolio.investment_amount = Number(userProfile.investment_amount) || 10000;
        portfolio.annual_income = Number(userProfile.salary) || 0; // Use salary field from user profile for annual income
        portfolio.investment_goal = userProfile.investment_goals || 'wealth';
        portfolio.time_horizon = userProfile.time_horizon || 'medium';
        portfolio.risk_score = Number(userProfile.risk_score) || 0.5;

        // Extract and normalize asset allocation
        const assetAllocation = backendData.asset_allocation || {};
        portfolio.asset_allocation = {};

        // Ensure asset_allocation is an object with expected structure
        Object.keys(assetAllocation).forEach(assetClass => {
            const assetData = assetAllocation[assetClass];
            portfolio.asset_allocation[assetClass] = {
                percentage: Number(assetData.percentage) || 0,
                amount: Number(assetData.amount) || 0
            };
        });

        // If asset_allocation is empty, populate with default values
        if (Object.keys(portfolio.asset_allocation).length === 0) {
            portfolio.asset_allocation = {
                stocks: { percentage: 0.4, amount: portfolio.investment_amount * 0.4 },
                bonds: { percentage: 0.3, amount: portfolio.investment_amount * 0.3 },
                crypto: { percentage: 0.2, amount: portfolio.investment_amount * 0.2 },
                gold: { percentage: 0.1, amount: portfolio.investment_amount * 0.1 }
            };
        }

        // Extract specific investment recommendations
        const specificAllocations = backendData.specific_allocations || {};
        portfolio.recommendations = [];
        
        // Log for debugging
        console.log("Processing specific allocations:", specificAllocations);
        
        // Process each asset class
        Object.keys(specificAllocations).forEach(assetClass => {
            const assetData = specificAllocations[assetClass];
            console.log(`Processing ${assetClass}:`, assetData);
            
            // Check if allocations exist as a sub-property (common pattern in the API)
            if (assetData && assetData.allocations && typeof assetData.allocations === 'object') {
                const allocations = assetData.allocations;
                const nameMap = assetData.names || {};
                
                // Extract actual allocation entries (key-value pairs where value is a number)
                Object.entries(allocations).forEach(([ticker, amount]) => {
                    // Skip metadata entries (where amount is an object, not a number)
                    if (typeof amount === 'number') {
                        portfolio.recommendations.push({
                            ticker: ticker,
                            name: nameMap[ticker] || `${assetClass.charAt(0).toUpperCase() + assetClass.slice(1)} Investment`,
                            type: assetClass,
                            amount: amount
                        });
                    }
                });
            }
            // Handle case where investments is an array
            else if (Array.isArray(assetData)) {
                assetData.forEach(investment => {
                    if (typeof investment === 'object' && investment.ticker && typeof investment.amount === 'number') {
                        portfolio.recommendations.push({
                            ticker: investment.ticker,
                            name: investment.name || `${investment.ticker} Fund`,
                            type: assetClass,
                            amount: investment.amount
                        });
                    }
                });
            }
        });

        // If no valid recommendations found, use sample recommendations
        if (portfolio.recommendations.length === 0) {
            console.log("No valid recommendations found, using samples");
            portfolio.recommendations = sampleRecommendations();
        }

        console.log("Transformed portfolio:", portfolio);
        return portfolio;
    } catch (error) {
        console.error("Error transforming backend data:", error);
        
        // Return fallback data
        return {
            investment_amount: 10000,
            investment_goal: 'wealth',
            time_horizon: 'medium',
            risk_score: 0.5,
            asset_allocation: {
                stocks: { percentage: 0.4, amount: 4000 },
                bonds: { percentage: 0.3, amount: 3000 },
                crypto: { percentage: 0.2, amount: 2000 },
                gold: { percentage: 0.1, amount: 1000 }
            },
            recommendations: sampleRecommendations()
        };
    }
}

/**
 * Generate sample recommendations for fallback
 * @returns {Array} Sample recommendations
 */
function sampleRecommendations() {
    return [
        { ticker: 'VTI', name: 'Vanguard Total Stock Market ETF', type: 'stocks', amount: 2000 },
        { ticker: 'VXUS', name: 'Vanguard Total International Stock ETF', type: 'stocks', amount: 2000 },
        { ticker: 'BND', name: 'Vanguard Total Bond Market ETF', type: 'bonds', amount: 3000 },
        { ticker: 'BTC', name: 'Bitcoin', type: 'crypto', amount: 2000 },
        { ticker: 'GLD', name: 'SPDR Gold Shares', type: 'gold', amount: 1000 }
    ];
}

/**
 * Get color for asset class
 * @param {string} assetClass - The asset class name
 * @returns {string} CSS color value
 */
function getAssetColor(assetClass) {
    const colorMap = {
        'stocks': '#4C6FFF',
        'bonds': '#FF8F56',
        'crypto': '#82C43C',
        'gold': '#FFC107',
        'alternatives': '#9D50FF',
        'cash': '#36CABC'
    };

    return colorMap[assetClass.toLowerCase()] || '#777777';
}

/**
 * Get risk description based on risk score
 * @param {number} riskScore - The calculated risk score
 * @returns {string} Description of the risk profile
 */
function getRiskDescription(riskScore) {
    if (riskScore < 30) {
        return "Your conservative risk profile prioritizes capital preservation with modest growth. This approach minimizes volatility but may limit long-term returns.";
    } else if (riskScore < 70) {
        return "Your balanced risk profile seeks moderate growth while managing volatility. This approach aims for steady returns through market cycles with moderate drawdowns during corrections.";
    } else {
        return "Your aggressive risk profile prioritizes maximal growth potential. This approach accepts significant short-term volatility for potentially higher long-term returns.";
    }
}

/**
 * Validate the form inputs
 * @returns {boolean} Whether the form is valid
 */
function validateForm() {
    const form = document.getElementById('investmentForm');
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;

            // Add invalid styling
            input.classList.add('is-invalid');
            const inputGroup = input.closest('.app-input-group');
            inputGroup.classList.add('shake-animation');

            // Remove animation class after animation completes
            setTimeout(() => {
                inputGroup.classList.remove('shake-animation');
            }, 500);
        } else {
            input.classList.remove('is-invalid');
        }
    });

    return isValid;
}

/**
 * Get data from the form
 * @returns {Object} The form data
 */
function getFormData() {
    return {
        investmentAmount: parseFloat(document.getElementById('investmentAmount').value),
        annualIncome: parseFloat(document.getElementById('annualIncome').value),
        investmentGoal: document.getElementById('investmentGoal').value,
        riskTolerance: document.getElementById('riskTolerance').value,
        timeHorizon: document.getElementById('timeHorizon').value
    };
}

/**
 * Display the portfolio results
 * @param {Object} portfolio - The portfolio data
 */
function displayPortfolioResults(portfolio) {
    console.log("Displaying portfolio results:", portfolio);

    // Cache the current portfolio
    currentPortfolio = portfolio;

    // Hide initial state and show results
    document.getElementById('initialState').classList.add('d-none');
    document.getElementById('portfolioResults').classList.remove('d-none');

    // Update investor profile
    document.getElementById('profile-amount').textContent = `$${portfolio.investment_amount.toLocaleString()}`;
    document.getElementById('profile-goal').textContent = formatInvestmentGoal(portfolio.investment_goal);
    document.getElementById('profile-time').textContent = formatTimeHorizon(portfolio.time_horizon);

    // Update risk meter
    const riskScore = portfolio.risk_score * 100; // Convert to percentage
    const riskMeterFill = document.getElementById('risk-meter-fill');
    riskMeterFill.style.width = `${riskScore}%`;

    // Get risk level
    const riskLevel = getRiskLevel(riskScore);
    const riskLevelBadge = document.getElementById('risk-level-badge');
    riskLevelBadge.textContent = riskLevel;
    riskLevelBadge.className = `risk-badge ${getRiskClass(riskLevel)}`;

    // Update allocation chart and breakdown
    const allocation = Object.entries(portfolio.asset_allocation).map(([key, value]) => {
        return {
            id: key,
            name: key.charAt(0).toUpperCase() + key.slice(1),
            percentage: Math.round(value.percentage * 100),
            amount: value.amount,
            color: getAssetColor(key)
        };
    });

    createAllocationChart(allocation);
    updateAllocationBreakdown(allocation);

    // Create additional charts
    createHistoricalPerformanceChart(portfolio);
    createRiskReturnChart(portfolio);
    createProjectionChart(portfolio);

    // Update investment recommendations
    updateRecommendations(portfolio.recommendations);

    // Hook up the detailed recommendations button
    const viewDetailsBtn = document.getElementById('viewDetailsBtn');
    if (viewDetailsBtn) {
        // Remove any existing event listeners to prevent duplicates
        viewDetailsBtn.replaceWith(viewDetailsBtn.cloneNode(true));
        
        // Get the fresh reference after replacement
        const freshViewDetailsBtn = document.getElementById('viewDetailsBtn');
        
        freshViewDetailsBtn.addEventListener('click', function() {
            console.log("View details button clicked");
            displayDetailedRecommendations(portfolio);
        });
    } else {
        console.error("viewDetailsBtn element not found");
    }
}

/**
 * Update the allocation breakdown in the results section
 * @param {Array} allocation - The asset allocation
 */
function updateAllocationBreakdown(allocation) {
    const allocationList = document.getElementById('allocation-breakdown');

    if (!allocationList) {
        console.error("Allocation breakdown container not found");
        return;
    }

    allocationList.innerHTML = '';

    allocation.forEach(item => {
        const div = document.createElement('div');
        div.className = 'allocation-item';

        const headerDiv = document.createElement('div');
        headerDiv.className = 'allocation-header';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'asset-name';
        nameSpan.textContent = item.name;

        const percentSpan = document.createElement('span');
        percentSpan.className = 'asset-percentage';
        percentSpan.style.backgroundColor = item.color;
        percentSpan.textContent = Math.round(item.percentage) + '%';

        headerDiv.appendChild(nameSpan);
        headerDiv.appendChild(percentSpan);

        const amountDiv = document.createElement('div');
        amountDiv.className = 'asset-amount';
        amountDiv.textContent = '$' + item.amount.toLocaleString();

        div.appendChild(headerDiv);
        div.appendChild(amountDiv);

        allocationList.appendChild(div);
    });
}

/**
 * Update the recommendations list
 * @param {Array} recommendations - The investment recommendations
 */
function updateRecommendations(recommendations) {
    console.log("Updating recommendations:", recommendations);
    const recommendationsContainer = document.getElementById('investment-recommendations');

    if (!recommendationsContainer) {
        console.error("Recommendations container not found");
        return;
    }

    recommendationsContainer.innerHTML = '';

    // Create a table for recommendations
    const table = document.createElement('table');
    table.className = 'table table-hover';

    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    const headers = ['Ticker', 'Name', 'Type', 'Amount'];
    headers.forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');
    tbody.id = 'recommendationsList';

    // Ensure we have array of recommendations
    if (!Array.isArray(recommendations) || recommendations.length === 0) {
        recommendations = sampleRecommendations();
    }

    recommendations.forEach(rec => {
        const tr = document.createElement('tr');
        tr.className = 'recommendation-item';

        // Ticker Cell
        const tickerCell = document.createElement('td');
        const tickerBadge = document.createElement('span');
        tickerBadge.className = 'badge bg-secondary';
        // Make sure ticker is a string and not an object or array
        const ticker = typeof rec.ticker === 'string' ? rec.ticker : 
                      (rec.symbol || rec.id || 'UNKN');
        tickerBadge.textContent = ticker;
        tickerCell.appendChild(tickerBadge);

        // Name Cell
        const nameCell = document.createElement('td');
        nameCell.textContent = rec.name || 'Unknown Asset';

        // Type Cell with proper capitalization
        const typeCell = document.createElement('td');
        let assetType = rec.type || 'Other';
        // Capitalize first letter
        assetType = assetType.charAt(0).toUpperCase() + assetType.slice(1);
        typeCell.textContent = assetType;

        // Amount Cell
        const amountCell = document.createElement('td');
        amountCell.className = 'text-end';
        const amount = typeof rec.amount === 'number' ? rec.amount : 0;
        amountCell.textContent = '$' + amount.toLocaleString();

        tr.appendChild(tickerCell);
        tr.appendChild(nameCell);
        tr.appendChild(typeCell);
        tr.appendChild(amountCell);

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    recommendationsContainer.appendChild(table);
}

/**
 * Show the detailed recommendations modal
 * @param {Object} portfolio - The portfolio data
 */
function displayDetailedRecommendations(portfolio) {
    console.log("Displaying detailed recommendations:", portfolio);
    
    if (!portfolio) {
        console.error("No portfolio data available");
        return;
    }
    
    // Check if the modal HTML element exists
    const modalElement = document.getElementById('recommendationsModal');
    if (!modalElement) {
        console.error("Modal element not found in the DOM");
        alert("Unable to display detailed recommendations. Please try again.");
        return;
    }
    
    if (!recommendationsModal) {
        console.warn("Recommendations modal not initialized. Attempting to initialize it now.");
        try {
            recommendationsModal = new bootstrap.Modal(modalElement);
            console.log("Modal initialized successfully");
        } catch (error) {
            console.error("Error initializing modal:", error);
            
            // Fallback - if bootstrap isn't available or working, try creating a simple modal ourselves
            try {
                // Make the modal visible manually
                modalElement.style.display = 'block';
                modalElement.classList.add('show');
                document.body.classList.add('modal-open');
                
                // Create a backdrop if it doesn't exist
                let backdrop = document.querySelector('.modal-backdrop');
                if (!backdrop) {
                    backdrop = document.createElement('div');
                    backdrop.className = 'modal-backdrop fade show';
                    document.body.appendChild(backdrop);
                }
                
                // Add close handler to close button
                const closeButtons = modalElement.querySelectorAll('[data-bs-dismiss="modal"]');
                closeButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        modalElement.style.display = 'none';
                        modalElement.classList.remove('show');
                        document.body.classList.remove('modal-open');
                        if (backdrop && backdrop.parentNode) {
                            backdrop.parentNode.removeChild(backdrop);
                        }
                    });
                });
                
                console.log("Fallback modal display method used");
            } catch (e) {
                console.error("All modal display methods failed:", e);
                alert("Unable to display detailed recommendations. Please try again later.");
                return;
            }
        }
    }
    
    // First, update all the detailed sections
    const detailedProfile = document.getElementById('detailed-profile');
    const detailedStrategy = document.getElementById('detailed-strategy');
    const detailedAllocation = document.getElementById('detailed-allocation');
    const detailedEtfs = document.getElementById('detailed-etfs');
    const detailedOutlook = document.getElementById('detailed-outlook');
    const detailedRebalancing = document.getElementById('detailed-rebalancing');
    const detailedProjections = document.getElementById('detailed-projections');

    // Update profile section if it exists
    if (detailedProfile) {
        detailedProfile.innerHTML = `
            <p><strong>Investment Amount:</strong> $${portfolio.investment_amount.toLocaleString()}</p>
            <p><strong>Annual Income:</strong> ${portfolio.annual_income ? '$' + portfolio.annual_income.toLocaleString() : 'Not Provided'}</p>
            <p><strong>Investment Goal:</strong> ${formatInvestmentGoal(portfolio.investment_goal)}</p>
            <p><strong>Risk Tolerance:</strong> ${formatRiskTolerance(portfolio.risk_tolerance || getRiskLevel(portfolio.risk_score * 100))}</p>
            <p><strong>Time Horizon:</strong> ${formatTimeHorizon(portfolio.time_horizon)}</p>
        `;
    }

    if (detailedAllocation) {
        // Create allocation table
        let allocationHtml = '<table class="table table-sm"><thead><tr><th>Asset Class</th><th>Allocation</th><th>Amount</th></tr></thead><tbody>';

        Object.entries(portfolio.asset_allocation).forEach(([assetClass, data]) => {
            const color = getAssetColor(assetClass);
            const name = assetClass.charAt(0).toUpperCase() + assetClass.slice(1);
            const percentage = Math.round(data.percentage * 100);
            const amount = data.amount;
            
            allocationHtml += `
                <tr>
                    <td><span class="color-dot" style="background-color: ${color}"></span> ${name}</td>
                    <td>${percentage}%</td>
                    <td>$${amount.toLocaleString()}</td>
                </tr>
            `;
        });

        allocationHtml += '</tbody></table>';
        detailedAllocation.innerHTML = allocationHtml;
    }

    if (detailedEtfs) {
        // Group securities by type
        const securitiesByType = {};

        // Group the recommendations by type
        portfolio.recommendations.forEach(item => {
            const type = item.type || 'Other';
            if (!securitiesByType[type]) {
                securitiesByType[type] = [];
            }
            securitiesByType[type].push(item);
        });

        // Create a container for all tables
        let securitiesHtml = '<div class="securities-container">';

        // Process each type of securities
        Object.keys(securitiesByType).forEach(type => {
            // Skip empty arrays
            if (securitiesByType[type].length === 0) {
                return;
            }
            
            // Sort securities by allocation amount in descending order
            securitiesByType[type].sort((a, b) => b.amount - a.amount);

            // Capitalize asset class for title
            const formattedType = type.charAt(0).toUpperCase() + type.slice(1);

            // Create a section title for this asset class
            securitiesHtml += `
                <h5 class="asset-class-title mt-4 mb-3">
                    <span class="etf-type-badge badge-${type.toLowerCase()}">${formattedType}</span>
                    ${formattedType} Allocation
                </h5>
            `;

            // Create a table for this type
            securitiesHtml += `
                <table class="etf-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Name</th>
                            <th>Allocation ($)</th>
                            <th>Weight (%)</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            // Calculate total amount for this type to determine weights
            const totalTypeAmount = securitiesByType[type].reduce((sum, item) => sum + (item.amount || 0), 0);

            // Add rows for each security
            securitiesByType[type].forEach(item => {
                const weight = totalTypeAmount > 0 ? ((item.amount || 0) / totalTypeAmount * 100).toFixed(1) : 0;
                securitiesHtml += `
                    <tr>
                        <td class="etf-ticker">${item.ticker || 'N/A'}</td>
                        <td class="etf-name">${item.name || 'Unknown Investment'}</td>
                        <td class="etf-allocation">$${(item.amount || 0).toLocaleString()}</td>
                        <td class="etf-weight">${weight}%</td>
                    </tr>
                `;
            });

            securitiesHtml += '</tbody></table>';
        });

        securitiesHtml += '</div>';
        detailedEtfs.innerHTML = securitiesHtml;
    }

    // Create simple strategy explanation if detailed strategy section exists
    if (detailedStrategy) {
        const riskLevel = getRiskLevel(portfolio.risk_score * 100).toLowerCase();
        const timeHorizon = portfolio.time_horizon || 'medium';
        const investmentGoal = portfolio.investment_goal || 'wealth';
        
        const strategyHtml = `
            <div class="strategy-wrapper">
                <div class="strategy-card">
                    <h5><i class="bi bi-bullseye"></i> Investment Approach</h5>
                    <p>Based on your ${riskLevel} risk profile and investment goals, a ${riskLevel} approach with focus on long-term growth has been recommended.</p>
                    <p>This portfolio is designed with a ${riskLevel} risk profile to align with your ${timeHorizon} time horizon and ${investmentGoal} goals.</p>
                </div>
                
                <div class="strategy-card">
                    <h5><i class="bi bi-graph-up-arrow"></i> Asset Selection Methodology</h5>
                    <p>Securities were selected using a combination of Modern Portfolio Theory optimization and analysis of historical performance.</p>
                    <p>Diversification across asset classes helps reduce volatility while maintaining potential for growth. The specific allocations are calibrated to your risk tolerance and time horizon.</p>
                </div>
            </div>
        `;
        
        detailedStrategy.innerHTML = strategyHtml;
    }

    // Create rebalancing section if it exists
    if (detailedRebalancing) {
        const rebalancingHtml = `
            <div class="rebalancing-wrapper">
                <div class="strategy-card">
                    <h5><i class="bi bi-arrow-repeat"></i> Rebalancing Schedule</h5>
                    <p>To maintain optimal asset allocation and manage risk, a quarterly rebalancing strategy is recommended.</p>
                    <p>Rebalancing should be triggered when any asset class deviates by 5% or more from its target allocation.</p>
                </div>
                
                <div class="strategy-card">
                    <h5><i class="bi bi-diagram-3"></i> Rebalancing Method</h5>
                    <p>This threshold-based approach helps ensure your portfolio stays aligned with your risk profile and investment goals.</p>
                    <p>The rebalancing process involves selling a portion of over-weighted assets and purchasing under-weighted assets to bring the portfolio back to target allocations.</p>
                </div>
            </div>
        `;
        
        detailedRebalancing.innerHTML = rebalancingHtml;
    }

    // Create projections table if the section exists
    if (detailedProjections) {
        const initialAmount = portfolio.investment_amount;
        // Calculate expected return based on allocation
        let expectedReturn = 0;
        
        Object.entries(portfolio.asset_allocation).forEach(([asset, data]) => {
            let assetReturn;
            switch(asset) {
                case 'stocks': assetReturn = 0.09; break;
                case 'bonds': assetReturn = 0.03; break;
                case 'crypto': assetReturn = 0.15; break;
                case 'gold': assetReturn = 0.05; break;
                default: assetReturn = 0.06;
            }
            expectedReturn += assetReturn * data.percentage;
        });
        
        // If no allocation data, use default
        if (expectedReturn === 0) {
            expectedReturn = 0.07;
        }
        
        // Generate projections
        const projectionsHtml = `
            <table class="table table-sm">
                <thead>
                    <tr>
                        <th>Time Period</th>
                        <th>Projected Value</th>
                        <th>Total Return</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1 Year</td>
                        <td>$${Math.round(initialAmount * (1 + expectedReturn)).toLocaleString()}</td>
                        <td>${((expectedReturn) * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                        <td>5 Years</td>
                        <td>$${Math.round(initialAmount * Math.pow(1 + expectedReturn, 5)).toLocaleString()}</td>
                        <td>${((Math.pow(1 + expectedReturn, 5) - 1) * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                        <td>10 Years</td>
                        <td>$${Math.round(initialAmount * Math.pow(1 + expectedReturn, 10)).toLocaleString()}</td>
                        <td>${((Math.pow(1 + expectedReturn, 10) - 1) * 100).toFixed(1)}%</td>
                    </tr>
                </tbody>
            </table>
            <p class="small text-muted mt-3">Projections based on historical data and expected returns. Actual results may vary significantly. Past performance is not indicative of future results.</p>
        `;
        
        detailedProjections.innerHTML = projectionsHtml;
    }

    // Create market outlook section if it exists
    if (detailedOutlook) {
        const outlookHtml = `
            <div class="outlook-container">
                <div class="outlook-item">
                    <div class="outlook-title">
                        <i class="bi bi-graph-up"></i>
                        Inflation
                    </div>
                    <div class="outlook-value outlook-negative">
                        2.8% <span class="outlook-trend">↑</span>
                    </div>
                </div>
                <div class="outlook-item">
                    <div class="outlook-title">
                        <i class="bi bi-graph-up"></i>
                        GDP Growth
                    </div>
                    <div class="outlook-value outlook-positive">
                        3.2% <span class="outlook-trend">↑</span>
                    </div>
                </div>
                <div class="outlook-item">
                    <div class="outlook-title">
                        <i class="bi bi-graph-up"></i>
                        Interest Rates
                    </div>
                    <div class="outlook-value outlook-neutral">
                        5.1% <span class="outlook-trend">→</span>
                    </div>
                </div>
                <div class="outlook-item">
                    <div class="outlook-title">
                        <i class="bi bi-graph-up"></i>
                        Unemployment
                    </div>
                    <div class="outlook-value outlook-positive">
                        3.9% <span class="outlook-trend">↓</span>
                    </div>
                </div>
            </div>
        `;
        
        detailedOutlook.innerHTML = outlookHtml;
    }

    // Show the modal
    try {
        console.log("Attempting to show recommendations modal");
        if (recommendationsModal && typeof recommendationsModal.show === 'function') {
            recommendationsModal.show();
        } else if (modalElement) {
            // Fallback if the bootstrap modal object isn't working
            modalElement.style.display = 'block';
            modalElement.classList.add('show');
            document.body.classList.add('modal-open');
            
            let backdrop = document.querySelector('.modal-backdrop');
            if (!backdrop) {
                backdrop = document.createElement('div');
                backdrop.className = 'modal-backdrop fade show';
                document.body.appendChild(backdrop);
            }
        } else {
            throw new Error("No valid modal display method available");
        }
    } catch (error) {
        console.error("Error showing modal:", error);
        // Fallback - try with jQuery if available
        try {
            if (typeof $ !== 'undefined' && typeof $.fn.modal === 'function') {
                $('#recommendationsModal').modal('show');
                console.log("Modal displayed with jQuery");
            } else {
                throw new Error("jQuery not available");
            }
        } catch (e) {
            console.error("Could not show modal with any method:", e);
            alert("Unable to display detailed recommendations. Please try refreshing the page.");
        }
    }
}

// Helper functions for formatting
function formatInvestmentGoal(goal) {
    const goals = {
        'retirement': 'Retirement Planning',
        'education': 'Education Funding',
        'house': 'Home Purchase',
        'wealth': 'Wealth Building'
    };
    return goals[goal] || goal;
}

function formatRiskTolerance(risk) {
    const risks = {
        'low': 'Conservative',
        'medium': 'Moderate',
        'high': 'Aggressive'
    };
    return risks[risk] || risk;
}

function formatTimeHorizon(horizon) {
    const horizons = {
        'short': '< 5 Years',
        'medium': '5-10 Years',
        'long': '> 10 Years'
    };
    return horizons[horizon] || horizon;
}

/**
 * Create the allocation chart
 * @param {Array} allocation - The asset allocation
 */
function createAllocationChart(allocation) {
    console.log("Creating allocation chart with data:", allocation);
    const chartContainer = document.querySelector('.chart-container');

    if (!chartContainer) {
        console.error("Chart container not found");
        return;
    }

    // Clear previous chart
    chartContainer.innerHTML = '';

    // Check if we have allocation data
    if (!allocation || allocation.length === 0) {
        console.error("No allocation data available");
        const errorMessage = document.createElement('div');
        errorMessage.className = 'text-center text-muted py-5';
        errorMessage.textContent = 'No allocation data available';
        chartContainer.appendChild(errorMessage);
        return;
    }

    // Create canvas for the chart
    const canvas = document.createElement('canvas');
    canvas.id = 'allocationChart';
    chartContainer.appendChild(canvas);

    // Prepare data for Chart.js
    const ctx = canvas.getContext('2d');
    const labels = allocation.map(item => item.name);
    const data = allocation.map(item => item.percentage);
    const colors = allocation.map(item => item.color);

    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error("Chart.js not loaded, falling back to simple chart");
        createSimpleChart(ctx, labels, data, colors);
        return;
    }

    // Create chart with Chart.js
    try {
        // Destroy existing chart if it exists
        if (window.allocationPieChart) {
            window.allocationPieChart.destroy();
        }

        window.allocationPieChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const item = allocation[context.dataIndex];
                                return `${item.name}: ${item.percentage}% ($${item.amount.toLocaleString()})`;
                            }
                        }
                    }
                },
                cutout: '60%'
            }
        });

        // Debug info
        console.log("Allocation pie chart created successfully");
    } catch (error) {
        console.error("Error creating chart:", error);
        // Add visible error message if chart creation fails
        const errorMsg = document.createElement('div');
        errorMsg.className = 'chart-error-message';
        errorMsg.innerHTML = `<p>Chart rendering issue. Please try refreshing.</p><p>Error: ${error.message}</p>`;
        errorMsg.style.color = '#ff5a5f';
        errorMsg.style.textAlign = 'center';
        errorMsg.style.padding = '20px';
        chartContainer.innerHTML = '';
        chartContainer.appendChild(errorMsg);
        
        // Fallback to simple chart
        createSimpleChart(ctx, labels, data, colors);
    }
}

/**
 * Create a simple chart when Chart.js is not available
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array} labels - Chart labels
 * @param {Array} data - Chart data
 * @param {Array} colors - Chart colors
 */
function createSimpleChart(ctx, labels, data, colors) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) * 0.8;
    const innerRadius = radius * 0.6; // For doughnut effect

    // Calculate total for percentages
    const total = data.reduce((sum, value) => sum + value, 0);

    // Draw the chart
    let startAngle = -Math.PI / 2; // Start at top

    // First draw outer circle
    for (let i = 0; i < data.length; i++) {
        const sliceAngle = (data[i] / total) * 2 * Math.PI;
        const endAngle = startAngle + sliceAngle;

        // Draw slice
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.closePath();
        ctx.fillStyle = colors[i];
        ctx.fill();

        startAngle = endAngle;
    }

    // Draw inner circle for doughnut effect
    ctx.beginPath();
    ctx.arc(centerX, centerY, innerRadius, 0, 2 * Math.PI);
    ctx.fillStyle = '#1e1e1e'; // Match background color
    ctx.fill();
}

/**
 * Get risk level text based on risk score
 * @param {number} riskScore - The risk score (0-100)
 * @returns {string} Risk level text
 */
function getRiskLevel(riskScore) {
    if (riskScore < 30) {
        return 'Conservative';
    } else if (riskScore < 70) {
        return 'Moderate';
    } else {
        return 'Aggressive';
    }
}

/**
 * Get CSS class for risk level
 * @param {string} riskLevel - The risk level text
 * @returns {string} CSS class
 */
function getRiskClass(riskLevel) {
    const classMap = {
        'Conservative': 'risk-conservative',
        'Moderate': 'risk-moderate',
        'Aggressive': 'risk-aggressive'
    };
    return classMap[riskLevel] || 'risk-moderate';
}

/**
 * Create a chart showing historical performance of each asset class
 * @param {Object} portfolio - The portfolio data
 */
function createHistoricalPerformanceChart(portfolio) {
    console.log("Creating historical performance chart");
    try {
        // Clear previous chart by clearing the container
        const chartContainer = document.querySelector('#performance-chart .chart-container-lg');
        if (!chartContainer) {
            console.error("Chart container for historical performance not found");
            return;
        }
        
        // Clear container and create new canvas
        chartContainer.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.id = 'historicalPerformanceChart';
        chartContainer.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Generate mock historical data based on asset classes
        const labels = ['1 Year', '3 Years', '5 Years', '10 Years'];
        const datasets = [];
        
        // Colors for asset classes
        const assetClasses = Object.keys(portfolio.asset_allocation || {});
        console.log("Asset classes for historical chart:", assetClasses);
        
        // If no asset classes found, use dummy data
        if (assetClasses.length === 0) {
            assetClasses.push('stocks', 'bonds', 'crypto', 'gold');
        }
        
        assetClasses.forEach(asset => {
            const color = getAssetColor(asset);
            
            // Generate realistic historical returns based on asset class
            let returns;
            switch(asset) {
                case 'stocks':
                    returns = [9.5, 12.3, 10.8, 11.2];
                    break;
                case 'bonds':
                    returns = [2.1, 3.3, 2.8, 3.5];
                    break;
                case 'crypto':
                    returns = [22.5, 42.3, 35.8, null]; // No 10-year data for crypto
                    break;
                case 'gold':
                    returns = [5.2, 7.1, 6.5, 4.8];
                    break;
                default:
                    returns = [6.5, 8.2, 7.5, 7.2];
            }
            
            datasets.push({
                label: asset.charAt(0).toUpperCase() + asset.slice(1),
                data: returns,
                backgroundColor: color,
                borderColor: color,
                borderWidth: 1
            });
        });
        
        // Add portfolio overall performance (weighted average)
        const portfolioReturns = [0, 0, 0, 0];
        assetClasses.forEach((asset, index) => {
            const weight = portfolio.asset_allocation && portfolio.asset_allocation[asset] ? 
                portfolio.asset_allocation[asset].percentage : 0.25;
            const assetData = datasets[index].data;
            
            assetData.forEach((value, i) => {
                if (value !== null) {
                    portfolioReturns[i] += value * weight;
                }
            });
        });
        
        datasets.push({
            label: 'Portfolio Overall',
            data: portfolioReturns,
            backgroundColor: '#3a9ffb',
            borderColor: '#3a9ffb',
            borderWidth: 2,
            type: 'line',
            fill: false,
            tension: 0.1,
            pointBackgroundColor: '#ffffff',
            pointRadius: 5,
            pointHoverRadius: 7
        });
        
        // Create a new chart instance without trying to destroy previous one
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(1) + '%';
                                }
                                return label;
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Historical Annualized Returns',
                        color: '#e0e0e0',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#c0c0c0',
                            boxWidth: 12,
                            padding: 15
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Annualized Return %',
                            color: '#c0c0c0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#c0c0c0',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#c0c0c0'
                        }
                    }
                }
            }
        });
        
        console.log("Historical performance chart created successfully");
    } catch (error) {
        console.error("Error creating historical performance chart:", error);
        
        // Add visible error message
        const container = document.querySelector('#performance-chart .chart-container-lg');
        if (container) {
            const errorMsg = document.createElement('div');
            errorMsg.className = 'chart-error-message';
            errorMsg.innerHTML = `<p>Unable to render historical performance chart.</p><p>Error: ${error.message}</p>`;
            errorMsg.style.color = '#ff5a5f';
            errorMsg.style.textAlign = 'center';
            errorMsg.style.padding = '20px';
            container.innerHTML = '';
            container.appendChild(errorMsg);
        }
    }
}

/**
 * Create a risk-return scatter plot
 * @param {Object} portfolio - The portfolio data
 */
function createRiskReturnChart(portfolio) {
    console.log("Creating risk-return chart");
    try {
        // Clear previous chart by clearing the container
        const chartContainer = document.querySelector('#risk-return-chart .chart-container-lg');
        if (!chartContainer) {
            console.error("Chart container for risk-return not found");
            return;
        }
        
        // Clear container and create new canvas
        chartContainer.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.id = 'riskReturnChart';
        chartContainer.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Generate risk-return data points for each asset class
        const assetClasses = Object.keys(portfolio.asset_allocation || {});
        console.log("Asset classes for risk-return chart:", assetClasses);
        
        // If no asset classes found, use dummy data
        if (assetClasses.length === 0) {
            assetClasses.push('stocks', 'bonds', 'crypto', 'gold');
        }
        
        const dataPoints = [];
        
        // Define realistic risk-return characteristics
        const riskReturnMap = {
            'stocks': { return: 9.8, risk: 15.5 },
            'bonds': { return: 3.2, risk: 4.5 },
            'crypto': { return: 25.0, risk: 45.0 },
            'gold': { return: 5.5, risk: 12.0 }
        };
        
        // Calculate portfolio's risk and return (weighted average)
        let portfolioReturn = 0;
        let portfolioRisk = 0;
        
        assetClasses.forEach(asset => {
            const weight = portfolio.asset_allocation && portfolio.asset_allocation[asset] ? 
                portfolio.asset_allocation[asset].percentage : 0.25;
            const assetData = riskReturnMap[asset] || { return: 6.0, risk: 10.0 };
            
            portfolioReturn += assetData.return * weight;
            portfolioRisk += assetData.risk * weight; // This is a simplification
            
            dataPoints.push({
                label: asset.charAt(0).toUpperCase() + asset.slice(1),
                data: [{
                    x: assetData.risk,
                    y: assetData.return
                }],
                backgroundColor: getAssetColor(asset),
                borderColor: getAssetColor(asset),
                borderWidth: 1,
                pointRadius: 8,
                pointHoverRadius: 10
            });
        });
        
        // Add portfolio point
        dataPoints.push({
            label: 'Your Portfolio',
            data: [{
                x: portfolioRisk,
                y: portfolioReturn
            }],
            backgroundColor: '#3a9ffb',
            borderColor: '#ffffff',
            borderWidth: 2,
            pointRadius: 10,
            pointHoverRadius: 12,
            pointStyle: 'star'
        });
        
        // Create the risk-return benchmark line (Capital Market Line)
        const riskPoints = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
        const cmlPoints = riskPoints.map(risk => {
            // Simplified CML: rf + risk premium * risk
            return 2.0 + 0.22 * risk;
        });
        
        dataPoints.push({
            label: 'Efficient Frontier',
            data: riskPoints.map((risk, i) => ({
                x: risk,
                y: cmlPoints[i]
            })),
            backgroundColor: 'rgba(0, 0, 0, 0)',
            borderColor: 'rgba(255, 255, 255, 0.3)',
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
            pointHoverRadius: 0,
            type: 'line'
        });
        
        // Create a new chart instance without trying to destroy previous one
        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: dataPoints
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const risk = context.parsed.x.toFixed(1) + '%';
                                const ret = context.parsed.y.toFixed(1) + '%';
                                return `${label}: Return ${ret}, Risk ${risk}`;
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Risk vs. Return Analysis',
                        color: '#e0e0e0',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#c0c0c0',
                            boxWidth: 12,
                            padding: 15
                        }
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Expected Return %',
                            color: '#c0c0c0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#c0c0c0',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Risk (Volatility) %',
                            color: '#c0c0c0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#c0c0c0',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
        
        console.log("Risk-return chart created successfully");
    } catch (error) {
        console.error("Error creating risk-return chart:", error);
        
        // Add visible error message
        const container = document.querySelector('#risk-return-chart .chart-container-lg');
        if (container) {
            const errorMsg = document.createElement('div');
            errorMsg.className = 'chart-error-message';
            errorMsg.innerHTML = `<p>Unable to render risk-return chart.</p><p>Error: ${error.message}</p>`;
            errorMsg.style.color = '#ff5a5f';
            errorMsg.style.textAlign = 'center';
            errorMsg.style.padding = '20px';
            container.innerHTML = '';
            container.appendChild(errorMsg);
        }
    }
}

/**
 * Create a projection chart showing potential growth over time
 * @param {Object} portfolio - The portfolio data
 */
function createProjectionChart(portfolio) {
    console.log("Creating projection chart");
    try {
        // Clear previous chart by clearing the container
        const chartContainer = document.querySelector('#projection-chart .chart-container-lg');
        if (!chartContainer) {
            console.error("Chart container for projection not found");
            return;
        }
        
        // Clear container and create new canvas
        chartContainer.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.id = 'projectionChart';
        chartContainer.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Set up time horizons based on user's input
        const years = [];
        const timeHorizon = portfolio.time_horizon || 'medium';
        let maxYears = 10;
        
        switch(timeHorizon) {
            case 'short':
                maxYears = 5;
                break;
            case 'medium':
                maxYears = 10;
                break;
            case 'long':
                maxYears = 20;
                break;
        }
        
        // Generate year labels
        for (let i = 0; i <= maxYears; i++) {
            years.push(`Year ${i}`);
        }
        
        // Calculate expected portfolio return rate
        let expectedReturn = 0;
        const assetClasses = Object.keys(portfolio.asset_allocation || {});
        
        // If no asset classes found, use dummy data
        if (assetClasses.length === 0) {
            expectedReturn = 0.08; // Default 8% return
        } else {
            assetClasses.forEach(asset => {
                const weight = portfolio.asset_allocation && portfolio.asset_allocation[asset] ? 
                    portfolio.asset_allocation[asset].percentage : 0.25;
                let assetReturn;
                
                switch(asset) {
                    case 'stocks':
                        assetReturn = 0.09;
                        break;
                    case 'bonds':
                        assetReturn = 0.03;
                        break;
                    case 'crypto':
                        assetReturn = 0.15;
                        break;
                    case 'gold':
                        assetReturn = 0.05;
                        break;
                    default:
                        assetReturn = 0.06;
                }
                
                expectedReturn += assetReturn * weight;
            });
        }
        
        // Generate three growth scenarios
        const initialAmount = portfolio.investment_amount || 10000;
        const pessimistic = [initialAmount];
        const expected = [initialAmount];
        const optimistic = [initialAmount];
        
        // Risk-based variability
        const riskFactor = 0.4 + ((portfolio.risk_score || 0.5) * 0.6); // Higher risk = higher variability
        
        // Calculate compound growth
        for (let i = 1; i <= maxYears; i++) {
            pessimistic.push(pessimistic[i-1] * (1 + (expectedReturn * 0.5)));
            expected.push(expected[i-1] * (1 + expectedReturn));
            optimistic.push(optimistic[i-1] * (1 + (expectedReturn * 1.5)));
        }
        
        // Create a new chart instance without trying to destroy previous one
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: years,
                datasets: [
                    {
                        label: 'Conservative Estimate',
                        data: pessimistic,
                        borderColor: 'rgba(255, 165, 0, 0.8)',
                        backgroundColor: 'rgba(255, 165, 0, 0.1)',
                        fill: '+1',
                        tension: 0.1,
                        borderWidth: 2,
                        pointRadius: 3
                    },
                    {
                        label: 'Expected Growth',
                        data: expected,
                        borderColor: '#3a9ffb',
                        backgroundColor: 'rgba(58, 159, 251, 0.1)',
                        fill: '+1',
                        tension: 0.1,
                        borderWidth: 3,
                        pointRadius: 4
                    },
                    {
                        label: 'Optimistic Estimate',
                        data: optimistic,
                        borderColor: 'rgba(75, 192, 192, 0.8)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        fill: true,
                        tension: 0.1,
                        borderWidth: 2,
                        pointRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += '$' + context.parsed.y.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
                                }
                                return label;
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Portfolio Growth Projection',
                        color: '#e0e0e0',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#c0c0c0',
                            boxWidth: 12,
                            padding: 15
                        }
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Portfolio Value ($)',
                            color: '#c0c0c0'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#c0c0c0',
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#c0c0c0'
                        }
                    }
                }
            }
        });
        
        console.log("Projection chart created successfully");
    } catch (error) {
        console.error("Error creating projection chart:", error);
        
        // Add visible error message
        const container = document.querySelector('#projection-chart .chart-container-lg');
        if (container) {
            const errorMsg = document.createElement('div');
            errorMsg.className = 'chart-error-message';
            errorMsg.innerHTML = `<p>Unable to render projection chart.</p><p>Error: ${error.message}</p>`;
            errorMsg.style.color = '#ff5a5f';
            errorMsg.style.textAlign = 'center';
            errorMsg.style.padding = '20px';
            container.innerHTML = '';
            container.appendChild(errorMsg);
        }
    }
}