// Authentication and User Management
// Using localStorage for client-side demo purposes

// Register a new user
function registerUser(name, email, password) {
    // Get existing users from localStorage
    let users = JSON.parse(localStorage.getItem('outfAIt_users')) || [];

    // Check if email already exists
    const existingUser = users.find(user => user.email === email);
    if (existingUser) {
        return {
            success: false,
            message: 'Email already registered'
        };
    }

    // Create new user object
    const newUser = {
        id: Date.now().toString(),
        name: name,
        email: email,
        password: password, // In production, this should be hashed
        createdAt: new Date().toISOString()
    };

    // Add to users array
    users.push(newUser);

    // Save to localStorage
    localStorage.setItem('outfAIt_users', JSON.stringify(users));

    // Set current user session
    const userSession = {
        id: newUser.id,
        name: newUser.name,
        email: newUser.email
    };
    localStorage.setItem('outfAIt_currentUser', JSON.stringify(userSession));

    return {
        success: true,
        user: userSession
    };
}

// Login user
function loginUser(email, password) {
    // Get existing users from localStorage
    let users = JSON.parse(localStorage.getItem('outfAIt_users')) || [];

    // Find user with matching email and password
    const user = users.find(u => u.email === email && u.password === password);

    if (!user) {
        return {
            success: false,
            message: 'Invalid email or password'
        };
    }

    // Set current user session
    const userSession = {
        id: user.id,
        name: user.name,
        email: user.email
    };
    localStorage.setItem('outfAIt_currentUser', JSON.stringify(userSession));

    return {
        success: true,
        user: userSession
    };
}

// Get current logged-in user
function getCurrentUser() {
    const userSession = localStorage.getItem('outfAIt_currentUser');
    return userSession ? JSON.parse(userSession) : null;
}

// Logout user
function logout() {
    localStorage.removeItem('outfAIt_currentUser');
}

// Check if user is logged in
function isLoggedIn() {
    return getCurrentUser() !== null;
}
