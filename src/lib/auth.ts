
export interface User {
  fullName: string;
  email: string;
}

interface UserWithPassword extends User {
    password: string;
}

const USERS_KEY = "eduperform_users";
const CURRENT_USER_KEY = "eduperform_current_user";

// Helper to work with localStorage safely on the client side
const getLocalStorage = (key: string) => {
    if (typeof window !== 'undefined') {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    }
    return null;
}

const setLocalStorage = (key: string, value: any) => {
    if (typeof window !== 'undefined') {
        localStorage.setItem(key, JSON.stringify(value));
        // Dispatch a storage event to notify other tabs
        window.dispatchEvent(new Event('storage'));
    }
}


export const signup = ({ fullName, email, password }: UserWithPassword) => {
  const users: UserWithPassword[] = getLocalStorage(USERS_KEY) || [];

  if (users.find(user => user.email === email)) {
    throw new Error("User with this email already exists.");
  }

  users.push({ fullName, email, password });
  setLocalStorage(USERS_KEY, users);
};

export const login = (email: string, password: string): User => {
  const users: UserWithPassword[] = getLocalStorage(USERS_KEY) || [];
  const user = users.find(u => u.email === email && u.password === password);

  if (!user) {
    throw new Error("Invalid email or password");
  }
  
  const { password: _, ...userWithoutPassword } = user;
  setLocalStorage(CURRENT_USER_KEY, userWithoutPassword);
  return userWithoutPassword;
};

export const logout = () => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(CURRENT_USER_KEY);
    // Dispatch a storage event to notify other tabs
    window.dispatchEvent(new Event('storage'));
  }
};

export const getCurrentUser = (): User | null => {
  return getLocalStorage(CURRENT_USER_KEY);
};
