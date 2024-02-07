import { useState } from "react";
import { useAuth } from "../hooks/AuthProvider";

const Login = () => {
    const [isLoading, setIsLoading] = useState(false);

    const [input, setInput] = useState({
        username: "",
        password: "",
    });

    const auth = useAuth();

    const handleSubmitEvent = (e) => {
        
        e.preventDefault();

        // Improved validation logic
        if (input.username.trim() === "" || input.password.trim() === "") {
            alert("Please provide valid input for both username and password.");
            setIsLoading(false);
            return;
        }

        setIsLoading(true);
        auth.loginAction(input);
        
        setIsLoading(false);
    };

    const handleInput = (e) => {
        const { name, value } = e.target;

        // Simplified state update using the functional form
        setInput((prev) => ({
            ...prev,
            [name]: value,
        }));
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-200 to-purple-200">
            <form onSubmit={handleSubmitEvent} className="bg-white shadow-md rounded-lg px-12 pt-6 pb-12 mb-4 w-4/12">
                <h1 className="text-2xl text-center font-bold pt-3 pb-6 py-6">Sign in to your account</h1>
                <div className="mb-4">
                    <label htmlFor="username" className="block text-gray-700 text-sm font-bold mb-2">
                        Movielens User ID (1-943)
                    </label>
                    <input
                        type="text"
                        id="username"
                        name="username"
                        placeholder="username"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={input.username}
                        onChange={handleInput}
                    />
                </div>
                <div className="mb-6">
                    <label htmlFor="password" className="block text-gray-700 text-sm font-bold mb-2">
                        Password
                    </label>
                    <input
                        type="password"
                        id="password"
                        name="password"
                        placeholder="Password"
                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        value={input.password}
                        onChange={handleInput}
                    />
                </div>
                <div className="flex items-center justify-between">
                    {isLoading ? (<svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                    >
                        <path
                            fill="currentColor"
                            d="M12,4a8,8,0,0,1,7.89,6.7A1.53,1.53,0,0,0,21.38,12h0a1.5,1.5,0,0,0,1.48-1.75,11,11,0,0,0-21.72,0A1.5,1.5,0,0,0,2.62,12h0a1.53,1.53,0,0,0,1.49-1.3A8,8,0,0,1,12,4Z"
                        >
                            <animateTransform
                                attributeName="transform"
                                dur="0.75s"
                                repeatCount="indefinite"
                                type="rotate"
                                values="0 12 12;360 12 12"
                            />
                        </path>
                    </svg>
                    ) :
                        <button type="submit" className="bg-blue-500 hover:bg-blue-700 text-white w-full font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline">
                            Login
                        </button>}
                </div>
            </form>
        </div>
    );
};

export default Login;
