import React, { useEffect, useState } from "react";
import { useAuth } from "../hooks/AuthProvider";
import { useNavigate } from "react-router-dom";

const Chat = () => {
    const apiUrl = import.meta.env.VITE_API_URL;
    
    const auth = useAuth();
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(false);


    useEffect(() => {
        const fetchUser = async () => {
            try {
                const userResponse = await fetch(apiUrl+"/users/me/", {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + auth.token
                    }
                });

                if (userResponse.ok) {
                    const userData = await userResponse.json();
                    setUser(userData);
                } else {
                    navigate("/login");
                    console.error("User request failed:", userResponse.statusText);
                }
            } catch (error) {
                console.error("Error fetching user:", error);
            }
        };

        fetchUser();
    }, [auth.token, navigate]);



    const [chatHistory, setChatHistory] = useState([]);
    const [inputText, setInputText] = useState("");
    const [role, setRole] = useState("");

    const handleSubmitMessage = async () => {
        try {
            setLoading(true);
            const response = await fetch("http://localhost:8000/invoke", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${auth.token}`
                },
                body: JSON.stringify({
                    input: {
                        input: inputText,
                        role: role,
                        chat_history: chatHistory
                    }
                })
            });

            const result = await response.json();

            const newHumanMessage = {
                "content": inputText,
                "additional_kwargs": {},
                "type": "human",
                "example": false
            };

            const newAIMessage = {
                "content": result.output.output,
                "additional_kwargs": {},
                "type": "ai",
                "example": false
            };

            setChatHistory([...chatHistory, newHumanMessage, newAIMessage]);

            // Clear the input field
            setInputText("");
            setLoading(false);

        } catch (error) {
            console.error("Error sending message:", error);
            // Handle errors as needed
        }
    };


    const handleInputChange = (e) => {
        setInputText(e.target.value);
    };

    const handleRoleChange = (e) => {
        setRole(e.target.value);
    };

    return (
        <div className="container mx-auto p-8">
            <div className="bg-white rounded shadow p-8">
                <div className="flex justify-between items-center mb-4">
                    <h1 className="text-2xl font-bold">Welcome, {user?.full_name}!</h1>
                    <button onClick={() => auth.logOut()} className="p-2 btn-submit bg-blue-500 rounded-lg text-white">
                        Logout
                    </button>
                </div>

                <div className="mt-4">
                    <label htmlFor="role" className="block text-sm font-medium text-gray-700">
                        Masukkan Role ChatRec
                    </label>
                    <input
                        type="text"
                        id="role"
                        placeholder="Role"
                        value={role}
                        onChange={handleRoleChange}
                        className="mt-1 p-2 input-field border border-gray-500 rounded-xl"
                    />
                </div>

                <div className="mt-4">
                    {chatHistory.map((message, index) => (
                        <div
                            key={index}
                            className={`flex mb-2 ${message.type === 'ai' ? 'justify-start' : 'justify-end'}`}
                        >
                            <div className={'bg-gray-200 p-2 rounded'}>
                                <p className="text-gray-800 font-bold">
                                    {message.type === 'ai' ? 'ChatRec' : 'Human'}
                                </p>
                                <p>{message.content}</p>
                            </div>
                        </div>
                    ))}
                </div>


                <form>
                    <label htmlFor="chat" className="sr-only">
                        Your message
                    </label>
                    <div className="flex items-center px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-700">
                        <textarea
                            id="chat"
                            value={inputText}
                            onChange={handleInputChange}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                    e.preventDefault();
                                    handleSubmitMessage();
                                }
                            }}
                            rows="1"
                            className="block mx-4 p-2.5 w-full text-sm text-gray-900 bg-white rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                            placeholder="Your message..."
                        ></textarea>

                        {loading ? (
                            <div className="inline-flex justify-center p-2 text-blue-600 rounded-full cursor-pointer hover:bg-blue-100 dark:text-blue-500 dark:hover:bg-gray-600">
                                {/* Spinner SVG */}
                                <svg
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
                                <span className="sr-only">Sending message...</span>
                            </div>
                        ) : (
                            <button
                                onClick={handleSubmitMessage}
                                type="button"
                                className="inline-flex justify-center p-2 text-blue-600 rounded-full cursor-pointer hover:bg-blue-100 dark:text-blue-500 dark:hover:bg-gray-600"
                            >
                                {/* Your original button content */}
                                <svg
                                    className="w-5 h-5 rotate-90 rtl:-rotate-90"
                                    aria-hidden="true"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="currentColor"
                                    viewBox="0 0 18 20"
                                >
                                    <path d="m17.914 18.594-8-18a1 1 0 0 0-1.828 0l-8 18a1 1 0 0 0 1.157 1.376L8 18.281V9a1 1 0 0 1 2 0v9.281l6.758 1.689a1 1 0 0 0 1.156-1.376Z" />
                                </svg>
                                <span className="sr-only">Send message</span>
                            </button>
                        )}
                    </div>
                </form>



            </div>
        </div>
    );
};
export default Chat;
