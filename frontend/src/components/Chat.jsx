import React, { useEffect, useState } from "react";
import { useAuth } from "../hooks/AuthProvider";
import { useNavigate } from "react-router-dom";

const Chat = () => {
    const apiUrl = import.meta.env.VITE_API_URL;

    const auth = useAuth();
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(false);

    const [chatHistory, setChatHistory] = useState([]);
    const [inputText, setInputText] = useState("");
    const [role, setRole] = useState("");


    useEffect(() => {
        const fetchUser = async () => {
            try {
                const userResponse = await fetch(apiUrl + "/users/me/", {
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





    const handleSubmitMessage = async () => {
        try {
            setLoading(true);

            const newHumanMessage = {
                "content": inputText,
                "additional_kwargs": {},
                "type": "human",
                "example": false
            };

            setChatHistory(prevChatHistory => [...prevChatHistory, newHumanMessage]);

            const response = await fetch(apiUrl + "/invoke", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${auth.token}`
                },
                body: JSON.stringify({
                    input: {
                        input: inputText,
                        role: role,
                        chat_history: chatHistory.slice(0, -1)
                    }
                })
            });

            const result = await response.json();


            const newAIMessage = {
                "content": result.output.output,
                "additional_kwargs": {},
                "type": "ai",
                "example": false
            };

            setChatHistory(prevChatHistory => [...prevChatHistory, newAIMessage]);

            setInputText("");
            setLoading(false);

        } catch (error) {
            console.error("Error sending message:", error);
        }
    };


    const handleInputChange = (e) => {
        setInputText(e.target.value);
    };

    const handleRoleChange = (e) => {
        setRole(e.target.value);
    };

    return (
        <div className="container mx-auto p-8 flex flex-col justify-between h-screen">
            <div className="flex justify-between items-center bg-teal-500 p-4 rounded-xl">
                <h1 className="text-2xl font-bold text-white">Welcome, {user?.full_name}!</h1>
                <button onClick={() => auth.logOut()} className="p-2 bg-teal-700 rounded-lg text-white">
                    Logout
                </button>
            </div>

            <div className="mt-4 mb-4">
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

            {chatHistory.length > 0 ? (
                <div className="bg-white rounded p-8 mb-8">
                    <div className="mt-4">
                        {chatHistory.map((message, index) => (
                            <div
                                key={index}
                                className={`flex mb-2 ${message.type === 'ai' ? 'justify-start' : 'justify-end'}`}
                            >
                                <div className={'bg-gray-200 p-2 rounded max-w-lg'}>
                                    <p className="text-gray-800 font-bold">
                                        {message.type === 'ai' ? 'ChatRec' : 'Human'}
                                    </p>
                                    <p>{message.content}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="flex justify-center items-center h-full align-middle">
                    <h1 className="text-4xl text-gray-800">Ask ChatRec</h1>
                </div>
            )}

            <form className="mt-8 mb-8 fixed inset-x-0 bottom-0 left-0 w-">
                <label htmlFor="chat" className="sr-only">
                    Your message
                </label>
                <div className="flex items-center px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-700">
                    <textarea
                        id="chat"
                        value={loading ? "" : inputText}
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
    );
};
export default Chat;
