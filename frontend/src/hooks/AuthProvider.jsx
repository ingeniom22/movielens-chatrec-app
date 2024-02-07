import { useContext, createContext, useState } from "react";
import { useNavigate } from "react-router-dom";

const AuthContext = createContext();

const AuthProvider = ({ children }) => {
    const apiUrl = import.meta.env.VITE_API_URL;
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem("site") || "");
    const navigate = useNavigate();

    const loginAction = async (formData) => {
        try {
            const formDataEncoded = Object.keys(formData)
                .map(key => encodeURIComponent(key) + '=' + encodeURIComponent(formData[key]))
                .join('&');

            console.log(formDataEncoded)

            const response = await fetch(apiUrl+"/token", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: formDataEncoded, 
            });

            const res = await response.json();
            if (res) {
                localStorage.setItem("site", res.access_token);
                setToken(res.access_token);

                const userResponse = await fetch(apiUrl+"/users/me/", {
                    method: "GET",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + res.access_token
                    }
                });

                if (userResponse.ok) {
                    const userData = await userResponse.json();
                    setUser(userData);

                    navigate("/");
                    return;
                } else {
                    alert("Wrong username or password")
                    console.error("User request failed:", userResponse.statusText);
                }
            }
            throw new Error(res.message);
        } catch (err) {
            console.error(err);
        }
    };


    const logOut = () => {
        setUser(null);
        setToken("");
        localStorage.removeItem("site");
        navigate("/login");
    };

    return (
        <AuthContext.Provider value={{ user, token, loginAction, logOut }}>
            {children}
        </AuthContext.Provider>
    );

};

export default AuthProvider;

export const useAuth = () => {
    return useContext(AuthContext);
};