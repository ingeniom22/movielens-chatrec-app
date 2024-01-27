import { BrowserRouter as Router, Route, Routes, useNavigate } from "react-router-dom";
import Login from "./components/Login";
import Chat from "./components/Chat";
import AuthProvider, { useAuth } from "./hooks/AuthProvider";
import PrivateRoute from "./router/Route";



function App() {
  return (
    <div className="App">
      <Router>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
            
            <Route element={<PrivateRoute />}>
              <Route path="/" element={<Chat />} />
            </Route>
            
          </Routes>
        </AuthProvider>
      </Router>
    </div>
  );
}

export default App;