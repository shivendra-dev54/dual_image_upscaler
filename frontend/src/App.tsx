// import { useState } from "react"
import { Routes, Route } from "react-router"
import Navbar from "./Components/Navbar"
import HomePage from "./pages/HomePage"
import MainPage from "./pages/MainPage"


function App() {

  // const [isLoggedIn, setIsLoggedIn] = useState(false)

  return (
    <div className='h-screen w-screen text-blue-100 bg-black flex flex-col overflow-x-hidden select-none'>
      <Navbar />

      <div className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/main_page" element={<MainPage />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
