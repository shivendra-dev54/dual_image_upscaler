import { Link } from "react-router";

// interface NavbarProps {
//   isLoggedIn: boolean;
//   setIsLoggedIn: (loggedIn: boolean) => void;
// }

const Navbar = () => {
  // const navigate = useNavigate();

  // const handleLogout = () => {
  //   setIsLoggedIn(false);
  //   localStorage.clear();
  //   navigate("/");
  // };

  return (
    <nav className="h-16 bg-gradient-to-b from-black via-[#1e293b] to-[#031634] text-white flex items-center justify-between px-4 shadow-md border-b border-slate-700">
      <div className="text-xl font-bold tracking-wide">
        <Link to="/" className="hover:text-emerald-400 transition duration-300">
          Heart
        </Link>
      </div>

      {/* <div className="flex space-x-4 text-sm sm:text-base">
        {isLoggedIn ? (
          <>
            <Link
              to="/main_page"
              className="hover:text-blue-400 transition duration-300"
            >
              Home
            </Link>
            <Link
              to="/profile"
              className="hover:text-blue-400 transition duration-300"
            >
              Profile
            </Link>
            <button
              onClick={handleLogout}
              className="hover:text-red-400 transition duration-300"
            >
              Logout
            </button>
          </>
        ) : (
          <>
            <Link
              to="/sign_in"
              className="hover:text-emerald-400 transition duration-300"
            >
              Sign In
            </Link>
            <Link
              to="/sign_up"
              className="hover:text-emerald-400 transition duration-300"
            >
              Sign Up
            </Link>
          </>
        )}
      </div> */}
    </nav>
  );
};

export default Navbar;