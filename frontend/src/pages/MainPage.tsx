import { useState } from "react";

const MainPage = () => {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!image1 || !image2) return;

    const formData = new FormData();
    formData.append("file1", image1);
    formData.append("file2", image2);

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/upscale/", {
        method: "POST",
        body: formData,
      });

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-black via-gray-900 to-gray-800 px-4 py-10">
      <div className="bg-gray-900 bg-opacity-80 rounded-xl shadow-2xl p-8 w-full max-w-md flex flex-col items-center">
        <h1 className="text-3xl font-extrabold mb-6 text-white tracking-wide drop-shadow-lg">
          Dual-Image <span className="text-blue-400">Super-Resolution</span>
        </h1>

        <label className="w-full mb-3">
          <span className="block text-gray-300 mb-1 font-medium">Image 1</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImage1(e.target.files?.[0] || null)}
            className="block w-full text-gray-200 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
          />
        </label>
        <label className="w-full mb-4">
          <span className="block text-gray-300 mb-1 font-medium">Image 2</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImage2(e.target.files?.[0] || null)}
            className="block w-full text-gray-200 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
          />
        </label>

        <button
          onClick={handleSubmit}
          className="w-full mt-2 px-6 py-2 bg-blue-600 rounded-lg font-semibold text-white shadow hover:bg-blue-700 disabled:bg-gray-700 transition"
          disabled={!image1 || !image2 || loading}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-2 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
              </svg>
              Processing...
            </span>
          ) : (
            "Upscale"
          )}
        </button>

        {resultUrl && (
          <div className="mt-8 w-full flex flex-col items-center">
            <h2 className="text-xl font-bold mb-3 text-blue-300">Upscaled Output:</h2>
            <div className="bg-gray-800 border-2 border-blue-700 rounded-lg p-3 shadow-lg">
              <img
                src={resultUrl}
                alt="Upscaled"
                className="max-w-xs max-h-80 rounded shadow-lg border border-gray-700"
              />
            </div>
            <a
              href={resultUrl}
              download="upscaled.png"
              className="mt-4 inline-block px-4 py-2 bg-blue-700 text-white rounded hover:bg-blue-800 transition"
            >
              Download Image
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default MainPage;
