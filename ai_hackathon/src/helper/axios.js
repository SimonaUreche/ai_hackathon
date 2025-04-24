import axios from "axios"

const getBaseURL = () => {
    const path = window.location.pathname;
    if (path.includes("login")) {
        return "http://localhost:8082";
    }
    return "http://localhost:8080";
};

const axiosInstance = axios.create({
    baseURL: getBaseURL(),
    headers: {
        post: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers":
                "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With"
        }
    }
});

export default axiosInstance;