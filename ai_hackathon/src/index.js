import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import axiosInstance from "./helper/axios";

// axiosInstance.interceptors.request.use(
//     request => {
//         let basicToken = localStorage.getItem("token");
//         if (basicToken) {
//             request.headers['Authorization'] = basicToken;
//         }
//         return request;
//     },
//     error => {
//         return Promise.reject(error);
//     }
// );
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <App/>
    </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
