import React from "react";
import Button from "@mui/material/Button"
import TextField from "@mui/material/TextField";
import Container from "@mui/material/Container";
import axiosInstance from "./helper/axios";
import Grid from "@mui/material/Grid";
import history from './helper/history';


class Login extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            username: "",
            password: "",
            loginSuccess: {
                id: 0,
            }
        };
    }

    handleInput = event => {
        const {value, name} = event.target;
        this.setState({
            [name]: value
        });
        console.log(this.state);
    };

    onSubmitFunction = event => {
        event.preventDefault();
        let credentials = {
            username: this.state.username,
            password: this.state.password
        }

        axiosInstance.post("/login", credentials)
            .then(
                res => {
                    const val = res.data;
                    this.setState({
                        loginSuccess: val
                    });
                    if (val.id !== 0) {
                        localStorage.setItem("USER_ID", res.data.id);
                        history.push("/home");
                        window.location.reload();
                    }
                }
            )
            .catch(error => {
                console.log(error);
                alert("Invalid Credentials");
            })
    }


    render() {
        return (
            <Container maxWidth="sm">
                <div>
                    <Grid>
                        <form onSubmit={this.onSubmitFunction}>
                            <TextField
                                variant="outlined"
                                margin="normal"
                                required
                                fullWidth
                                id="username"
                                label="Username"
                                name="username"
                                autoComplete="string"
                                onChange={this.handleInput}
                                autoFocus
                            />
                            <TextField
                                variant="outlined"
                                margin="normal"
                                required
                                fullWidth
                                name="password"
                                label="Password"
                                type="password"
                                id="password"
                                onChange={this.handleInput}
                                autoComplete="current-password"
                            />
                            <Button
                                type="submit"
                                fullWidth
                                variant="contained"
                                color="primary"
                            >
                                Sign In
                            </Button>
                        </form>
                    </Grid>
                </div>
            </Container>
        );
    }

}

export default Login;