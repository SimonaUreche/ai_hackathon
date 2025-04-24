import React from "react";
import axiosInstance from "./helper/axios";
import * as SockJS from 'sockjs-client';
import * as Stomp from 'stompjs';

import {List} from "@mui/material";
import OwnerItem from "./OwnerItem";


class Home extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            owners: [],
        }
    };

    connect() {
        const URL = "http://localhost:8080/socket";
        const websocket = new SockJS(URL);
        const stompClient = Stomp.over(websocket);
        stompClient.connect({}, frame => {
            stompClient.subscribe("/topic/socket/owner", notification => {
                let message = notification.body;
                console.log(message);
                alert(message);

            })
        })
    }

    componentDidMount() {
        if (!this.wsconnected) {
            this.connect();
            this.wsconnected = true;
        }

        axiosInstance
            .get(
                "/owner",
            )
            .then(res => {
                const val = res.data;
                this.setState({
                    owners: val
                });
            })
            .catch(error => {
                console.log(error);
            });
    };


    render() {
        return (
            <React.Fragment>
                <List key={"owners"}>
                    {this.state.owners.map(owner => (
                        // <ListItem key={owner.id}>
                        //     <ListItemIcon>
                        //         <Avatar>{"O"}</Avatar>
                        //     </ListItemIcon>
                        //     <ListItemText primary={owner.id + " " + owner.name}/>
                        // </ListItem>
                        <div>
                            <OwnerItem owner={owner}/>
                        </div>

                    ))}
                </List>
            </React.Fragment>
        )
    }
}

export default Home;