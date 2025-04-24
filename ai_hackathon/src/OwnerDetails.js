import React from "react";

import {Avatar, List, ListItem, ListItemIcon, ListItemText} from "@mui/material";

class OwnerDetails extends React.Component {
    constructor(props) {
        super(props)
    };


    render() {
        return (
            <React.Fragment>
                <ListItem key={"details"+this.props.owner.id}>
                    <ListItemIcon>
                        <Avatar>{"U"}</Avatar>
                    </ListItemIcon>
                    <ListItemText
                        primary={this.props.owner.username}/>
                </ListItem>
                <ListItem key={this.props.owner.id}>
                    <ListItemIcon>
                        <Avatar>{"P"}</Avatar>
                    </ListItemIcon>
                    <ListItemText
                        primary={this.props.owner.password}/>
                </ListItem>
            </React.Fragment>
        )
    }
}

export default OwnerDetails;