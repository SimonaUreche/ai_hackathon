import React from "react";
import {Avatar, ListItem, ListItemIcon, ListItemText} from "@mui/material";
import Button from "@mui/material/Button";
import {Edit} from "@mui/icons-material";
import OwnerDetails from "./OwnerDetails";

class OwnerItem extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            displayDetails: false,
        }
    };

    displayDetails = () => {
        this.setState({
            displayDetails: !this.state.displayDetails
        })
    }

    render() {
        return (
            <React.Fragment>
                <ListItem key={this.props.owner.id}>
                    <ListItemIcon>
                        <Avatar>{"O"}</Avatar>
                    </ListItemIcon>
                    <ListItemText
                        primary={this.props.owner.id + " " + this.props.owner.name}/>
                    <br/>
                    <Button onClick={() => this.displayDetails()}>
                        <Edit />DETAILS
                    </Button>
                    <div>
                        {this.state.displayDetails ? <OwnerDetails owner={this.props.owner} /> : null}
                    </div>
                </ListItem>
            </React.Fragment>
        )
    }
}

export default OwnerItem;