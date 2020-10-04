package org.neutralmike;
import org.neutralmike.gym.*;
import py4j.GatewayServer;

public class CartPoleEntryPoint
{
    private final CartPole cartPole;

    public CartPoleEntryPoint(){
        cartPole = new CartPole();
    }

    public CartPole getCartPole() {
        return cartPole;
    }

    public static void main( String[] args ) {
        GatewayServer gatewayServer = new GatewayServer(new CartPoleEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }
}
