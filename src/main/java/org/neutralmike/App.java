package org.neutralmike;

import org.neutralmike.gym.*;
import org.tensorflow.*;

import java.io.IOException;

public class App
{
    public static void main( String[] args ) throws IOException {

        SavedModelBundle model = org.tensorflow.SavedModelBundle.load("src/main/resources/Qmodel", "serve");
        Session tfSession = model.session();
        CartPole TestCart = new CartPole();

        System.out.println("50 tries to balance the pole by actor critic reinforcement learning model from keras examples");

        for (int j = 0; j < 50; j++) {
            TestCart.Reset();
            float reward = 0.0f;
            while (!TestCart.isDone() && reward < 5000.0f) {
                Tensor output = tfSession.runner().fetch("StatefulPartitionedCall").feed("serving_default_input_1", Tensor.create(new float[][] {TestCart.getState()})).run().get(0);
                float[][] ot = new float[1][2];
                output.copyTo(ot);
                if(ot[0][0] > ot[0][1]){
                    TestCart.Step(0);
                } else {
                    TestCart.Step(1);
                }
                reward += TestCart.getReward();

            }
            System.out.print("Reward: ");
            System.out.print(reward);
            System.out.println("/5000");
        }
    }
}
