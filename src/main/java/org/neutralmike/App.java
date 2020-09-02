package org.neutralmike;

import org.neutralmike.gym.*;
import org.tensorflow.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.OptionalDouble;

public class App
{
    public static void main( String[] args ) throws IOException {

        SavedModelBundle model = org.tensorflow.SavedModelBundle.load("src/main/resources/ActorCriticModel", "serve");
        Session tfSession = model.session();
        CartPole TestCart = new CartPole();
        float rewardLimit = 3000.0f;
        int tries = 50;
        ArrayList<Float> rewards = new ArrayList<>();

        System.out.println(tries + " tries to balance the pole by actor critic reinforcement learning model from keras examples");
        System.out.println("Stop limit: " + (int)rewardLimit);

        for (int j = 0; j < tries; j++) {
            TestCart.Reset();
            float reward = 0.0f;
            while (!TestCart.isDone() && reward < rewardLimit) {
                try (Tensor output = tfSession.runner().fetch("StatefulPartitionedCall").feed("serving_default_input_1", Tensor.create(new float[][] {TestCart.getState()})).run().get(0);) {
                    float[][] ot = new float[1][2];
                    output.copyTo(ot);
                    if(ot[0][0] > ot[0][1]){
                        TestCart.Step(0);
                    } else {
                        TestCart.Step(1);
                    }
                    reward += TestCart.getReward();
                }
            }
            rewards.add(reward);
        }

        OptionalDouble averageReward = rewards.stream().mapToDouble(a -> a).average();
        OptionalDouble minReward = rewards.stream().mapToDouble(a -> a).min();
        OptionalDouble maxReward = rewards.stream().mapToDouble(a -> a).max();

        System.out.println("Rewards: " + Arrays.toString(rewards.toArray()));
        System.out.println("Average: " + (averageReward.isPresent() ? averageReward.getAsDouble() : 0));
        System.out.println("Min: " + (minReward.isPresent() ? minReward.getAsDouble() : 0));
        System.out.println("Max: " + (maxReward.isPresent() ? maxReward.getAsDouble() : 0));
    }
}
