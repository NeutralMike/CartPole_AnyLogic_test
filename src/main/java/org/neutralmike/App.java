package org.neutralmike;

import org.neutralmike.gym.*;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.Arrays;

public class App
{
    public static void main( String[] args )
    {
        Graph tfGraph = new Graph();
        Session tfSession = new Session(tfGraph);
        CartPole TestCart = new CartPole();
        for (int j = 0; j<50; j++)
        {
//            Q = (1 -learning_rate(i));
            TestCart.Reset();
            int stepId = 0;
            while(!TestCart.isDone())
            {
                int action = Policy(TestCart.getDesretizedState());
                TestCart.Step(action);
                stepId++;
            }
            System.out.print("Done on step: ");
            System.out.println(stepId+1);
        }
    }

    static int Policy(int[] obs) {
        if (obs[3] > 0) return 1;
        return 0;
    }

    static double learning_rate(int n ,double min_rate)
    {
        return (int) Arrays.stream(new double[] {min_rate, Arrays.stream(new double [] {1.0, 1.0 - Math.log10(((double)n + 1.0) / 25.0)}).min().getAsDouble()}).max().getAsDouble();
    }
}
