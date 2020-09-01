package org.neutralmike;

import org.neutralmike.gym.*;
import org.tensorflow.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;

public class App
{
    public static void main( String[] args ) throws IOException {
        SavedModelBundle model = org.tensorflow.SavedModelBundle.load("src/main/resources/Qmodel", "serve");
        Graph graphDef = model.graph();
        Session tfSession = model.session();
//        Iterator iterator = graphDef.operations();
//        while (iterator.hasNext()) {
//            System.out.println(iterator.next());
//        }
        CartPole TestCart = new CartPole();
        for (int j = 0; j < 50; j++) {
            TestCart.Reset();
            int stepId = 0;
            while (!TestCart.isDone() && stepId < 5000) {
                Tensor output = tfSession.runner().fetch("StatefulPartitionedCall").feed("serving_default_input_1", Tensor.create(new float[][] {TestCart.getState()})).run().get(0);
                float[][] ot = new float[1][2];
                output.copyTo(ot);
//                        System.out.print(ot[0][0]);
//                        System.out.print("    ");
//                        System.out.println(ot[0][1]);
//                System.out.println(TestCart.getState()[0]);
                if(ot[0][0] > ot[0][1]){
                    TestCart.Step(0);
                } else {
                    TestCart.Step(1);
                }
                stepId++;

            }
            System.out.print("Done on step: ");
            System.out.println(stepId + 1);
        }
    }
}
