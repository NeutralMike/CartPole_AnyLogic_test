package org.neutralmike;

import org.neutralmike.gym.*;
import org.tensorflow.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class App
{
    public static void main( String[] args ) throws IOException {
        final byte[] graphDef = Files.readAllBytes(Paths.get("src/main/resources/DQNgraph.pb"));
        try (Graph tfGraph = new Graph()) {
            tfGraph.importGraphDef(graphDef);
            try (Session tfSession = new Session(tfGraph);
                 Tensor output = tfSession.runner().fetch("mul").feed("x", Tensor.create(12.0f)).feed("y", Tensor.create(2.0f)).run().get(0);
            ){
                System.out.println(output.floatValue());
            }
//            try (Session tfSession = new Session(tfGraph);
//                 Tensor output = tfSession.runner().fetch(a).run().get(0)) {
//                    System.out.println(output.intValue());
//                }
            }
        CartPole TestCart = new CartPole();
        for (int j = 0; j<50; j++)
        {
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

    public static <T> Output<T> addConstant(Graph g, String name, Object value) {
        try (Tensor<?> t = Tensor.create(value)) {
            return g.opBuilder("Const", name)
                    .setAttr("dtype", t.dataType())
                    .setAttr("value", t)
                    .build()
                    .<T>output(0);
        }
    }
    public static <T> Output<T> addAddOperation(Graph g, Output<?>... inputs) {
        return g.opBuilder("AddN", "TheBigAdder")
                .addInputList(inputs)
                .build()
                .<T>output(0);
    }
}
