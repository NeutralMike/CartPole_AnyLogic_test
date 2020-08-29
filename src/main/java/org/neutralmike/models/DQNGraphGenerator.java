package org.neutralmike.models;
import org.tensorflow.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DQNGraphGenerator {
    public static void Generate(int numActions, int numStates, int stateDiscretizationNumber){
        try (Graph tfGraph = new Graph();) {
            Operation x = tfGraph.opBuilder("Placeholder", "x")
                    .setAttr("dtype", DataType.FLOAT)
                    .build();
            Operation y = tfGraph.opBuilder("Placeholder", "y")
                    .setAttr("dtype", DataType.FLOAT)
                    .build();
            Operation mul = tfGraph.opBuilder("Mul", "mul")
                .addInput(x.output(0))
                .addInput(y.output(0))
                .build();
//            try (Session tfSession = new Session(tfGraph);
//                 Tensor output = tfSession.runner().fetch("mul").feed("x", Tensor.create(12.0f)).feed("y", Tensor.create(2.0f)).run().get(0);
//            ){
//                System.out.println(output.floatValue());
//            }
            byte[] readyGraph = tfGraph.toGraphDef();
            Files.write(Paths.get("src/main/resources/DQNgraph.pb"), readyGraph);
//            Operation QTable =  tfGraph.opBuilder("Placeholder", "states")
//                    .setAttr("dtype", DataType.FLOAT)
//                    .setAttr("shape", Shape.make(this.stateDiscretizationNumber, this.numActions, this.numStates))
//                    .build();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
