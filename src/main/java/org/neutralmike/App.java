package org.neutralmike;
import org.neutralmike.gym.*;

public class App
{
    public static void main( String[] args )
    {
        CartPole TestCart = new CartPole();
        for (int j = 0; j<50; j++) {
            TestCart.Reset();
            for (int i = 0; i< 500; i++) {
                int action = Policy(TestCart.getState());
                TestCart.Step(action);
                if(TestCart.isDone() || i == 499) {
                    System.out.print("Done on step: ");
                    System.out.println(i+1);
                    break;
                }
            }
        }
    }

    static int Policy(double[] obs) {
        if (obs[3] > 0) return 1;
        return 0;
    }
}
