package org.neutralmike.gym;

public class CartPole {
    double gravity = 9.8;
    double masscart = 1.0;
    double masspole = 0.1;
    double totalMass = masscart + masspole;
    double length = 0.5;
    double polemassLength = masspole * length;
    double forceMag = 10.0;
    double tau = 0.02;
    double thetaThresholdRadians = 12 * 2 * Math.PI / 360;
    double xThreshold = 2.4;

    double x;
    double xDot;
    double theta;
    double thetaDot;

    int[] state;
    double reward;
    int stepsBeyondDone = -1;
    boolean done;

    public void Step(int action) {
        double force = forceMag;
        if (action == 0) {
            force = -force;
        }
        double cosTheta = Math.cos(theta);
        double sinTheta = Math.sin(theta);
        double temp = (force + polemassLength * Math.pow(thetaDot, 2) * sinTheta) / totalMass;
        double thetaacc = (gravity * sinTheta - cosTheta * temp) / (length * (4.0 / 3.0 - masspole * Math.pow(cosTheta,2) / totalMass));
        double xacc = temp - polemassLength * thetaacc * cosTheta / totalMass;

        x = x + tau * xDot;
        xDot = xDot + tau * xacc;
        theta = theta + tau * thetaDot;
        thetaDot = thetaDot + tau * thetaacc;
        updateState();
        done = (
                x < -xThreshold
                        || x > xThreshold
                        || theta < -thetaThresholdRadians
                        || theta > thetaThresholdRadians
        );
        if (!done) reward = 1.0;
        else if(stepsBeyondDone == -1) {
            stepsBeyondDone = 0;
            reward = 1.0;
        }
        else {
            stepsBeyondDone += 1;
            reward = 0.0;
        }
    }

    public void Reset() {
        x = Math.random() * 0.1 - 0.05;
        xDot = Math.random() * 0.1 - 0.05;
        theta = Math.random() * 0.1 - 0.05;
        thetaDot = Math.random() * 0.1 - 0.05;
        updateState();
        stepsBeyondDone = -1;
        done = false;
    }

    void updateState()
    {
        state = new int[] {discretize(x, xThreshold), discretize(xDot, 2*xThreshold), discretize(theta, thetaThresholdRadians), discretize(thetaDot, 2*thetaThresholdRadians)};
    }

    int discretize(double val,double threshold)
    {
        return (int) (12 * (val)/(threshold));
    }

    public int[] getState() {
        return state;
    }

    public double getReward() {
        return reward;
    }

    public boolean isDone() {
        return done;
    }

}