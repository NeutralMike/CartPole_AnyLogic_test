package org.neutralmike.gym;

public class CartPole {
    float gravity = 9.8f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float totalMass = masscart + masspole;
    float length = 0.5f;
    float polemassLength = masspole * length;
    float forceMag = 10.0f;
    float tau = 0.02f;
    float thetaThresholdRadians = 12 * 2 * (float) Math.PI / 360;
    float xThreshold = 2.4f;

    float x;
    float xDot;
    float theta;
    float thetaDot;

    float reward;
    int stepsBeyondDone = -1;
    boolean done;

    public void Step(int action) {
        float force = forceMag;
        if (action == 0) {
            force = -force;
        }
        float cosTheta = (float) Math.cos(theta);
        float sinTheta = (float) Math.sin(theta);
        float temp = (force + polemassLength * (float) Math.pow(thetaDot, 2) * sinTheta) / totalMass;
        float thetaacc = (gravity * sinTheta - cosTheta * temp) / (length * (4.0f / 3.0f - masspole * (float) Math.pow(cosTheta,2) / totalMass));
        float xacc = temp - polemassLength * thetaacc * cosTheta / totalMass;

        x = x + tau * xDot;
        xDot = xDot + tau * xacc;
        theta = theta + tau * thetaDot;
        thetaDot = thetaDot + tau * thetaacc;
        done = (
                x < -xThreshold
                        || x > xThreshold
                        || theta < -thetaThresholdRadians
                        || theta > thetaThresholdRadians
        );
        if (!done) reward = 1.0f;
        else if(stepsBeyondDone == -1) {
            stepsBeyondDone = 0;
            reward = 1.0f;
        }
        else {
            stepsBeyondDone += 1;
            reward = 0.0f;
        }
    }

    public void Reset() {
        x = (float) Math.random() * 0.1f - 0.05f;
        xDot = (float) Math.random() * 0.1f - 0.05f;
        theta = (float) Math.random() * 0.1f - 0.05f;
        thetaDot = (float) Math.random() * 0.1f - 0.05f;
        stepsBeyondDone = -1;
        done = false;
    }

    int discretize(float val,float threshold)
    {
        return (int) (24 * (val+threshold)/(2*threshold));
    }

    public float[] getState() {
        return new float[] {x,xDot,theta,thetaDot};
    }
    public int[] getDesretizedState() {
        return new int[] {discretize(x, xThreshold), discretize(xDot, 2*xThreshold), discretize(theta, thetaThresholdRadians), discretize(thetaDot, 2*thetaThresholdRadians)};
    }

    public double getReward() {
        return reward;
    }

    public boolean isDone() {
        return done;
    }

}