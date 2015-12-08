package mlp;

public class Main {

    private static final int[] ARCHITECTURE = {2, 2, 1};

    private static final double TOLERANCE = 0.1, MAX_ITERATIONS = 1000000;
    private static final int[][] TRUTH_TABLE = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(ARCHITECTURE);

        double[] out;
        double maxError;
        int iterations = 0;
        do {
            mlp.train(TRUTH_TABLE);
            out = mlp.test(TRUTH_TABLE);
            maxError = calculeMaxError(out);

            iterations++;
        } while (maxError > TOLERANCE && iterations < MAX_ITERATIONS);

        if (iterations == MAX_ITERATIONS) {
            System.out.println("THERE WAS NOT CONVERGENCE.");
            return;
        }
        
        System.out.println("Iterations: " + iterations);
        for (int i = 0; i < TRUTH_TABLE.length; i++) {
            for (int j = 0; j < TRUTH_TABLE[i].length; j++) {
                System.out.print(TRUTH_TABLE[i][j] + "\t");
            }
            System.out.println(out[i]);
        }
    }

    private static double calculeMaxError(double[] out) {
        double maxError = 0.0;
        for (int i = 0; i < TRUTH_TABLE.length; i++) {
            maxError = Math.max(maxError, Math.abs(TRUTH_TABLE[i][TRUTH_TABLE[i].length - 1]
                    - out[i]));
        }
        return maxError;
    }
}
