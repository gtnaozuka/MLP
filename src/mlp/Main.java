package mlp;

public class Main {

    private static final int[] ARCHITECTURE = {2, 2, 1};

    private static final int MAX_TRAIN = 10000;
    private static final int[][] TRUTH_TABLE = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };

    public static void main(String[] args) {
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(ARCHITECTURE);
        for (int i = 0; i < MAX_TRAIN; i++) {
            mlp.train(TRUTH_TABLE);
        }
        mlp.test(TRUTH_TABLE);
    }
}
