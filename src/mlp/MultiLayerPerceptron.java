package mlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiLayerPerceptron {

    private final int[] architecture;
    private final List<Neuron[]> layers;

    private static final Random RANDOM = new Random();
    private static final double MAX_THETA = 1.0, MIN_THETA = 0.0;
    private final double MAX_WEIGHT, MIN_WEIGHT;
    private static final double LAMBDA = 1.0, ALPHA = 1.0;

    public MultiLayerPerceptron(int[] architecture) {
        this.architecture = architecture;

        this.MAX_WEIGHT = 2.4 / (double) this.architecture[0];
        this.MIN_WEIGHT = -MAX_WEIGHT;

        layers = new ArrayList<>();
        for (int i = 0; i < this.architecture.length; i++) {
            Neuron[] neurons = new Neuron[this.architecture[i]];
            for (int j = 0; j < this.architecture[i]; j++) {
                neurons[j] = new Neuron();
                if (i != 0) {
                    neurons[j].setTheta(RANDOM.nextDouble() * (MAX_THETA - MIN_THETA)
                            + MIN_THETA);

                    List<Edge> edges = new ArrayList<>();
                    for (int k = 0; k < this.architecture[i - 1]; k++) {
                        Edge edge = new Edge();
                        edge.setWeight(RANDOM.nextDouble() * (MAX_WEIGHT - MIN_WEIGHT)
                                + MIN_WEIGHT);
                        edge.setNeuronIn(layers.get(i - 1)[k]);
                        edges.add(edge);
                    }
                    neurons[j].setEdges(edges);
                }
            }
            layers.add(neurons);
        }
    }

    public void train(int[][] truthTable) {
        for (int i = 0; i < truthTable.length; i++) {
            for (int j = 0; j < architecture[0]; j++) {
                layers.get(0)[j].setY(truthTable[i][j]);
            }
            backpropagation(truthTable, i);
        }
    }

    private void backpropagation(int[][] truthTable, int index) {
        for (int i = 1; i < architecture.length; i++) {
            Neuron[] neurons = layers.get(i);
            for (int j = 0; j < architecture[i]; j++) {
                double sum = 0.0;
                for (Edge edge : neurons[j].getEdges()) {
                    sum += edge.getNeuronIn().getY() * edge.getWeight();
                }
                neurons[j].setX(sum - neurons[j].getTheta());
                if (i != architecture.length - 1) {
                    neurons[j].setY(sigmoide(neurons[j].getX()));
                }
            }
        }
        Neuron neuronK = layers.get(architecture.length - 1)[0];
        double yK = neuronK.getX();
        double error = truthTable[index][architecture[0]] - yK;

        double deltaK = yK * (1 - yK) * error;
        neuronK.getEdges().stream().forEach((edge) -> {
            edge.setWeight(edge.getWeight() + ALPHA * edge.getNeuronIn().getY()
                    * deltaK);
        });
        for (int i = architecture.length - 2; i > 0; i--) {
            Neuron[] neurons = layers.get(i);
            for (int j = 0; j < architecture[i]; j++) {
                double yJ = neurons[j].getY();

                double sum = 0.0;
                for (int k = 0; k < architecture[i + 1]; k++) {
                    sum += findWeight(neurons[j], i + 1, k) * deltaK;
                }

                double deltaJ = yJ * (1 - yJ) * sum;
                neurons[j].getEdges().stream().forEach((edge) -> {
                    edge.setWeight(edge.getWeight() + ALPHA * edge.getNeuronIn().getY()
                            * deltaJ);
                });
            }
        }
    }

    private static double sigmoide(double x) {
        return (1.0 / (1.0 + Math.exp(-LAMBDA * x)));
    }

    private double findWeight(Neuron neuron, int neuronArrayIndex, int neuronIndex) {
        for (Edge edge : layers.get(neuronArrayIndex)[neuronIndex].getEdges()) {
            if (edge.getNeuronIn().equals(neuron)) {
                return edge.getWeight();
            }
        }
        return 0.0;
    }

    public void test(int[][] truthTable) {
        for (int[] line : truthTable) {
            for (int j = 0; j < architecture[0]; j++) {
                layers.get(0)[j].setY(line[j]);
                System.out.print(layers.get(0)[j].getY() + "\t");
            }

            for (int j = 1; j < architecture.length; j++) {
                Neuron[] neurons = layers.get(j);
                for (int k = 0; k < architecture[j]; k++) {
                    double sum = 0.0;
                    for (Edge edge : neurons[k].getEdges()) {
                        sum += edge.getNeuronIn().getY() * edge.getWeight();
                    }
                    neurons[k].setX(sum - neurons[k].getTheta());
                    if (j != architecture.length - 1) {
                        neurons[k].setY(sigmoide(neurons[k].getX()));
                    }
                }
            }
            System.out.println(layers.get(architecture.length - 1)[0].getX());
        }
    }
}
