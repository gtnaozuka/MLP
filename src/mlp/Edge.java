package mlp;

public class Edge {

    private double weight;
    private Neuron neuronIn;

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public Neuron getNeuronIn() {
        return neuronIn;
    }

    public void setNeuronIn(Neuron neuronIn) {
        this.neuronIn = neuronIn;
    }
}
