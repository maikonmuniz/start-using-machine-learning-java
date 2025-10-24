import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.util.ArrayList;

public class WekaJavaPureExample {
    public static void main(String[] args) throws Exception {
        ArrayList<Attribute> attributes = new ArrayList<>();

        Attribute age = new Attribute("age");
        Attribute income = new Attribute("income");

        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("yes");
        classValues.add("no");
        Attribute purchased = new Attribute("purchased", classValues);

        attributes.add(age);
        attributes.add(income);
        attributes.add(purchased);

        Instances data = new Instances("Clients", attributes, 0);
        data.setClassIndex(data.numAttributes() - 1);

        double[] example1 = {25, 2500, data.attribute("purchased").indexOfValue("no")};
        double[] example2 = {40, 6000, data.attribute("purchased").indexOfValue("yes")};
        double[] example3 = {30, 3200, data.attribute("purchased").indexOfValue("no")};
        double[] example4 = {45, 8000, data.attribute("purchased").indexOfValue("yes")};
        double[] example5 = {35, 5000, data.attribute("purchased").indexOfValue("yes")};

        data.add(new DenseInstance(1.0, example1));
        data.add(new DenseInstance(1.0, example2));
        data.add(new DenseInstance(1.0, example3));
        data.add(new DenseInstance(1.0, example4));
        data.add(new DenseInstance(1.0, example5));

        J48 tree = new J48();
        tree.buildClassifier(data);

        System.out.println("=== Trained Decision Tree ===");
        System.out.println(tree);

        double[] newClient = {38, 4800, 0};
        DenseInstance newInstance = new DenseInstance(1.0, newClient);
        newInstance.setDataset(data);

        double result = tree.classifyInstance(newInstance);
        String predictedClass = data.classAttribute().value((int) result);

        System.out.println("\nPrediction for client: " + predictedClass);
    }
}
