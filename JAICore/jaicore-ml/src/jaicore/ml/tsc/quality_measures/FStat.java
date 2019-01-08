package jaicore.ml.tsc.quality_measures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

public class FStat implements IQualityMeasure {
	/**
	 * Generated serial version UID.
	 */
	private static final long serialVersionUID = 6991529180002046551L;

	@Override
	public double assessQuality(final List<Double> distances, final INDArray classValues) {
		HashMap<Integer, List<Double>> classDistances = new HashMap<>();
		for (int i = 0; i < distances.size(); i++) {
			if (!classDistances.containsKey(classValues.getInt(i)))
				classDistances.put(classValues.getInt(i), new ArrayList<>());

			classDistances.get(classValues.getInt(i)).add(distances.get(i));
		}

		int numClasses = classDistances.size();

		double result = 0;

		// Calculate class means
		HashMap<Integer, Double> classMeans = new HashMap<>();
		for (Integer clazz : classDistances.keySet()) {
			classMeans.put(clazz, classDistances.get(clazz).stream().mapToDouble(a -> a).average().getAsDouble());
		}
		double completeMean = distances.stream().mapToDouble(a -> a).average().getAsDouble();
		double denominator = 0;

		for (Integer clazz : classMeans.keySet()) {
			result += Math.pow(classMeans.get(clazz) - completeMean, 2);

			for (Double dist : classDistances.get(clazz)) {
				denominator += Math.pow(dist - classMeans.get(clazz), 2);
			}
		}
		result /= numClasses - 1;
		denominator /= distances.size() - numClasses;
		result /= denominator;

		return result;
	}
}