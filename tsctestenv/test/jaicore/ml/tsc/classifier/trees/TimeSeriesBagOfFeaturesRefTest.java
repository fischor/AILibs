package jaicore.ml.tsc.classifier.trees;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import org.apache.commons.lang3.reflect.FieldUtils;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.core.exception.EvaluationException;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.tsc.classifier.SimplifiedTSClassifierTest;
import jaicore.ml.tsc.classifier.trees.TimeSeriesBagOfFeaturesClassifier;
import jaicore.ml.tsc.dataset.TimeSeriesDataset;
import jaicore.ml.tsc.exceptions.TimeSeriesLoadingException;
import jaicore.ml.tsc.util.ClassMapper;
import jaicore.ml.tsc.util.SimplifiedTimeSeriesLoader;
import timeseriesweka.classifiers.TSBF;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 * Reference tests for {@link TimeSeriesBagOfFeaturesClassifier} objects.
 * 
 * @author Julian Lienen
 *
 */
@SuppressWarnings("unused")
public class TimeSeriesBagOfFeaturesRefTest {
	private static final Logger LOGGER = LoggerFactory.getLogger(TimeSeriesBagOfFeaturesRefTest.class);

	private static final String UNIVARIATE_PREFIX = "D:\\Data\\TSC\\UnivariateTSCProblems\\";

	private static final String CAR_TRAIN = UNIVARIATE_PREFIX + "Car\\Car_TRAIN.arff";
	private static final String CAR_TEST = UNIVARIATE_PREFIX + "Car\\Car_TEST.arff";

	private static final String BEEF_TRAIN = UNIVARIATE_PREFIX + "Beef\\Beef_TRAIN.arff";
	private static final String BEEF_TEST = UNIVARIATE_PREFIX + "Beef\\Beef_TEST.arff";

	private static final String ARROW_HEAD_TRAIN = UNIVARIATE_PREFIX + "ArrowHead\\ArrowHead\\ArrowHead_TRAIN.arff";
	private static final String ARROW_HEAD_TEST = UNIVARIATE_PREFIX + "ArrowHead\\ArrowHead\\ArrowHead_TEST.arff";

	private static final String ITALY_POWER_DEMAND_TRAIN = UNIVARIATE_PREFIX
			+ "ItalyPowerDemand\\ItalyPowerDemand_TRAIN.arff";
	private static final String ITALY_POWER_DEMAND_TEST = UNIVARIATE_PREFIX
			+ "ItalyPowerDemand\\ItalyPowerDemand_TEST.arff";

	private static final String RACKET_SPORTS_TRAIN = UNIVARIATE_PREFIX + "RacketSports\\RacketSports_TRAIN.arff";
	private static final String RACKET_SPORTS_TEST = UNIVARIATE_PREFIX + "RacketSports\\RacketSports_TEST.arff";

	private static final String SYNTHETIC_CONTROL_TRAIN = UNIVARIATE_PREFIX
			+ "\\SyntheticControl\\SyntheticControl_TRAIN.arff";
	private static final String SYNTHETIC_CONTROL_TEST = UNIVARIATE_PREFIX
			+ "\\SyntheticControl\\SyntheticControl_TEST.arff";

	private static final String COMPUTERS_TRAIN = UNIVARIATE_PREFIX + "\\Computers\\Computers_TRAIN.arff";
	private static final String COMPUTERS_TEST = UNIVARIATE_PREFIX + "\\Computers\\Computers_TEST.arff";

	// @Test
	public void compareClassifierPredictions()
			throws TimeSeriesLoadingException, Exception {
		int seed = 30; // seedRandom.nextInt(100);

		Random random = new Random(seed);
		int numBins = 10; // 1 + random.nextInt(20); // As in the reference implementation
		int numFolds = 10; // 3 + random.nextInt(15); // As in the reference implementation
		double zProp = 0.1; // z[i % z.length];// 0.01 + random.nextDouble(); // As in the reference
							// implementation


		int minIntervalLength = 5; // 2 + random.nextInt(10); // As in the reference implementation

		TimeSeriesBagOfFeaturesClassifier ownClf = new TimeSeriesBagOfFeaturesClassifier(seed, numBins, numFolds, zProp,
				minIntervalLength);

		TSBF refClf = new TSBF();
		refClf.seedRandom(seed);
		FieldUtils.writeField(refClf, "stepWise", false, true);
		FieldUtils.writeField(refClf, "numReps", 1, true);
		refClf.setParamSearch(false);
		refClf.searchParameters(false);
		System.out.println("Wrote fields.");

		Pair<TimeSeriesDataset, ClassMapper> trainPair = SimplifiedTimeSeriesLoader
				.loadArff(new File(ITALY_POWER_DEMAND_TRAIN));
		TimeSeriesDataset train = trainPair.getX();
		ownClf.setClassMapper(trainPair.getY());
		Pair<TimeSeriesDataset, ClassMapper> testPair = SimplifiedTimeSeriesLoader
				.loadArff(new File(ITALY_POWER_DEMAND_TEST));
		TimeSeriesDataset test = testPair.getX();

		ArffReader arffReader = new ArffReader(new FileReader(new File(ITALY_POWER_DEMAND_TRAIN)));
		final Instances trainingInstances = arffReader.getData();
		trainingInstances.setClassIndex(trainingInstances.numAttributes() - 1);

		arffReader = new ArffReader(new FileReader(new File(ITALY_POWER_DEMAND_TEST)));
		final Instances testInstances = arffReader.getData();
		testInstances.setClassIndex(testInstances.numAttributes() - 1);

		ownClf.train(train);
		refClf.buildClassifier(trainingInstances);

		// Predict first instance


		System.out.println(refClf.classifyInstance(testInstances.get(0)));
		System.out.println(Arrays.toString(refClf.distributionForInstance(testInstances.get(0))));
		System.out.println(ownClf.predict(test.getValues(0)[0]));
	}

	@Test
	public void testClassifier() throws FileNotFoundException, EvaluationException, TrainingException,
			PredictionException, IOException, TimeSeriesLoadingException, ClassNotFoundException,
			IllegalAccessException {

		// TODO: Change this?
		org.apache.log4j.Logger.getLogger("jaicore").setLevel(org.apache.log4j.Level.DEBUG);

		// int seed = 42;
		// int numBins = 20; // As in the reference implementation
		// int numFolds = 20; // As in the reference implementation
		// double zProp = 1; // As in the reference implementation
		// int minIntervalLength = 5; // As in the reference implementation

		double currBest = 0;
		double[] z = new double[] { 0.1, 0.25, 0.5, 0.75 };

		int numTotalIterations = 1;
		Random seedRandom = new Random(42);
		for (int i = 0; i < numTotalIterations; i++) {
			int seed = 30; // seedRandom.nextInt(100);

			Random random = new Random(seed);
			int numBins = 10; // 1 + random.nextInt(20); // As in the reference implementation
			int numFolds = 10; // 3 + random.nextInt(15); // As in the reference implementation
			double zProp = 0.7; // z[i % z.length];// 0.01 + random.nextDouble(); // As in the reference
								// implementation
			if (zProp > 1)
				zProp = 1d;
			int minIntervalLength = 5; // 2 + random.nextInt(10); // As in the reference implementation

			TimeSeriesBagOfFeaturesClassifier ownClf = new TimeSeriesBagOfFeaturesClassifier(seed, numBins, numFolds,
					zProp, minIntervalLength);

			TSBF refClf = new TSBF();
			refClf.seedRandom(seed);
			refClf.setZLevel(zProp);
			// FieldUtils.writeField(refClf, "stepWise", false, true);
			FieldUtils.writeField(refClf, "numReps", 1, true);
			refClf.setParamSearch(false);
			refClf.searchParameters(false);
			System.out.println("Wrote fields.");

			Map<String, Object> result = SimplifiedTSClassifierTest.compareClassifiers(refClf, ownClf, seed, null, null,
					new File(ITALY_POWER_DEMAND_TRAIN), new File(ITALY_POWER_DEMAND_TEST));
			if (((double) result.get("accuracy")) > currBest) {
				currBest = ((double) result.get("accuracy"));
				LOGGER.info(
						"New best score {} with numBins {}, numFolds {}, zProp {} and minIntervalLength {} (seed {}).",
						currBest, numBins, numFolds, zProp, minIntervalLength, seed);
			}
			
			System.out.println(String.format("subSeries = %s.", Arrays.deepToString(ownClf.getSubseries())));
			System.out.println(String.format("intervals = %s.", Arrays.deepToString(ownClf.getIntervals())));
			System.out.println(
					"Ref subseries: "
							+ Arrays.deepToString((int[][]) FieldUtils.readDeclaredField(refClf, "subSeries", true)));
			System.out.println(
					"Ref intervals: "
							+ Arrays.deepToString((int[][][]) FieldUtils.readDeclaredField(refClf, "intervals", true)));

			// System.out.println("own clf performance: " + result.get("accuracy"));
			// System.out.println("ref clf performance: " + result.get("ref_accuracy"));

			if (i % 100 == 0)
				LOGGER.info("{}/{}", i, numTotalIterations);
		}

		LOGGER.info("Final best score: {}", currBest);

		// System.out.println("Ref clf parameters: " + refClf.getParameters());
		// System.out.println(result.toString());
	}

	public static void main(String[] args) throws IllegalAccessException, FileNotFoundException, EvaluationException,
			TrainingException, PredictionException, IOException, TimeSeriesLoadingException {
		// TODO: Change this?
		org.apache.log4j.Logger.getLogger("jaicore").setLevel(org.apache.log4j.Level.INFO);

		// int seed = 42;
		// int numBins = 20; // As in the reference implementation
		// int numFolds = 20; // As in the reference implementation
		// double zProp = 1; // As in the reference implementation
		// int minIntervalLength = 5; // As in the reference implementation

		double currBest = 0;
		double[] z = new double[] { 0.1, 0.25, 0.5, 0.75 };

		int numTotalIterations = 1000;
		Random seedRandom = new Random(42);
		for (int i = 0; i < numTotalIterations; i++) {
			int seed = seedRandom.nextInt(100);

			Random random = new Random(seed);
			int numBins = 10; // 1 + random.nextInt(20); // As in the reference implementation
			int numFolds = 10; // 3 + random.nextInt(15); // As in the reference implementation
			double zProp = z[i % z.length];// 0.01 + random.nextDouble(); // As in the reference implementation
			if (zProp > 1)
				zProp = 1d;
			int minIntervalLength = 5; // 2 + random.nextInt(10); // As in the reference implementation

			TimeSeriesBagOfFeaturesClassifier ownClf = new TimeSeriesBagOfFeaturesClassifier(seed, numBins, numFolds,
					zProp, minIntervalLength);

			TSBF refClf = new TSBF();
			refClf.seedRandom(seed);
			FieldUtils.writeField(refClf, "stepWise", false, true);
			FieldUtils.writeField(refClf, "numReps", 1, true);
			refClf.setParamSearch(false);
			refClf.searchParameters(false);
			System.out.println("Wrote fields.");

			Map<String, Object> result = SimplifiedTSClassifierTest.compareClassifiers(refClf, ownClf, seed, null, null,
					new File(ITALY_POWER_DEMAND_TRAIN), new File(ITALY_POWER_DEMAND_TEST));
			if (((double) result.get("accuracy")) > currBest) {
				currBest = ((double) result.get("accuracy"));
				LOGGER.info(
						"New best score {} with numBins {}, numFolds {}, zProp {} and minIntervalLength {} (seed {}).",
						currBest, numBins, numFolds, zProp, minIntervalLength, seed);
			}

			if (i % 100 == 0)
				LOGGER.info("{}/{}", i, numTotalIterations);
		}

		LOGGER.info("Final best score: {}", currBest);

		// System.out.println("Ref clf parameters: " + refClf.getParameters());
		// System.out.println(result.toString());
	}
}
